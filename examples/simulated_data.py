#!/usr/bin/env python3
"""
Simulated Data Testing Example

This example demonstrates the Kalman filter with various simulated
sensor data scenarios for validation and testing.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kalman_filter import MultiSensorFusion
from kalman_filter.utils.simulation import (
    generate_simulated_data, 
    create_test_scenarios,
    add_sensor_failures,
    generate_ground_truth_trajectory
)
from kalman_filter.utils.preprocessing import synchronize_sensor_data


def test_scenario(scenario_name, scenario_params, fusion_system):
    """Test a specific scenario with the fusion system."""
    
    print(f"\nTesting scenario: {scenario_name}")
    print(f"Description: {scenario_params['description']}")
    
    # Generate data for this scenario
    data = generate_simulated_data(
        duration=scenario_params['duration'],
        dt=scenario_params['dt'],
        scenario='urban_driving',  # Base scenario
        add_noise=True
    )
    
    # Add sensor failures for robustness testing
    if 'add_failures' in scenario_params and scenario_params['add_failures']:
        data = add_sensor_failures(data, failure_probability=0.1, failure_duration=2.0)
        print("  Added sensor failures for robustness testing")
    
    # Generate ground truth
    ground_truth = generate_ground_truth_trajectory(data)
    
    # Synchronize data
    sync_data = synchronize_sensor_data(data, time_window=0.05)
    
    # Reset fusion system
    fusion_system.reset()
    
    # Process data
    results = []
    start_time = time.time()
    
    for sync_point in sync_data:
        timestamp = sync_point['timestamp']
        
        # Prepare sensor data
        sensors = {}
        for sensor_type in ['gps', 'imu', 'wheel_speed', 'steering']:
            if sensor_type in sync_point:
                sensors[sensor_type] = sync_point[sensor_type]
        
        # Update fusion
        result = fusion_system.update_with_sensor_data(sensors, timestamp)
        results.append(result)
    
    processing_time = time.time() - start_time
    
    # Calculate metrics
    metrics = calculate_performance_metrics(results, ground_truth)
    metrics['processing_time'] = processing_time
    metrics['processing_rate'] = len(sync_data) / processing_time if processing_time > 0 else 0
    
    print(f"  Processed {len(sync_data)} samples in {processing_time:.3f}s")
    print(f"  Processing rate: {metrics['processing_rate']:.1f} Hz")
    
    return results, metrics


def calculate_performance_metrics(results, ground_truth):
    """Calculate performance metrics for fusion results."""
    
    # Filter initialized results
    init_results = [r for r in results if r.get('initialized', False)]
    
    if not init_results:
        return {
            'initialization_rate': 0.0,
            'position_error': {'mean': float('inf'), 'rms': float('inf'), 'max': float('inf')},
            'convergence_time': float('inf')
        }
    
    metrics = {
        'initialization_rate': len(init_results) / len(results) if results else 0.0,
        'total_samples': len(results),
        'initialized_samples': len(init_results)
    }
    
    # Position error analysis
    position_errors = []
    convergence_time = None
    
    for i, result in enumerate(init_results):
        # Find closest ground truth
        result_time = result['timestamp']
        closest_truth = min(ground_truth, 
                          key=lambda x: abs(x['timestamp'] - result_time),
                          default=None)
        
        if closest_truth:
            # Calculate position error in meters
            lat_diff = result['gps_coordinates']['latitude'] - closest_truth['latitude']
            lon_diff = result['gps_coordinates']['longitude'] - closest_truth['longitude']
            
            # Convert to meters (approximate)
            lat_error = lat_diff * 111000
            lon_error = lon_diff * 111000 * np.cos(np.radians(closest_truth['latitude']))
            
            error = np.sqrt(lat_error**2 + lon_error**2)
            position_errors.append(error)
            
            # Check for convergence (error < 5 meters for 10 consecutive samples)
            if convergence_time is None and len(position_errors) >= 10:
                recent_errors = position_errors[-10:]
                if all(e < 5.0 for e in recent_errors):
                    convergence_time = result_time
    
    if position_errors:
        metrics['position_error'] = {
            'mean': np.mean(position_errors),
            'rms': np.sqrt(np.mean(np.square(position_errors))),
            'max': np.max(position_errors),
            'std': np.std(position_errors),
            'percentile_95': np.percentile(position_errors, 95)
        }
    else:
        metrics['position_error'] = {
            'mean': float('inf'),
            'rms': float('inf'), 
            'max': float('inf')
        }
    
    metrics['convergence_time'] = convergence_time or float('inf')
    
    return metrics


def run_comprehensive_tests():
    """Run comprehensive tests with multiple scenarios."""
    
    print("Running Comprehensive Kalman Filter Tests")
    print("=" * 60)
    
    # Initialize fusion system
    device = 'cuda' if os.path.exists('/dev/nvidia0') else 'cpu'
    print(f"Using device: {device}")
    
    fusion = MultiSensorFusion(
        device=device,
        update_rates={
            'gps': 1.0,
            'imu': 10.0,
            'wheel_speed': 5.0,
            'steering': 5.0
        }
    )
    
    # Get test scenarios
    scenarios = create_test_scenarios()
    
    # Add custom test scenarios
    custom_scenarios = {
        'sensor_failure_test': {
            'description': 'Test robustness with sensor failures',
            'duration': 30.0,
            'dt': 0.1,
            'add_failures': True
        },
        
        'high_frequency_test': {
            'description': 'Test with high-frequency updates',
            'duration': 20.0,
            'dt': 0.05,  # Higher frequency
            'add_failures': False
        },
        
        'initialization_test': {
            'description': 'Test initialization robustness',
            'duration': 40.0,
            'dt': 0.1,
            'delayed_gps': True  # Start GPS after 5 seconds
        }
    }
    
    scenarios.update(custom_scenarios)
    
    # Results storage
    all_results = {}
    
    # Run tests
    for scenario_name, scenario_params in scenarios.items():
        try:
            results, metrics = test_scenario(scenario_name, scenario_params, fusion)
            all_results[scenario_name] = {
                'results': results,
                'metrics': metrics,
                'params': scenario_params
            }
            
            # Print metrics
            print(f"  Metrics:")
            print(f"    Initialization rate: {metrics['initialization_rate']:.1%}")
            if 'position_error' in metrics and metrics['position_error']['mean'] != float('inf'):
                print(f"    Mean position error: {metrics['position_error']['mean']:.2f}m")
                print(f"    RMS position error: {metrics['position_error']['rms']:.2f}m")
                print(f"    Max position error: {metrics['position_error']['max']:.2f}m")
            
            if metrics.get('convergence_time', float('inf')) != float('inf'):
                print(f"    Convergence time: {metrics['convergence_time']:.1f}s")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary report
    generate_test_report(all_results)
    
    # Create comparison plots
    create_comparison_plots(all_results)
    
    return all_results


def generate_test_report(all_results):
    """Generate a comprehensive test report."""
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST REPORT")
    print('='*60)
    
    # Summary table
    print(f"\n{'Scenario':<25} {'Init Rate':<10} {'Mean Err':<10} {'RMS Err':<10} {'Conv Time':<10}")
    print('-' * 70)
    
    for scenario_name, data in all_results.items():
        metrics = data['metrics']
        
        init_rate = f"{metrics['initialization_rate']:.1%}"
        
        pos_err = metrics.get('position_error', {})
        mean_err = f"{pos_err.get('mean', float('inf')):.2f}m" if pos_err.get('mean') != float('inf') else "N/A"
        rms_err = f"{pos_err.get('rms', float('inf')):.2f}m" if pos_err.get('rms') != float('inf') else "N/A"
        
        conv_time = metrics.get('convergence_time', float('inf'))
        conv_str = f"{conv_time:.1f}s" if conv_time != float('inf') else "N/A"
        
        print(f"{scenario_name:<25} {init_rate:<10} {mean_err:<10} {rms_err:<10} {conv_str:<10}")
    
    # Performance statistics
    valid_errors = []
    valid_conv_times = []
    
    for data in all_results.values():
        metrics = data['metrics']
        pos_err = metrics.get('position_error', {})
        if pos_err.get('mean') != float('inf'):
            valid_errors.append(pos_err['mean'])
        
        conv_time = metrics.get('convergence_time', float('inf'))
        if conv_time != float('inf'):
            valid_conv_times.append(conv_time)
    
    if valid_errors:
        print(f"\nOverall Performance Statistics:")
        print(f"  Average position error: {np.mean(valid_errors):.2f}m")
        print(f"  Best position error: {np.min(valid_errors):.2f}m")
        print(f"  Worst position error: {np.max(valid_errors):.2f}m")
    
    if valid_conv_times:
        print(f"  Average convergence time: {np.mean(valid_conv_times):.1f}s")
        print(f"  Fastest convergence: {np.min(valid_conv_times):.1f}s")
    
    # Processing performance
    total_samples = sum(data['metrics']['total_samples'] for data in all_results.values())
    total_time = sum(data['metrics']['processing_time'] for data in all_results.values())
    
    if total_time > 0:
        print(f"\nProcessing Performance:")
        print(f"  Total samples processed: {total_samples:,}")
        print(f"  Total processing time: {total_time:.2f}s")
        print(f"  Average processing rate: {total_samples/total_time:.1f} Hz")


def create_comparison_plots(all_results):
    """Create comparison plots for different scenarios."""
    
    try:
        # Create performance comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        scenario_names = []
        init_rates = []
        mean_errors = []
        rms_errors = []
        conv_times = []
        
        for scenario_name, data in all_results.items():
            if len(scenario_name) > 15:  # Shorten long names
                display_name = scenario_name[:12] + '...'
            else:
                display_name = scenario_name
                
            scenario_names.append(display_name)
            
            metrics = data['metrics']
            init_rates.append(metrics['initialization_rate'] * 100)
            
            pos_err = metrics.get('position_error', {})
            mean_errors.append(pos_err.get('mean', 0) if pos_err.get('mean') != float('inf') else 0)
            rms_errors.append(pos_err.get('rms', 0) if pos_err.get('rms') != float('inf') else 0)
            
            conv_time = metrics.get('convergence_time', 0)
            conv_times.append(conv_time if conv_time != float('inf') else 0)
        
        # Initialization rate
        ax1.bar(scenario_names, init_rates)
        ax1.set_title('Initialization Rate (%)')
        ax1.set_ylabel('Percentage')
        ax1.tick_params(axis='x', rotation=45)
        
        # Mean position error
        ax2.bar(scenario_names, mean_errors)
        ax2.set_title('Mean Position Error (m)')
        ax2.set_ylabel('Error (meters)')
        ax2.tick_params(axis='x', rotation=45)
        
        # RMS position error
        ax3.bar(scenario_names, rms_errors)
        ax3.set_title('RMS Position Error (m)')
        ax3.set_ylabel('Error (meters)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Convergence time
        nonzero_conv_times = [t if t > 0 else None for t in conv_times]
        ax4.bar(scenario_names, conv_times)
        ax4.set_title('Convergence Time (s)')
        ax4.set_ylabel('Time (seconds)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('test_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved as 'test_comparison.png'")
        
    except Exception as e:
        print(f"Plotting error: {e}")


def main():
    """Main function for comprehensive testing."""
    print("PyTorch Kalman Filter - Comprehensive Testing")
    print("=" * 60)
    
    try:
        results = run_comprehensive_tests()
        print("\nTesting completed successfully!")
        
        # Save results for later analysis
        # np.save('test_results.npy', results)
        # print("Results saved to 'test_results.npy'")
        
    except Exception as e:
        print(f"Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()