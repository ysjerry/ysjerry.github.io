#!/usr/bin/env python3
"""
Basic Multi-Sensor Fusion Example

This example demonstrates how to use the PyTorch Kalman Filter system
to fuse data from multiple sensors for vehicle state estimation.
"""

import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kalman_filter import MultiSensorFusion
from kalman_filter.utils.simulation import generate_simulated_data, generate_ground_truth_trajectory
from kalman_filter.utils.preprocessing import synchronize_sensor_data


def run_basic_fusion_demo(duration=60.0, scenario='urban_driving'):
    """
    Run a basic fusion demonstration with simulated data.
    
    Args:
        duration: Simulation duration in seconds
        scenario: Driving scenario to simulate
    """
    print(f"Running PyTorch Kalman Filter Multi-Sensor Fusion Demo")
    print(f"Scenario: {scenario}, Duration: {duration}s")
    print("-" * 60)
    
    # Generate simulated sensor data
    print("Generating simulated sensor data...")
    sensor_data = generate_simulated_data(
        duration=duration,
        dt=0.1,
        scenario=scenario,
        add_noise=True,
        device='cpu'
    )
    
    # Generate ground truth for comparison
    ground_truth = generate_ground_truth_trajectory(sensor_data)
    
    print(f"Generated data for {len(sensor_data)} sensor types:")
    for sensor_type, data in sensor_data.items():
        print(f"  {sensor_type}: {len(data)} samples")
    
    # Initialize fusion system
    print("\nInitializing fusion system...")
    
    # Use GPU if available
    device = 'cuda' if hasattr(os.environ.get('CUDA_VISIBLE_DEVICES', ''), '__iter__') else 'cpu'
    print(f"Using device: {device}")
    
    # Get reference point from first GPS sample
    gps_ref = None
    if sensor_data['gps']:
        first_gps = sensor_data['gps'][0]
        gps_ref = (first_gps['latitude'], first_gps['longitude'])
    
    # Initialize fusion system
    fusion = MultiSensorFusion(
        device=device,
        gps_reference_point=gps_ref,
        update_rates={
            'gps': 1.0,
            'imu': 10.0,
            'wheel_speed': 5.0,
            'steering': 5.0
        }
    )
    
    # Synchronize sensor data
    print("Synchronizing sensor data...")
    synchronized_data = synchronize_sensor_data(sensor_data, time_window=0.1)
    print(f"Synchronized to {len(synchronized_data)} time points")
    
    # Run fusion
    print("\nRunning sensor fusion...")
    results = []
    
    start_time = time.time()
    for i, sync_point in enumerate(synchronized_data):
        timestamp = sync_point['timestamp']
        
        # Prepare sensor data for this time point
        current_sensors = {}
        for sensor_type in ['gps', 'imu', 'wheel_speed', 'steering']:
            if sensor_type in sync_point:
                current_sensors[sensor_type] = sync_point[sensor_type]
        
        # Update fusion system
        result = fusion.update_with_sensor_data(current_sensors, timestamp)
        results.append(result)
        
        # Print progress
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(synchronized_data)} time points")
    
    processing_time = time.time() - start_time
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print(f"Average processing rate: {len(synchronized_data)/processing_time:.1f} Hz")
    
    # Analyze results
    print("\nAnalyzing results...")
    analyze_fusion_results(results, ground_truth, fusion)
    
    # Plot results
    plot_fusion_results(results, ground_truth, sensor_data)
    
    return results, ground_truth, fusion


def analyze_fusion_results(results, ground_truth, fusion):
    """Analyze fusion results and print statistics."""
    
    # Filter initialized results
    initialized_results = [r for r in results if r.get('initialized', False)]
    
    if not initialized_results:
        print("Warning: No initialized results found!")
        return
    
    print(f"Initialization: {len(initialized_results)}/{len(results)} samples processed")
    
    # Calculate position errors
    position_errors = []
    for result in initialized_results:
        result_time = result['timestamp']
        
        # Find closest ground truth
        closest_truth = min(ground_truth, 
                          key=lambda x: abs(x['timestamp'] - result_time),
                          default=None)
        
        if closest_truth:
            # Calculate error in meters (approximate)
            lat_error = (result['gps_coordinates']['latitude'] - closest_truth['latitude']) * 111000
            lon_error = (result['gps_coordinates']['longitude'] - closest_truth['longitude']) * 111000 * np.cos(np.radians(closest_truth['latitude']))
            
            error = np.sqrt(lat_error**2 + lon_error**2)
            position_errors.append(error)
    
    if position_errors:
        print(f"\nPosition Accuracy:")
        print(f"  Mean error: {np.mean(position_errors):.2f} meters")
        print(f"  RMS error: {np.sqrt(np.mean(np.square(position_errors))):.2f} meters")
        print(f"  Max error: {np.max(position_errors):.2f} meters")
        print(f"  95th percentile: {np.percentile(position_errors, 95):.2f} meters")
    
    # Print sensor update statistics
    final_diagnostics = initialized_results[-1].get('diagnostics', {})
    if 'update_counts' in final_diagnostics:
        print(f"\nSensor Update Counts:")
        for sensor, count in final_diagnostics['update_counts'].items():
            print(f"  {sensor}: {count}")
    
    if 'actual_update_rates' in final_diagnostics:
        print(f"\nActual Update Rates:")
        for sensor, rate in final_diagnostics['actual_update_rates'].items():
            print(f"  {sensor}: {rate:.2f} Hz")
    
    # Print sensor information
    sensor_info = fusion.get_sensor_info()
    print(f"\nSensor Configuration:")
    for sensor_type, info in sensor_info.items():
        if sensor_type != 'fusion_parameters':
            print(f"  {sensor_type}: {info.get('sensor_type', 'Unknown')}")


def plot_fusion_results(results, ground_truth, sensor_data):
    """Plot fusion results for visualization."""
    try:
        # Filter initialized results
        initialized_results = [r for r in results if r.get('initialized', False)]
        
        if not initialized_results:
            print("No initialized results to plot")
            return
        
        # Extract data for plotting
        timestamps = [r['timestamp'] for r in initialized_results]
        fused_lats = [r['gps_coordinates']['latitude'] for r in initialized_results]
        fused_lons = [r['gps_coordinates']['longitude'] for r in initialized_results]
        
        truth_times = [gt['timestamp'] for gt in ground_truth]
        truth_lats = [gt['latitude'] for gt in ground_truth]
        truth_lons = [gt['longitude'] for gt in ground_truth]
        
        # Raw GPS data
        gps_times = [gps['timestamp'] for gps in sensor_data['gps']]
        gps_lats = [gps['latitude'] for gps in sensor_data['gps']]
        gps_lons = [gps['longitude'] for gps in sensor_data['gps']]
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trajectory plot
        ax1.plot(truth_lons, truth_lats, 'g-', linewidth=2, label='Ground Truth', alpha=0.7)
        ax1.scatter(gps_lons, gps_lats, c='r', s=20, alpha=0.6, label='Raw GPS')
        ax1.plot(fused_lons, fused_lats, 'b-', linewidth=2, label='Fused Estimate')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude') 
        ax1.set_title('Vehicle Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Latitude over time
        ax2.plot(truth_times, truth_lats, 'g-', linewidth=2, label='Ground Truth', alpha=0.7)
        ax2.scatter(gps_times, gps_lats, c='r', s=10, alpha=0.6, label='Raw GPS')
        ax2.plot(timestamps, fused_lats, 'b-', linewidth=1, label='Fused Estimate')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Latitude vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Longitude over time
        ax3.plot(truth_times, truth_lons, 'g-', linewidth=2, label='Ground Truth', alpha=0.7)
        ax3.scatter(gps_times, gps_lons, c='r', s=10, alpha=0.6, label='Raw GPS')
        ax3.plot(timestamps, fused_lons, 'b-', linewidth=1, label='Fused Estimate')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Longitude')
        ax3.set_title('Longitude vs Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Speed comparison
        fused_speeds = []
        for r in initialized_results:
            vx, vy, vz = r['velocity']['vx'], r['velocity']['vy'], r['velocity']['vz']
            speed = np.sqrt(vx**2 + vy**2)
            fused_speeds.append(speed)
        
        truth_speeds = [gt['speed'] for gt in ground_truth]
        gps_speeds = [gps['speed'] for gps in sensor_data['gps'] if gps['speed'] is not None]
        gps_speed_times = [gps['timestamp'] for gps in sensor_data['gps'] if gps['speed'] is not None]
        
        ax4.plot(truth_times, truth_speeds, 'g-', linewidth=2, label='Ground Truth', alpha=0.7)
        if gps_speeds:
            ax4.scatter(gps_speed_times, gps_speeds, c='r', s=10, alpha=0.6, label='Raw GPS')
        ax4.plot(timestamps, fused_speeds, 'b-', linewidth=1, label='Fused Estimate')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Speed (m/s)')
        ax4.set_title('Speed vs Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fusion_results.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved as 'fusion_results.png'")
        
        # Try to display if in interactive environment
        try:
            plt.show()
        except:
            pass
            
    except Exception as e:
        print(f"Plotting error: {e}")


def main():
    """Main function to run the demonstration."""
    print("PyTorch Kalman Filter Multi-Sensor Fusion Demo")
    print("=" * 60)
    
    # Run different scenarios
    scenarios = ['urban_driving', 'highway']
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Running scenario: {scenario}")
        print('='*60)
        
        try:
            results, ground_truth, fusion = run_basic_fusion_demo(
                duration=30.0,  # Shorter duration for demo
                scenario=scenario
            )
            
            print(f"Successfully completed {scenario} scenario")
            
        except Exception as e:
            print(f"Error in {scenario} scenario: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDemo completed!")


if __name__ == '__main__':
    main()