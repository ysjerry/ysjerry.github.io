"""
Simulation utilities for generating realistic sensor data.

Creates simulated multi-sensor data for testing and demonstration purposes.
"""

import torch
import numpy as np
import math
from typing import Dict, Any, List, Tuple, Optional
import random


def generate_simulated_data(duration: float = 60.0,
                          dt: float = 0.1,
                          scenario: str = 'urban_driving',
                          add_noise: bool = True,
                          device: str = 'cpu') -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate simulated multi-sensor data for a driving scenario.
    
    Args:
        duration: Simulation duration in seconds
        dt: Time step between samples
        scenario: Driving scenario ('urban_driving', 'highway', 'parking')
        add_noise: Whether to add realistic sensor noise
        device: PyTorch device
        
    Returns:
        Dictionary containing simulated data for all sensors
    """
    if scenario == 'urban_driving':
        return generate_urban_driving_data(duration, dt, add_noise, device)
    elif scenario == 'highway':
        return generate_highway_data(duration, dt, add_noise, device)
    elif scenario == 'parking':
        return generate_parking_data(duration, dt, add_noise, device)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def generate_urban_driving_data(duration: float = 60.0,
                              dt: float = 0.1,
                              add_noise: bool = True,
                              device: str = 'cpu') -> Dict[str, List[Dict[str, Any]]]:
    """Generate urban driving scenario with turns, stops, and acceleration."""
    
    # Initialize trajectory parameters
    x, y, z = 0.0, 0.0, 100.0  # Starting position (z is altitude)
    vx, vy, vz = 0.0, 0.0, 0.0  # Starting velocity
    roll, pitch, yaw = 0.0, 0.0, 0.0  # Starting orientation
    roll_rate, pitch_rate, yaw_rate = 0.0, 0.0, 0.0
    wheel_speed = 0.0
    steering_angle = 0.0
    
    # Reference GPS coordinates (San Francisco downtown)
    ref_lat, ref_lon = 37.7749, -122.4194
    
    # Data storage
    gps_data = []
    imu_data = []
    wheel_data = []
    steering_data = []
    
    # Simulation parameters
    time_steps = int(duration / dt)
    
    for i in range(time_steps):
        t = i * dt
        
        # Generate realistic driving profile
        if t < 10:  # Acceleration phase
            target_speed = min(t * 1.5, 15.0)  # Accelerate to 15 m/s (54 km/h)
            steering_angle = math.sin(t * 0.2) * 0.1  # Gentle steering
        elif t < 20:  # Straight driving
            target_speed = 15.0
            steering_angle = 0.05 * math.sin(t * 0.1)  # Small corrections
        elif t < 30:  # Turn and deceleration
            target_speed = max(15.0 - (t - 20) * 0.8, 5.0)
            steering_angle = 0.3 * math.sin((t - 20) * 0.5)  # Significant turn
        elif t < 40:  # Stop and go
            target_speed = 5.0 + 3.0 * math.sin((t - 30) * 0.8)
            steering_angle = 0.1 * math.cos(t * 0.3)
        elif t < 50:  # Another turn
            target_speed = 12.0
            steering_angle = -0.25 * math.sin((t - 40) * 0.7)
        else:  # Final deceleration
            target_speed = max(12.0 - (t - 50) * 1.2, 0.0)
            steering_angle = 0.0
        
        # Update wheel speed with some dynamics
        speed_diff = target_speed - wheel_speed
        wheel_speed += speed_diff * 0.1 * dt / 0.1  # Simple first-order dynamics
        
        # Calculate yaw rate from steering angle and speed
        wheel_base = 2.5  # meters
        if wheel_speed > 0.1:
            yaw_rate = (wheel_speed * math.tan(steering_angle)) / wheel_base
        else:
            yaw_rate = 0.0
        
        # Update orientation
        yaw += yaw_rate * dt
        roll = -0.1 * steering_angle  # Banking in turns
        pitch = 0.05 * (target_speed - wheel_speed)  # Pitch during acceleration/braking
        
        roll_rate = (-0.1 * steering_angle - roll) / dt if dt > 0 else 0.0
        pitch_rate = (0.05 * (target_speed - wheel_speed) - pitch) / dt if dt > 0 else 0.0
        
        # Update velocity components
        acceleration = (target_speed - wheel_speed) / dt if dt > 0 else 0.0
        vx = wheel_speed * math.cos(yaw)
        vy = wheel_speed * math.sin(yaw)
        vz = 0.0
        
        # Update position
        x += vx * dt
        y += vy * dt
        
        # Convert to GPS coordinates
        earth_radius = 6371000.0
        ref_lat_rad = math.radians(ref_lat)
        
        dlat = y / earth_radius
        dlon = x / (earth_radius * math.cos(ref_lat_rad))
        
        lat = ref_lat + math.degrees(dlat)
        lon = ref_lon + math.degrees(dlon)
        
        # Generate GPS data (lower frequency - 1 Hz)
        if i % int(1.0 / dt) == 0:  # 1 Hz
            gps_sample = {
                'timestamp': t,
                'latitude': lat,
                'longitude': lon,
                'altitude': z,
                'speed': wheel_speed,
                'heading': math.degrees(yaw) % 360,
                'accuracy': 5.0
            }
            
            if add_noise:
                gps_sample['latitude'] += random.gauss(0, 0.00005)  # ~5m noise
                gps_sample['longitude'] += random.gauss(0, 0.00005)
                gps_sample['altitude'] += random.gauss(0, 10)
                gps_sample['speed'] += random.gauss(0, 0.5)
                gps_sample['heading'] += random.gauss(0, 2)
                gps_sample['accuracy'] += random.gauss(0, 2)
            
            gps_data.append(gps_sample)
        
        # Generate IMU data (higher frequency - 10 Hz)
        if i % max(int(0.1 / dt), 1) == 0:  # 10 Hz
            # Calculate accelerations
            ax = acceleration * math.cos(yaw) - wheel_speed * yaw_rate * math.sin(yaw)
            ay = acceleration * math.sin(yaw) + wheel_speed * yaw_rate * math.cos(yaw)
            az = 9.81  # Gravity
            
            imu_sample = {
                'timestamp': t,
                'ax': ax,
                'ay': ay,
                'az': az,
                'gx': roll_rate,
                'gy': pitch_rate,
                'gz': yaw_rate
            }
            
            if add_noise:
                imu_sample['ax'] += random.gauss(0, 0.1)
                imu_sample['ay'] += random.gauss(0, 0.1)
                imu_sample['az'] += random.gauss(0, 0.1)
                imu_sample['gx'] += random.gauss(0, 0.01)
                imu_sample['gy'] += random.gauss(0, 0.01)
                imu_sample['gz'] += random.gauss(0, 0.01)
            
            imu_data.append(imu_sample)
        
        # Generate wheel speed data (medium frequency - 5 Hz)
        if i % max(int(0.2 / dt), 1) == 0:  # 5 Hz
            wheel_sample = {
                'timestamp': t,
                'wheel_speed': wheel_speed,
                'front_left': wheel_speed,
                'front_right': wheel_speed,
                'rear_left': wheel_speed,
                'rear_right': wheel_speed
            }
            
            if add_noise:
                noise = random.gauss(0, 0.1)
                wheel_sample['wheel_speed'] += noise
                wheel_sample['front_left'] += random.gauss(0, 0.1)
                wheel_sample['front_right'] += random.gauss(0, 0.1)
                wheel_sample['rear_left'] += random.gauss(0, 0.1)
                wheel_sample['rear_right'] += random.gauss(0, 0.1)
            
            wheel_data.append(wheel_sample)
        
        # Generate steering angle data (medium frequency - 5 Hz)
        if i % max(int(0.2 / dt), 1) == 0:  # 5 Hz
            steering_sample = {
                'timestamp': t,
                'steering_angle': steering_angle
            }
            
            if add_noise:
                steering_sample['steering_angle'] += random.gauss(0, 0.02)
            
            steering_data.append(steering_sample)
    
    return {
        'gps': gps_data,
        'imu': imu_data,
        'wheel_speed': wheel_data,
        'steering': steering_data
    }


def generate_highway_data(duration: float = 60.0,
                        dt: float = 0.1,
                        add_noise: bool = True,
                        device: str = 'cpu') -> Dict[str, List[Dict[str, Any]]]:
    """Generate highway driving scenario with lane changes."""
    
    # Similar structure to urban driving but with:
    # - Higher sustained speeds
    # - Gentler accelerations
    # - Smoother steering inputs
    # - Less frequent stops
    
    data = generate_urban_driving_data(duration, dt, add_noise, device)
    
    # Modify for highway characteristics
    for sensor_data in data['gps']:
        if 'speed' in sensor_data:
            sensor_data['speed'] = min(sensor_data['speed'] * 2.0, 30.0)  # Higher speeds
    
    for sensor_data in data['wheel_speed']:
        if 'wheel_speed' in sensor_data:
            sensor_data['wheel_speed'] = min(sensor_data['wheel_speed'] * 2.0, 30.0)
    
    for sensor_data in data['steering']:
        if 'steering_angle' in sensor_data:
            sensor_data['steering_angle'] *= 0.3  # Gentler steering
    
    return data


def generate_parking_data(duration: float = 30.0,
                        dt: float = 0.1,
                        add_noise: bool = True,
                        device: str = 'cpu') -> Dict[str, List[Dict[str, Any]]]:
    """Generate parking scenario with tight turns and low speeds."""
    
    data = generate_urban_driving_data(duration, dt, add_noise, device)
    
    # Modify for parking characteristics
    for sensor_data in data['gps']:
        if 'speed' in sensor_data:
            sensor_data['speed'] = min(sensor_data['speed'] * 0.3, 3.0)  # Very low speeds
    
    for sensor_data in data['wheel_speed']:
        if 'wheel_speed' in sensor_data:
            sensor_data['wheel_speed'] = min(sensor_data['wheel_speed'] * 0.3, 3.0)
    
    for sensor_data in data['steering']:
        if 'steering_angle' in sensor_data:
            sensor_data['steering_angle'] = max(-0.5, min(0.5, sensor_data['steering_angle'] * 3.0))  # Larger steering angles
    
    return data


def add_sensor_failures(data: Dict[str, List[Dict[str, Any]]], 
                       failure_probability: float = 0.05,
                       failure_duration: float = 2.0) -> Dict[str, List[Dict[str, Any]]]:
    """
    Add realistic sensor failures to simulated data.
    
    Args:
        data: Sensor data dictionary
        failure_probability: Probability of failure per sensor per time window
        failure_duration: Duration of failures in seconds
        
    Returns:
        Modified data with sensor failures
    """
    modified_data = {}
    
    for sensor_type, sensor_data in data.items():
        modified_sensor_data = []
        failure_end_time = -1
        
        for sample in sensor_data:
            sample_copy = sample.copy()
            current_time = sample.get('timestamp', 0)
            
            # Check if currently in failure mode
            if current_time < failure_end_time:
                # Mark as failed or corrupt data
                sample_copy['sensor_failure'] = True
                if sensor_type == 'gps':
                    sample_copy['accuracy'] = 999.0  # Very poor accuracy
                elif sensor_type == 'imu':
                    # Add large bias to IMU data
                    for key in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
                        if key in sample_copy:
                            sample_copy[key] += random.gauss(0, 5.0)
            else:
                # Check if failure should start
                if random.random() < failure_probability / len(sensor_data):
                    failure_end_time = current_time + failure_duration
                    sample_copy['sensor_failure'] = True
            
            modified_sensor_data.append(sample_copy)
        
        modified_data[sensor_type] = modified_sensor_data
    
    return modified_data


def generate_ground_truth_trajectory(data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Generate ground truth trajectory from simulated data.
    
    Args:
        data: Simulated sensor data
        
    Returns:
        List of ground truth state dictionaries
    """
    ground_truth = []
    
    # Extract ground truth from GPS data (assuming it's the most complete)
    gps_data = data.get('gps', [])
    
    for gps_sample in gps_data:
        truth_state = {
            'timestamp': gps_sample['timestamp'],
            'latitude': gps_sample['latitude'],
            'longitude': gps_sample['longitude'], 
            'altitude': gps_sample['altitude'],
            'speed': gps_sample['speed'],
            'heading': gps_sample['heading']
        }
        
        # Find corresponding wheel speed data
        timestamp = gps_sample['timestamp']
        wheel_data = data.get('wheel_speed', [])
        closest_wheel = min(wheel_data, 
                          key=lambda x: abs(x['timestamp'] - timestamp),
                          default={'wheel_speed': 0})
        truth_state['true_wheel_speed'] = closest_wheel.get('wheel_speed', 0)
        
        # Find corresponding steering data
        steering_data = data.get('steering', [])
        closest_steering = min(steering_data,
                             key=lambda x: abs(x['timestamp'] - timestamp), 
                             default={'steering_angle': 0})
        truth_state['true_steering_angle'] = closest_steering.get('steering_angle', 0)
        
        ground_truth.append(truth_state)
    
    return ground_truth


def create_test_scenarios() -> Dict[str, Dict[str, Any]]:
    """
    Create predefined test scenarios for validation.
    
    Returns:
        Dictionary of test scenarios with their parameters
    """
    scenarios = {
        'straight_line': {
            'description': 'Straight line driving at constant speed',
            'duration': 30.0,
            'dt': 0.1,
            'parameters': {
                'constant_speed': 20.0,
                'steering_angle': 0.0,
                'no_turns': True
            }
        },
        
        'circular_path': {
            'description': 'Driving in a perfect circle',
            'duration': 40.0,
            'dt': 0.1,
            'parameters': {
                'radius': 50.0,
                'speed': 10.0,
                'constant_steering': True
            }
        },
        
        'stop_and_go': {
            'description': 'Urban stop-and-go traffic',
            'duration': 60.0,
            'dt': 0.1,
            'parameters': {
                'stop_frequency': 0.1,  # stops per second
                'max_speed': 15.0,
                'acceleration_noise': 0.5
            }
        },
        
        'lane_change': {
            'description': 'Highway lane change maneuver',
            'duration': 20.0,
            'dt': 0.1,
            'parameters': {
                'initial_speed': 25.0,
                'lane_change_duration': 5.0,
                'max_steering_angle': 0.15
            }
        },
        
        'u_turn': {
            'description': 'U-turn maneuver',
            'duration': 15.0,
            'dt': 0.1,
            'parameters': {
                'turn_radius': 5.0,
                'max_speed': 5.0,
                'max_steering_angle': 0.5
            }
        }
    }
    
    return scenarios