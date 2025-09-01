"""
Data preprocessing utilities for sensor data.

Contains functions for cleaning, filtering, and preparing sensor data
for the Kalman filter.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import math
from collections import deque


def preprocess_sensor_data(sensor_data: Dict[str, Any], 
                         sensor_type: str,
                         window_size: int = 5) -> Dict[str, Any]:
    """
    Preprocess sensor data for Kalman filter input.
    
    Args:
        sensor_data: Raw sensor data
        sensor_type: Type of sensor ('gps', 'imu', 'wheel_speed', 'steering')
        window_size: Size of sliding window for smoothing
        
    Returns:
        Preprocessed sensor data
    """
    if sensor_type == 'gps':
        return preprocess_gps_data(sensor_data, window_size)
    elif sensor_type == 'imu':
        return preprocess_imu_data(sensor_data, window_size)
    elif sensor_type == 'wheel_speed':
        return preprocess_wheel_speed_data(sensor_data, window_size)
    elif sensor_type == 'steering':
        return preprocess_steering_data(sensor_data, window_size)
    else:
        return sensor_data


def preprocess_gps_data(gps_data: Dict[str, Any], window_size: int = 5) -> Dict[str, Any]:
    """
    Preprocess GPS data.
    
    Args:
        gps_data: Raw GPS data
        window_size: Smoothing window size
        
    Returns:
        Preprocessed GPS data
    """
    processed_data = gps_data.copy()
    
    # Validate GPS coordinates
    if 'latitude' in processed_data and 'longitude' in processed_data:
        lat = processed_data['latitude']
        lon = processed_data['longitude']
        
        # Check for valid coordinate ranges
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            processed_data['valid'] = False
            return processed_data
        
        # Check for unrealistic accuracy
        if 'accuracy' in processed_data and processed_data['accuracy'] > 100:  # > 100m accuracy
            processed_data['low_accuracy'] = True
    
    # Convert speed from various units to m/s
    if 'speed' in processed_data and processed_data['speed'] is not None:
        speed = processed_data['speed']
        # Assuming speed is already in m/s, but add conversion if needed
        processed_data['speed'] = max(0, speed)  # Ensure non-negative
    
    # Normalize heading to [0, 2π)
    if 'heading' in processed_data and processed_data['heading'] is not None:
        heading = processed_data['heading']
        processed_data['heading'] = heading % 360  # Normalize to [0, 360)
    
    # Add timestamp if missing
    if 'timestamp' not in processed_data:
        processed_data['timestamp'] = torch.tensor(0.0)  # Default timestamp
    
    processed_data['valid'] = True
    return processed_data


def preprocess_imu_data(imu_data: Dict[str, Any], window_size: int = 5) -> Dict[str, Any]:
    """
    Preprocess IMU data.
    
    Args:
        imu_data: Raw IMU data
        window_size: Smoothing window size
        
    Returns:
        Preprocessed IMU data
    """
    processed_data = imu_data.copy()
    
    # Check for required fields
    accel_keys = ['ax', 'ay', 'az']
    gyro_keys = ['gx', 'gy', 'gz']
    
    # Validate accelerometer data
    if all(key in processed_data for key in accel_keys):
        for key in accel_keys:
            value = processed_data[key]
            # Check for reasonable accelerometer values (-50g to 50g)
            if abs(value) > 490:  # 50g ≈ 490 m/s²
                processed_data[f'{key}_outlier'] = True
    
    # Validate gyroscope data
    if all(key in processed_data for key in gyro_keys):
        for key in gyro_keys:
            value = processed_data[key]
            # Check for reasonable gyroscope values (-35 to 35 rad/s)
            if abs(value) > 35:
                processed_data[f'{key}_outlier'] = True
    
    # Calculate magnitude for validation
    if all(key in processed_data for key in accel_keys):
        ax, ay, az = processed_data['ax'], processed_data['ay'], processed_data['az']
        accel_magnitude = math.sqrt(ax*ax + ay*ay + az*az)
        processed_data['accel_magnitude'] = accel_magnitude
        
        # Check if close to gravity (for static detection)
        if abs(accel_magnitude - 9.81) < 2.0:
            processed_data['likely_static'] = True
    
    processed_data['valid'] = True
    return processed_data


def preprocess_wheel_speed_data(wheel_data: Dict[str, Any], window_size: int = 5) -> Dict[str, Any]:
    """
    Preprocess wheel speed data.
    
    Args:
        wheel_data: Raw wheel speed data
        window_size: Smoothing window size
        
    Returns:
        Preprocessed wheel speed data
    """
    processed_data = wheel_data.copy()
    
    # Convert RPM to m/s if needed
    if 'rpm' in processed_data and 'wheel_speed' not in processed_data:
        rpm = processed_data['rpm']
        wheel_radius = processed_data.get('wheel_radius', 0.3)  # default 30cm
        speed = (rpm * 2 * math.pi * wheel_radius) / 60.0
        processed_data['wheel_speed'] = speed
    
    # Validate speed values
    speed_keys = ['wheel_speed', 'front_left', 'front_right', 'rear_left', 'rear_right']
    for key in speed_keys:
        if key in processed_data:
            speed = processed_data[key]
            if speed < 0:
                processed_data[f'{key}_negative'] = True
            if abs(speed) > 50:  # > 180 km/h seems unrealistic for most applications
                processed_data[f'{key}_outlier'] = True
    
    # Calculate average speed if multiple wheels available
    available_speeds = [processed_data[key] for key in speed_keys 
                       if key in processed_data and not processed_data.get(f'{key}_outlier', False)]
    if len(available_speeds) > 1:
        processed_data['average_speed'] = sum(available_speeds) / len(available_speeds)
        
        # Check for wheel slip (large differences between wheels)
        max_speed = max(available_speeds)
        min_speed = min(available_speeds)
        if max_speed > 1.0 and (max_speed - min_speed) / max_speed > 0.2:  # 20% difference
            processed_data['possible_wheel_slip'] = True
    
    processed_data['valid'] = True
    return processed_data


def preprocess_steering_data(steering_data: Dict[str, Any], window_size: int = 5) -> Dict[str, Any]:
    """
    Preprocess steering angle data.
    
    Args:
        steering_data: Raw steering data
        window_size: Smoothing window size
        
    Returns:
        Preprocessed steering data
    """
    processed_data = steering_data.copy()
    
    # Validate steering angle
    if 'steering_angle' in processed_data:
        angle = processed_data['steering_angle']
        
        # Convert to radians if in degrees
        if abs(angle) > 6.28:  # Likely in degrees if > 2π
            angle = math.radians(angle)
            processed_data['steering_angle'] = angle
        
        # Check for reasonable steering angle range
        max_angle = math.radians(45)  # 45 degrees maximum
        if abs(angle) > max_angle:
            processed_data['steering_angle_outlier'] = True
    
    processed_data['valid'] = True
    return processed_data


class SensorDataBuffer:
    """Buffer for storing and smoothing sensor data over time."""
    
    def __init__(self, buffer_size: int = 10):
        """
        Initialize data buffer.
        
        Args:
            buffer_size: Maximum number of samples to store
        """
        self.buffer_size = buffer_size
        self.buffers = {}
    
    def add_data(self, sensor_type: str, data: Dict[str, Any]):
        """Add new sensor data to buffer."""
        if sensor_type not in self.buffers:
            self.buffers[sensor_type] = deque(maxlen=self.buffer_size)
        
        self.buffers[sensor_type].append(data)
    
    def get_smoothed_data(self, sensor_type: str, smooth_keys: List[str]) -> Optional[Dict[str, Any]]:
        """
        Get smoothed sensor data.
        
        Args:
            sensor_type: Type of sensor
            smooth_keys: Keys to apply smoothing to
            
        Returns:
            Smoothed sensor data or None if insufficient data
        """
        if sensor_type not in self.buffers or len(self.buffers[sensor_type]) == 0:
            return None
        
        buffer = self.buffers[sensor_type]
        result = buffer[-1].copy()  # Start with most recent data
        
        # Apply smoothing to specified keys
        for key in smooth_keys:
            values = []
            for data in buffer:
                if key in data and data[key] is not None:
                    values.append(data[key])
            
            if values:
                # Simple moving average
                result[f'{key}_smoothed'] = sum(values) / len(values)
                result[f'{key}_std'] = np.std(values) if len(values) > 1 else 0.0
        
        return result
    
    def get_data_rate(self, sensor_type: str) -> float:
        """
        Calculate data rate for a sensor type.
        
        Args:
            sensor_type: Type of sensor
            
        Returns:
            Data rate in Hz, or 0 if insufficient data
        """
        if sensor_type not in self.buffers or len(self.buffers[sensor_type]) < 2:
            return 0.0
        
        buffer = self.buffers[sensor_type]
        timestamps = []
        
        for data in buffer:
            if 'timestamp' in data:
                timestamps.append(data['timestamp'])
        
        if len(timestamps) < 2:
            return 0.0
        
        # Calculate average time difference
        time_diffs = []
        for i in range(1, len(timestamps)):
            diff = timestamps[i] - timestamps[i-1]
            if diff > 0:
                time_diffs.append(diff)
        
        if time_diffs:
            avg_dt = sum(time_diffs) / len(time_diffs)
            return 1.0 / avg_dt if avg_dt > 0 else 0.0
        
        return 0.0


def detect_outliers(data_series: List[float], method: str = 'iqr', threshold: float = 1.5) -> List[bool]:
    """
    Detect outliers in a data series.
    
    Args:
        data_series: List of numerical data points
        method: Detection method ('iqr', 'zscore', 'modified_zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        List of boolean values indicating outliers
    """
    if len(data_series) < 3:
        return [False] * len(data_series)
    
    data = np.array(data_series)
    
    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outliers = z_scores > threshold
    
    elif method == 'modified_zscore':
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        outliers = np.abs(modified_z_scores) > threshold
    
    else:
        outliers = [False] * len(data_series)
    
    return outliers.tolist()


def synchronize_sensor_data(sensor_data_dict: Dict[str, List[Dict[str, Any]]], 
                          time_window: float = 0.1) -> List[Dict[str, Any]]:
    """
    Synchronize data from multiple sensors based on timestamps.
    
    Args:
        sensor_data_dict: Dictionary mapping sensor types to lists of data
        time_window: Time window for synchronization (seconds)
        
    Returns:
        List of synchronized sensor data dictionaries
    """
    if not sensor_data_dict:
        return []
    
    # Find common time range
    all_timestamps = []
    for sensor_type, data_list in sensor_data_dict.items():
        for data in data_list:
            if 'timestamp' in data:
                all_timestamps.append(data['timestamp'])
    
    if not all_timestamps:
        return []
    
    all_timestamps.sort()
    min_time = min(all_timestamps)
    max_time = max(all_timestamps)
    
    # Create synchronized data points
    synchronized_data = []
    current_time = min_time
    
    while current_time <= max_time:
        sync_point = {'timestamp': current_time}
        
        # Find closest data point for each sensor within time window
        for sensor_type, data_list in sensor_data_dict.items():
            closest_data = None
            min_time_diff = float('inf')
            
            for data in data_list:
                if 'timestamp' in data:
                    time_diff = abs(data['timestamp'] - current_time)
                    if time_diff <= time_window and time_diff < min_time_diff:
                        closest_data = data
                        min_time_diff = time_diff
            
            if closest_data:
                sync_point[sensor_type] = closest_data
        
        # Only add if we have data from at least one sensor
        if len(sync_point) > 1:  # More than just timestamp
            synchronized_data.append(sync_point)
        
        current_time += time_window
    
    return synchronized_data