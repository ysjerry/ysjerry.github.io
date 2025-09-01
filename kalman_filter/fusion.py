"""
Multi-Sensor Fusion Pipeline

Main fusion system that combines data from all sensors using the Kalman filter
to provide optimized GPS coordinates and vehicle state estimation.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import time
import math

from .core.kalman import KalmanFilter
from .sensors.gps import GPSSensor
from .sensors.imu import IMUSensor
from .sensors.wheel_speed import WheelSpeedSensor
from .sensors.steering import SteeringAngleSensor
from .utils.preprocessing import preprocess_sensor_data, SensorDataBuffer


class MultiSensorFusion:
    """
    Multi-sensor fusion system for vehicle state estimation.
    
    Combines data from GPS, IMU, wheel speed, and steering angle sensors
    to provide accurate and robust position and motion estimates.
    """
    
    def __init__(self,
                 device: str = 'cpu',
                 gps_reference_point: Optional[Tuple[float, float]] = None,
                 update_rates: Optional[Dict[str, float]] = None,
                 sensor_params: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize multi-sensor fusion system.
        
        Args:
            device: PyTorch device ('cpu' or 'cuda')
            gps_reference_point: Reference GPS coordinates (lat, lon)
            update_rates: Expected update rates for each sensor type
            sensor_params: Parameters for sensor configurations
        """
        self.device = torch.device(device)
        
        # Initialize Kalman filter
        self.kalman_filter = KalmanFilter(device=device)
        
        # Initialize sensors
        self.gps_sensor = GPSSensor()
        self.imu_sensor = IMUSensor()
        self.wheel_sensor = WheelSpeedSensor()
        self.steering_sensor = SteeringAngleSensor()
        
        # Set GPS reference point if provided
        if gps_reference_point is not None:
            self.gps_sensor.set_reference_point(gps_reference_point[0], gps_reference_point[1])
        
        # Configure sensor parameters
        if sensor_params:
            self._configure_sensors(sensor_params)
        
        # Default update rates (Hz)
        self.update_rates = update_rates or {
            'gps': 1.0,      # 1 Hz
            'imu': 10.0,     # 10 Hz
            'wheel_speed': 5.0,  # 5 Hz
            'steering': 5.0  # 5 Hz
        }
        
        # Data buffer for preprocessing
        self.data_buffer = SensorDataBuffer(buffer_size=20)
        
        # State tracking
        self.last_update_times = {}
        self.is_initialized = False
        self.initialization_count = 0
        self.min_init_samples = 5  # Minimum samples needed for initialization
        
        # Performance metrics
        self.update_counts = {sensor: 0 for sensor in self.update_rates.keys()}
        self.last_positions = []
        self.position_history_size = 10
        
    def _configure_sensors(self, sensor_params: Dict[str, Dict[str, Any]]):
        """Configure sensor parameters."""
        if 'gps' in sensor_params:
            params = sensor_params['gps']
            self.gps_sensor.position_noise_std = params.get('position_noise_std', self.gps_sensor.position_noise_std)
            self.gps_sensor.velocity_noise_std = params.get('velocity_noise_std', self.gps_sensor.velocity_noise_std)
            self.gps_sensor.heading_noise_std = params.get('heading_noise_std', self.gps_sensor.heading_noise_std)
        
        if 'imu' in sensor_params:
            params = sensor_params['imu']
            self.imu_sensor.accel_noise_std = params.get('accel_noise_std', self.imu_sensor.accel_noise_std)
            self.imu_sensor.gyro_noise_std = params.get('gyro_noise_std', self.imu_sensor.gyro_noise_std)
        
        if 'wheel_speed' in sensor_params:
            params = sensor_params['wheel_speed']
            self.wheel_sensor.noise_std = params.get('noise_std', self.wheel_sensor.noise_std)
            self.wheel_sensor.wheel_radius = params.get('wheel_radius', self.wheel_sensor.wheel_radius)
        
        if 'steering' in sensor_params:
            params = sensor_params['steering']
            self.steering_sensor.noise_std = params.get('noise_std', self.steering_sensor.noise_std)
            self.steering_sensor.wheel_base = params.get('wheel_base', self.steering_sensor.wheel_base)
    
    def initialize_with_gps(self, gps_data: Dict[str, Any]) -> bool:
        """
        Initialize the system with GPS data.
        
        Args:
            gps_data: GPS measurement data
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Set reference point if not already set
            if self.gps_sensor.reference_lat == 0.0 and self.gps_sensor.reference_lon == 0.0:
                if 'latitude' in gps_data and 'longitude' in gps_data:
                    self.gps_sensor.set_reference_point(gps_data['latitude'], gps_data['longitude'])
                else:
                    return False
            
            # Convert GPS to local coordinates for initialization
            if all(key in gps_data for key in ['latitude', 'longitude']):
                x, y, z = self.gps_sensor.gps_to_local_coords(
                    gps_data['latitude'], 
                    gps_data['longitude'], 
                    gps_data.get('altitude', 0.0)
                )
                
                # Initialize state vector
                self.kalman_filter.state[0] = x  # x position
                self.kalman_filter.state[1] = y  # y position
                self.kalman_filter.state[2] = z  # z position (altitude)
                
                # Initialize velocity if available
                if 'speed' in gps_data and 'heading' in gps_data:
                    if gps_data['speed'] is not None and gps_data['heading'] is not None:
                        speed = gps_data['speed']
                        heading = math.radians(gps_data['heading'])
                        
                        self.kalman_filter.state[3] = speed * math.cos(heading)  # vx
                        self.kalman_filter.state[4] = speed * math.sin(heading)  # vy
                        self.kalman_filter.state[8] = heading  # yaw
                
                self.initialization_count += 1
                
                if self.initialization_count >= self.min_init_samples:
                    self.is_initialized = True
                    return True
            
            return False
            
        except Exception as e:
            print(f"Initialization error: {e}")
            return False
    
    def update_with_sensor_data(self, sensor_data: Dict[str, Dict[str, Any]], timestamp: float) -> Dict[str, Any]:
        """
        Update the fusion system with new sensor data.
        
        Args:
            sensor_data: Dictionary containing data from various sensors
            timestamp: Current timestamp
            
        Returns:
            Updated state estimate and diagnostics
        """
        if not self.is_initialized:
            # Try to initialize with GPS data
            if 'gps' in sensor_data:
                self.initialize_with_gps(sensor_data['gps'])
                if not self.is_initialized:
                    return {'initialized': False, 'message': 'Waiting for GPS initialization'}
        
        # Prediction step
        dt = self._calculate_dt(timestamp)
        if dt > 0:
            self.kalman_filter.predict(dt)
        
        # Process each sensor type
        results = {'timestamp': timestamp, 'updates_applied': []}
        
        # GPS updates (lower frequency, higher accuracy for position)
        if 'gps' in sensor_data:
            success = self._update_with_gps(sensor_data['gps'])
            if success:
                results['updates_applied'].append('gps')
                self.update_counts['gps'] += 1
        
        # IMU updates (higher frequency, good for orientation and angular velocity)
        if 'imu' in sensor_data:
            success = self._update_with_imu(sensor_data['imu'])
            if success:
                results['updates_applied'].append('imu')
                self.update_counts['imu'] += 1
        
        # Wheel speed updates (medium frequency, good for speed validation)
        if 'wheel_speed' in sensor_data:
            success = self._update_with_wheel_speed(sensor_data['wheel_speed'], sensor_data.get('steering'))
            if success:
                results['updates_applied'].append('wheel_speed')
                self.update_counts['wheel_speed'] += 1
        
        # Steering angle updates (helps with yaw rate estimation)
        if 'steering' in sensor_data:
            current_speed = self.kalman_filter.get_wheel_speed()
            success = self._update_with_steering(sensor_data['steering'], current_speed)
            if success:
                results['updates_applied'].append('steering')
                self.update_counts['steering'] += 1
        
        # Get current state estimate
        state = self.kalman_filter.get_full_state()
        results.update(state)
        
        # Add GPS coordinates
        gps_coords = self.get_gps_coordinates()
        results['gps_coordinates'] = gps_coords
        
        # Update position history
        self._update_position_history(gps_coords)
        
        # Add diagnostics
        results['diagnostics'] = self._get_diagnostics()
        results['initialized'] = self.is_initialized
        
        return results
    
    def _calculate_dt(self, current_time: float) -> float:
        """Calculate time step since last update."""
        if not hasattr(self, '_last_prediction_time'):
            self._last_prediction_time = current_time
            return 0.1  # Default dt
        
        dt = current_time - self._last_prediction_time
        self._last_prediction_time = current_time
        
        # Limit dt to reasonable values
        return max(0.01, min(1.0, dt))
    
    def _update_with_gps(self, gps_data: Dict[str, Any]) -> bool:
        """Update with GPS sensor data."""
        try:
            # Preprocess GPS data
            processed_data = preprocess_sensor_data(gps_data, 'gps')
            if not processed_data.get('valid', False):
                return False
            
            # Process GPS measurements
            measurement, obs_matrix, noise = self.gps_sensor.process_gps_data(processed_data, self.device)
            
            if measurement is not None:
                # Update Kalman filter
                self.kalman_filter.update(measurement, obs_matrix, noise)
                return True
            
            return False
            
        except Exception as e:
            print(f"GPS update error: {e}")
            return False
    
    def _update_with_imu(self, imu_data: Dict[str, Any]) -> bool:
        """Update with IMU sensor data."""
        try:
            # Preprocess IMU data
            processed_data = preprocess_sensor_data(imu_data, 'imu')
            if not processed_data.get('valid', False):
                return False
            
            # Get current orientation for gravity compensation
            current_orientation = self.kalman_filter.get_orientation()
            
            # Process IMU measurements
            measurement, obs_matrix, noise = self.imu_sensor.process_imu_data(
                processed_data, current_orientation, self.device)
            
            if measurement is not None and measurement.numel() > 0:
                # Update Kalman filter
                self.kalman_filter.update(measurement, obs_matrix, noise)
                return True
            
            return False
            
        except Exception as e:
            print(f"IMU update error: {e}")
            return False
    
    def _update_with_wheel_speed(self, wheel_data: Dict[str, Any], steering_data: Optional[Dict[str, Any]] = None) -> bool:
        """Update with wheel speed sensor data."""
        try:
            # Preprocess wheel speed data
            processed_data = preprocess_sensor_data(wheel_data, 'wheel_speed')
            if not processed_data.get('valid', False):
                return False
            
            # Get steering angle if available
            steering_angle = None
            if steering_data and 'steering_angle' in steering_data:
                steering_angle = steering_data['steering_angle']
            
            # Process wheel speed measurements
            measurement, obs_matrix, noise = self.wheel_sensor.process_wheel_speed_data(
                processed_data, steering_angle, self.device)
            
            if measurement is not None:
                # Update Kalman filter
                self.kalman_filter.update(measurement, obs_matrix, noise)
                return True
            
            return False
            
        except Exception as e:
            print(f"Wheel speed update error: {e}")
            return False
    
    def _update_with_steering(self, steering_data: Dict[str, Any], current_speed: float) -> bool:
        """Update with steering angle sensor data."""
        try:
            # Preprocess steering data
            processed_data = preprocess_sensor_data(steering_data, 'steering')
            if not processed_data.get('valid', False):
                return False
            
            # Process steering measurements
            measurement, obs_matrix, noise = self.steering_sensor.process_steering_data(
                processed_data, current_speed, self.device)
            
            if measurement is not None:
                # Update Kalman filter
                self.kalman_filter.update(measurement, obs_matrix, noise)
                return True
            
            return False
            
        except Exception as e:
            print(f"Steering update error: {e}")
            return False
    
    def get_gps_coordinates(self) -> Dict[str, float]:
        """
        Get current GPS coordinates from the state estimate.
        
        Returns:
            Dictionary with latitude, longitude, and altitude
        """
        try:
            # Get current position estimate
            x, y, z = self.kalman_filter.get_position()
            
            # Convert back to GPS coordinates
            lat, lon, alt = self.gps_sensor.local_to_gps_coords(x, y, z)
            
            return {
                'latitude': lat,
                'longitude': lon,
                'altitude': alt
            }
            
        except Exception as e:
            print(f"GPS coordinate conversion error: {e}")
            return {'latitude': 0.0, 'longitude': 0.0, 'altitude': 0.0}
    
    def _update_position_history(self, gps_coords: Dict[str, float]):
        """Update position history for diagnostics."""
        self.last_positions.append(gps_coords.copy())
        if len(self.last_positions) > self.position_history_size:
            self.last_positions.pop(0)
    
    def _get_diagnostics(self) -> Dict[str, Any]:
        """Get system diagnostics."""
        diagnostics = {
            'update_counts': self.update_counts.copy(),
            'is_initialized': self.is_initialized,
            'initialization_count': self.initialization_count,
            'filter_device': str(self.kalman_filter.device),
        }
        
        # Calculate update rates
        if hasattr(self, '_last_prediction_time'):
            total_time = self._last_prediction_time
            if total_time > 0:
                diagnostics['actual_update_rates'] = {
                    sensor: count / total_time for sensor, count in self.update_counts.items()
                }
        
        # Position stability metric
        if len(self.last_positions) >= 2:
            distances = []
            for i in range(1, len(self.last_positions)):
                prev = self.last_positions[i-1]
                curr = self.last_positions[i]
                
                # Simple Euclidean distance (approximate for small distances)
                lat_diff = curr['latitude'] - prev['latitude']
                lon_diff = curr['longitude'] - prev['longitude']
                distance = math.sqrt(lat_diff**2 + lon_diff**2) * 111000  # Convert to meters (approximate)
                distances.append(distance)
            
            if distances:
                diagnostics['position_stability'] = {
                    'mean_movement': sum(distances) / len(distances),
                    'max_movement': max(distances),
                    'total_movement': sum(distances)
                }
        
        return diagnostics
    
    def reset(self):
        """Reset the fusion system to initial state."""
        self.kalman_filter.reset()
        self.is_initialized = False
        self.initialization_count = 0
        self.last_update_times.clear()
        self.update_counts = {sensor: 0 for sensor in self.update_rates.keys()}
        self.last_positions.clear()
        
        # Reset GPS reference point
        self.gps_sensor.set_reference_point(0.0, 0.0)
    
    def get_sensor_info(self) -> Dict[str, Any]:
        """Get information about all configured sensors."""
        return {
            'gps': self.gps_sensor.get_measurement_info(),
            'imu': self.imu_sensor.get_measurement_info(),
            'wheel_speed': self.wheel_sensor.get_measurement_info(),
            'steering': self.steering_sensor.get_measurement_info(),
            'fusion_parameters': {
                'update_rates': self.update_rates,
                'device': str(self.device),
                'min_init_samples': self.min_init_samples
            }
        }
    
    def to(self, device: str):
        """Move the fusion system to specified device."""
        self.device = torch.device(device)
        self.kalman_filter.to(device)
        return self