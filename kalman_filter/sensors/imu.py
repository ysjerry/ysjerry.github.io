"""
IMU (Inertial Measurement Unit) Sensor integration module.

Handles accelerometer and gyroscope data from IMU sensors.
Processes 3-axis acceleration and angular velocity measurements.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
import math


class IMUSensor:
    """IMU sensor integration for Kalman filter."""
    
    def __init__(self,
                 accel_noise_std: float = 0.1,  # m/s²
                 gyro_noise_std: float = 0.01,  # rad/s
                 gravity: float = 9.81):
        """
        Initialize IMU sensor.
        
        Args:
            accel_noise_std: Standard deviation of accelerometer measurements
            gyro_noise_std: Standard deviation of gyroscope measurements
            gravity: Gravity constant in m/s²
        """
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.gravity = gravity
        
        # Calibration offsets (to be determined during calibration)
        self.accel_bias = torch.zeros(3)
        self.gyro_bias = torch.zeros(3)
        
        # Previous velocity for acceleration integration
        self.prev_velocity = None
        self.prev_time = None
        
    def set_calibration(self, accel_bias: Optional[Tuple[float, float, float]] = None,
                       gyro_bias: Optional[Tuple[float, float, float]] = None):
        """
        Set calibration biases for IMU sensors.
        
        Args:
            accel_bias: Accelerometer bias (x, y, z)
            gyro_bias: Gyroscope bias (x, y, z)
        """
        if accel_bias is not None:
            self.accel_bias = torch.tensor(accel_bias)
        if gyro_bias is not None:
            self.gyro_bias = torch.tensor(gyro_bias)
    
    def process_accelerometer_data(self, accel_data: Dict[str, float], 
                                 current_orientation: Optional[Tuple[float, float, float]] = None,
                                 device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process accelerometer data for Kalman filter update.
        
        Args:
            accel_data: Dictionary with keys 'ax', 'ay', 'az' (m/s²)
            current_orientation: Current orientation (roll, pitch, yaw) for gravity compensation
            device: PyTorch device
            
        Returns:
            Tuple of (measurement_vector, observation_matrix, measurement_noise)
        """
        if not all(key in accel_data for key in ['ax', 'ay', 'az']):
            return None, None, None
        
        # Extract accelerometer measurements
        ax = accel_data['ax']
        ay = accel_data['ay']
        az = accel_data['az']
        
        # Apply calibration
        ax -= self.accel_bias[0].item()
        ay -= self.accel_bias[1].item()
        az -= self.accel_bias[2].item()
        
        # Compensate for gravity if orientation is known
        if current_orientation is not None:
            roll, pitch, yaw = current_orientation
            
            # Calculate gravity components in body frame
            gx = self.gravity * math.sin(pitch)
            gy = -self.gravity * math.sin(roll) * math.cos(pitch)
            gz = -self.gravity * math.cos(roll) * math.cos(pitch)
            
            # Remove gravity from measurements
            ax -= gx
            ay -= gy
            az -= gz + self.gravity  # az includes gravity when stationary
        
        # Create measurement vector
        measurements = [ax, ay, az]
        measurement_vector = torch.tensor(measurements, dtype=torch.float32, device=torch.device(device))
        
        # For accelerometer, we don't directly observe the state variables
        # Instead, we use it for validation or indirect updates
        # This is a simplified approach - in practice, you might use it for attitude estimation
        observation_matrix = torch.zeros(3, 13, device=torch.device(device))
        
        # Measurement noise
        noise_variances = [self.accel_noise_std**2] * 3
        measurement_noise = torch.diag(torch.tensor(noise_variances, dtype=torch.float32, device=torch.device(device)))
        
        return measurement_vector, observation_matrix, measurement_noise
    
    def process_gyroscope_data(self, gyro_data: Dict[str, float], device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process gyroscope data for Kalman filter update.
        
        Args:
            gyro_data: Dictionary with keys 'gx', 'gy', 'gz' (rad/s)
            device: PyTorch device
            
        Returns:
            Tuple of (measurement_vector, observation_matrix, measurement_noise)
        """
        if not all(key in gyro_data for key in ['gx', 'gy', 'gz']):
            return None, None, None
        
        # Extract gyroscope measurements
        gx = gyro_data['gx']
        gy = gyro_data['gy']
        gz = gyro_data['gz']
        
        # Apply calibration
        gx -= self.gyro_bias[0].item()
        gy -= self.gyro_bias[1].item()
        gz -= self.gyro_bias[2].item()
        
        # Create measurement vector
        measurements = [gx, gy, gz]
        measurement_vector = torch.tensor(measurements, dtype=torch.float32, device=torch.device(device))
        
        # Observation matrix for angular velocities (maps to state indices 9, 10, 11)
        observation_matrix = torch.zeros(3, 13, device=torch.device(device))
        observation_matrix[0, 9] = 1.0   # roll_rate
        observation_matrix[1, 10] = 1.0  # pitch_rate
        observation_matrix[2, 11] = 1.0  # yaw_rate
        
        # Measurement noise
        noise_variances = [self.gyro_noise_std**2] * 3
        measurement_noise = torch.diag(torch.tensor(noise_variances, dtype=torch.float32, device=torch.device(device)))
        
        return measurement_vector, observation_matrix, measurement_noise
    
    def process_imu_data(self, imu_data: Dict[str, Any], 
                        current_orientation: Optional[Tuple[float, float, float]] = None,
                        device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process combined IMU data (accelerometer + gyroscope).
        
        Args:
            imu_data: Dictionary containing both accelerometer and gyroscope data
            current_orientation: Current orientation for gravity compensation
            device: PyTorch device
            
        Returns:
            Tuple of (measurement_vector, observation_matrix, measurement_noise)
        """
        measurements = []
        observation_rows = []
        noise_variances = []
        
        # Process gyroscope data (more reliable for Kalman filter)
        if all(key in imu_data for key in ['gx', 'gy', 'gz']):
            gyro_data = {k: imu_data[k] for k in ['gx', 'gy', 'gz']}
            gyro_meas, gyro_obs, gyro_noise = self.process_gyroscope_data(gyro_data, device)
            
            if gyro_meas is not None:
                measurements.extend(gyro_meas.tolist())
                observation_rows.append(gyro_obs)
                noise_variances.extend(torch.diag(gyro_noise).tolist())
        
        # Process accelerometer data (for validation/cross-checking)
        if all(key in imu_data for key in ['ax', 'ay', 'az']):
            accel_data = {k: imu_data[k] for k in ['ax', 'ay', 'az']}
            accel_meas, accel_obs, accel_noise = self.process_accelerometer_data(
                accel_data, current_orientation, device)
            
            # For now, we don't directly use accelerometer in the update
            # In a more sophisticated implementation, you could use it for attitude estimation
            pass
        
        if not measurements:
            return None, None, None
        
        # Combine measurements
        measurement_vector = torch.tensor(measurements, dtype=torch.float32, device=torch.device(device))
        observation_matrix = torch.cat(observation_rows, dim=0) if observation_rows else torch.empty(0, 13, device=torch.device(device))
        measurement_noise = torch.diag(torch.tensor(noise_variances, dtype=torch.float32, device=torch.device(device)))
        
        return measurement_vector, observation_matrix, measurement_noise
    
    def estimate_attitude_from_accel(self, accel_data: Dict[str, float]) -> Tuple[float, float]:
        """
        Estimate roll and pitch from accelerometer data (assuming static conditions).
        
        Args:
            accel_data: Accelerometer data
            
        Returns:
            Tuple of (roll, pitch) in radians
        """
        if not all(key in accel_data for key in ['ax', 'ay', 'az']):
            return 0.0, 0.0
        
        ax = accel_data['ax']
        ay = accel_data['ay'] 
        az = accel_data['az']
        
        # Calculate roll and pitch from gravity vector
        roll = math.atan2(ay, math.sqrt(ax*ax + az*az))
        pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))
        
        return roll, pitch
    
    def get_measurement_info(self) -> Dict[str, Any]:
        """Get information about IMU measurements."""
        return {
            'sensor_type': 'IMU',
            'measurements': ['angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'],
            'noise_std': {
                'accelerometer': self.accel_noise_std,
                'gyroscope': self.gyro_noise_std
            },
            'calibration': {
                'accel_bias': self.accel_bias.tolist(),
                'gyro_bias': self.gyro_bias.tolist()
            }
        }