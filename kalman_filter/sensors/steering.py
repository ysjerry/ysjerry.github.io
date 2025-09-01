"""
Steering Angle Sensor integration module.

Handles steering angle sensor data for vehicle motion estimation.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
import math


class SteeringAngleSensor:
    """Steering angle sensor integration for Kalman filter."""
    
    def __init__(self,
                 noise_std: float = 0.02,      # radians (~1 degree)
                 wheel_base: float = 2.5,      # meters
                 max_steering_angle: float = 0.7):  # radians (~40 degrees)
        """
        Initialize steering angle sensor.
        
        Args:
            noise_std: Standard deviation of steering angle measurements
            wheel_base: Distance between front and rear axles
            max_steering_angle: Maximum possible steering angle
        """
        self.noise_std = noise_std
        self.wheel_base = wheel_base
        self.max_steering_angle = max_steering_angle
        
        # Calibration offset
        self.steering_offset = 0.0
        
    def set_calibration(self, steering_offset: float):
        """
        Set calibration offset for steering angle.
        
        Args:
            steering_offset: Steering angle offset in radians
        """
        self.steering_offset = steering_offset
    
    def process_steering_data(self, steering_data: Dict[str, float],
                            current_speed: Optional[float] = None,
                            device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process steering angle data for Kalman filter update.
        
        Args:
            steering_data: Dictionary with steering angle data
            current_speed: Current vehicle speed for yaw rate calculation
            device: PyTorch device
            
        Returns:
            Tuple of (measurement_vector, observation_matrix, measurement_noise)
        """
        if 'steering_angle' not in steering_data:
            return None, None, None
        
        # Get steering angle
        steering_angle = steering_data['steering_angle']
        
        # Apply calibration
        steering_angle -= self.steering_offset
        
        # Validate steering angle
        if not self.validate_steering_angle(steering_angle):
            return None, None, None
        
        measurements = []
        observation_rows = []
        noise_variances = []
        
        # If we have current speed, calculate expected yaw rate
        if current_speed is not None and abs(current_speed) > 0.1:
            expected_yaw_rate = self.calculate_yaw_rate(current_speed, steering_angle)
            
            # Add yaw rate measurement
            measurements.append(expected_yaw_rate)
            
            # Observation matrix for yaw rate (maps to state index 11)
            yaw_rate_obs = torch.zeros(1, 13, device=torch.device(device))
            yaw_rate_obs[0, 11] = 1.0  # yaw_rate state
            observation_rows.append(yaw_rate_obs)
            
            # Yaw rate noise (higher uncertainty due to model approximation)
            yaw_rate_noise = self.calculate_yaw_rate_noise(current_speed, steering_angle)
            noise_variances.append(yaw_rate_noise**2)
        
        # For very precise systems, you might also estimate lateral acceleration
        if current_speed is not None and abs(current_speed) > 0.1:
            lateral_accel = self.calculate_lateral_acceleration(current_speed, steering_angle)
            
            # This could be used for validation against IMU data
            # For now, we don't include it in the measurement but store for reference
            steering_data['estimated_lateral_accel'] = lateral_accel
        
        if not measurements:
            # If no speed available, we can't compute yaw rate
            # Return None to skip this measurement
            return None, None, None
        
        # Combine measurements
        measurement_vector = torch.tensor(measurements, dtype=torch.float32, device=torch.device(device))
        observation_matrix = torch.cat(observation_rows, dim=0)
        measurement_noise = torch.diag(torch.tensor(noise_variances, dtype=torch.float32, device=torch.device(device)))
        
        return measurement_vector, observation_matrix, measurement_noise
    
    def calculate_yaw_rate(self, speed: float, steering_angle: float) -> float:
        """
        Calculate expected yaw rate from speed and steering angle using bicycle model.
        
        Args:
            speed: Vehicle speed in m/s
            steering_angle: Steering angle in radians
            
        Returns:
            Expected yaw rate in rad/s
        """
        if abs(speed) < 0.1:  # Avoid division by zero
            return 0.0
        
        # Bicycle model
        yaw_rate = (speed * math.tan(steering_angle)) / self.wheel_base
        
        # Limit yaw rate to reasonable values
        max_yaw_rate = 2.0  # rad/s (about 115 deg/s)
        yaw_rate = max(-max_yaw_rate, min(max_yaw_rate, yaw_rate))
        
        return yaw_rate
    
    def calculate_lateral_acceleration(self, speed: float, steering_angle: float) -> float:
        """
        Calculate expected lateral acceleration from speed and steering angle.
        
        Args:
            speed: Vehicle speed in m/s
            steering_angle: Steering angle in radians
            
        Returns:
            Lateral acceleration in m/s²
        """
        yaw_rate = self.calculate_yaw_rate(speed, steering_angle)
        lateral_accel = speed * yaw_rate
        
        # Limit to reasonable values
        max_lateral_accel = 10.0  # m/s²
        lateral_accel = max(-max_lateral_accel, min(max_lateral_accel, lateral_accel))
        
        return lateral_accel
    
    def calculate_yaw_rate_noise(self, speed: float, steering_angle: float) -> float:
        """
        Calculate noise standard deviation for yaw rate estimation.
        
        Args:
            speed: Vehicle speed in m/s
            steering_angle: Steering angle in radians
            
        Returns:
            Noise standard deviation for yaw rate
        """
        # Base noise from steering angle uncertainty
        base_noise = self.noise_std
        
        # Additional uncertainty from speed dependency
        speed_factor = 1.0 + abs(speed) * 0.1
        
        # Additional uncertainty for large steering angles
        angle_factor = 1.0 + abs(steering_angle) * 2.0
        
        # Combined noise
        total_noise = base_noise * speed_factor * angle_factor
        
        return total_noise
    
    def validate_steering_angle(self, steering_angle: float) -> bool:
        """
        Validate steering angle measurement.
        
        Args:
            steering_angle: Steering angle in radians
            
        Returns:
            True if valid, False otherwise
        """
        # Check if angle is within reasonable bounds
        if abs(steering_angle) > self.max_steering_angle:
            return False
        
        # Check for NaN or infinite values
        if not math.isfinite(steering_angle):
            return False
        
        return True
    
    def estimate_turning_radius(self, steering_angle: float) -> Optional[float]:
        """
        Estimate turning radius from steering angle.
        
        Args:
            steering_angle: Steering angle in radians
            
        Returns:
            Turning radius in meters, or None if driving straight
        """
        if abs(steering_angle) < 0.01:  # Essentially straight
            return None
        
        # Bicycle model turning radius
        turning_radius = self.wheel_base / math.tan(abs(steering_angle))
        
        return turning_radius
    
    def get_measurement_info(self) -> Dict[str, Any]:
        """Get information about steering angle measurements."""
        return {
            'sensor_type': 'SteeringAngle',
            'measurements': ['yaw_rate', 'lateral_acceleration'],
            'noise_std': self.noise_std,
            'parameters': {
                'wheel_base': self.wheel_base,
                'max_steering_angle': self.max_steering_angle,
                'steering_offset': self.steering_offset
            }
        }