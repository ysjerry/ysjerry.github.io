"""
Wheel Speed Sensor integration module.

Handles wheel speed sensor data for vehicle motion estimation.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
import math


class WheelSpeedSensor:
    """Wheel speed sensor integration for Kalman filter."""
    
    def __init__(self,
                 wheel_radius: float = 0.3,  # meters
                 noise_std: float = 0.1,     # m/s
                 wheel_base: float = 2.5):   # meters (distance between front and rear axles)
        """
        Initialize wheel speed sensor.
        
        Args:
            wheel_radius: Radius of the wheels in meters
            noise_std: Standard deviation of speed measurements
            wheel_base: Distance between front and rear axles
        """
        self.wheel_radius = wheel_radius
        self.noise_std = noise_std
        self.wheel_base = wheel_base
        
    def process_wheel_speed_data(self, wheel_data: Dict[str, float],
                               steering_angle: Optional[float] = None,
                               device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process wheel speed data for Kalman filter update.
        
        Args:
            wheel_data: Dictionary with wheel speed data (RPM or direct speed)
            steering_angle: Current steering angle in radians (optional)
            device: PyTorch device
            
        Returns:
            Tuple of (measurement_vector, observation_matrix, measurement_noise)
        """
        measurements = []
        observation_rows = []
        noise_variances = []
        
        # Process different types of wheel speed data
        if 'wheel_speed' in wheel_data:
            # Direct wheel speed measurement
            speed = wheel_data['wheel_speed']
        elif 'rpm' in wheel_data:
            # Convert RPM to linear speed
            rpm = wheel_data['rpm']
            speed = (rpm * 2 * math.pi * self.wheel_radius) / 60.0  # m/s
        elif all(key in wheel_data for key in ['front_left', 'front_right', 'rear_left', 'rear_right']):
            # Average of all four wheels (if available)
            speeds = [wheel_data[key] for key in ['front_left', 'front_right', 'rear_left', 'rear_right']]
            speed = sum(speeds) / len(speeds)
        elif any(key in wheel_data for key in ['front_left', 'front_right']):
            # Average of available front wheels
            front_speeds = [wheel_data[key] for key in ['front_left', 'front_right'] if key in wheel_data]
            speed = sum(front_speeds) / len(front_speeds)
        else:
            return None, None, None
        
        # Add wheel speed measurement
        measurements.append(speed)
        
        # Observation matrix for wheel speed (maps to state index 12)
        speed_obs = torch.zeros(1, 13, device=torch.device(device))
        speed_obs[0, 12] = 1.0  # wheel_speed state
        observation_rows.append(speed_obs)
        
        # Noise
        noise_variances.append(self.noise_std**2)
        
        # If steering angle is available, we can also estimate velocity components
        if steering_angle is not None and 'wheel_speed' in wheel_data:
            # Calculate velocity components based on bicycle model
            vx, vy = self.bicycle_model_velocity(speed, steering_angle)
            
            measurements.extend([vx, vy])
            
            # Observation matrix for velocity components
            vel_obs = torch.zeros(2, 13, device=torch.device(device))
            vel_obs[0, 3] = 1.0  # vx
            vel_obs[1, 4] = 1.0  # vy
            observation_rows.append(vel_obs)
            
            # Add velocity noise (higher uncertainty due to model approximation)
            noise_variances.extend([self.noise_std**2 * 2, self.noise_std**2 * 2])
        
        # Combine measurements
        measurement_vector = torch.tensor(measurements, dtype=torch.float32, device=torch.device(device))
        observation_matrix = torch.cat(observation_rows, dim=0)
        measurement_noise = torch.diag(torch.tensor(noise_variances, dtype=torch.float32, device=torch.device(device)))
        
        return measurement_vector, observation_matrix, measurement_noise
    
    def bicycle_model_velocity(self, speed: float, steering_angle: float) -> Tuple[float, float]:
        """
        Convert wheel speed and steering angle to velocity components using bicycle model.
        
        Args:
            speed: Wheel speed in m/s
            steering_angle: Steering angle in radians
            
        Returns:
            Tuple of (vx, vy) velocity components
        """
        # Bicycle model kinematics
        # For small steering angles, we can approximate:
        beta = math.atan(0.5 * math.tan(steering_angle))  # Slip angle
        
        vx = speed * math.cos(beta)
        vy = speed * math.sin(beta)
        
        return vx, vy
    
    def estimate_yaw_rate(self, speed: float, steering_angle: float) -> float:
        """
        Estimate yaw rate from wheel speed and steering angle.
        
        Args:
            speed: Vehicle speed in m/s
            steering_angle: Steering angle in radians
            
        Returns:
            Yaw rate in rad/s
        """
        if abs(speed) < 0.1:  # Avoid division by zero
            return 0.0
            
        # Bicycle model yaw rate
        yaw_rate = (speed * math.tan(steering_angle)) / self.wheel_base
        return yaw_rate
    
    def validate_wheel_speeds(self, wheel_data: Dict[str, float]) -> bool:
        """
        Validate wheel speed measurements for consistency.
        
        Args:
            wheel_data: Wheel speed data
            
        Returns:
            True if measurements are consistent, False otherwise
        """
        if not isinstance(wheel_data, dict):
            return False
        
        # Check for reasonable speed values
        for key, value in wheel_data.items():
            if not isinstance(value, (int, float)):
                return False
            if abs(value) > 100:  # Unreasonably high speed (100 m/s = 360 km/h)
                return False
            if value < 0:  # Negative speed (could be valid for reverse)
                pass  # Allow negative speeds for now
        
        # Check for consistency between wheels (if multiple available)
        speeds = list(wheel_data.values())
        if len(speeds) > 1:
            max_speed = max(speeds)
            min_speed = min(speeds)
            if max_speed > 0 and (max_speed - min_speed) / max_speed > 0.5:  # 50% difference
                return False  # Speeds are too inconsistent
        
        return True
    
    def get_measurement_info(self) -> Dict[str, Any]:
        """Get information about wheel speed measurements."""
        return {
            'sensor_type': 'WheelSpeed',
            'measurements': ['wheel_speed', 'velocity_x', 'velocity_y'],
            'noise_std': self.noise_std,
            'parameters': {
                'wheel_radius': self.wheel_radius,
                'wheel_base': self.wheel_base
            }
        }