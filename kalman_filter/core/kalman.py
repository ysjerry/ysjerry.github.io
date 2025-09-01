"""
PyTorch-based Kalman Filter implementation for multi-sensor fusion.

This module implements a Kalman filter optimized for fusing data from multiple
sensors (IMU, GPS, wheel speed, steering angle) to provide optimized GPS coordinates.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any


class KalmanFilter(nn.Module):
    """
    PyTorch-based Kalman Filter for multi-sensor fusion.
    
    State vector (13 dimensions):
    [x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, wheel_speed]
    
    Where:
    - x, y, z: position coordinates (GPS lat/lon converted to meters, altitude)
    - vx, vy, vz: velocity components
    - roll, pitch, yaw: orientation angles
    - roll_rate, pitch_rate, yaw_rate: angular velocities from gyroscope
    - wheel_speed: wheel speed sensor data
    """
    
    def __init__(self, 
                 device: str = 'cpu',
                 process_noise_std: float = 0.1,
                 initial_uncertainty: float = 1.0):
        """
        Initialize Kalman Filter.
        
        Args:
            device: PyTorch device ('cpu' or 'cuda')
            process_noise_std: Standard deviation for process noise
            initial_uncertainty: Initial uncertainty in state estimates
        """
        super(KalmanFilter, self).__init__()
        
        self.device = torch.device(device)
        self.state_dim = 13  # State vector dimension
        self.dt = 1.0  # Default time step (will be updated dynamically)
        
        # State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, wheel_speed]
        self.state = torch.zeros(self.state_dim, 1, device=self.device)
        
        # State covariance matrix
        self.P = torch.eye(self.state_dim, device=self.device) * initial_uncertainty
        
        # Process noise covariance matrix
        self.Q = torch.eye(self.state_dim, device=self.device) * (process_noise_std ** 2)
        
        # State transition matrix (will be updated based on dt)
        self.F = torch.eye(self.state_dim, device=self.device)
        
        self.last_update_time = None
        
    def update_state_transition_matrix(self, dt: float):
        """Update state transition matrix based on time step."""
        self.dt = dt
        F = torch.eye(self.state_dim, device=self.device)
        
        # Position = position + velocity * dt
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt
        
        # Orientation = orientation + angular_velocity * dt
        F[6, 9] = dt   # roll += roll_rate * dt
        F[7, 10] = dt  # pitch += pitch_rate * dt
        F[8, 11] = dt  # yaw += yaw_rate * dt
        
        self.F = F
        
    def predict(self, dt: Optional[float] = None) -> torch.Tensor:
        """
        Prediction step of Kalman filter.
        
        Args:
            dt: Time step. If None, uses stored dt.
            
        Returns:
            Predicted state vector
        """
        if dt is not None:
            self.update_state_transition_matrix(dt)
            
        # Predict state: x_k = F * x_k-1
        self.state = torch.matmul(self.F, self.state)
        
        # Predict covariance: P_k = F * P_k-1 * F^T + Q
        self.P = torch.matmul(torch.matmul(self.F, self.P), self.F.T) + self.Q
        
        return self.state.clone()
    
    def update(self, 
               measurement: torch.Tensor,
               measurement_matrix: torch.Tensor,
               measurement_noise: torch.Tensor) -> torch.Tensor:
        """
        Update step of Kalman filter.
        
        Args:
            measurement: Sensor measurement vector
            measurement_matrix: Observation matrix H
            measurement_noise: Measurement noise covariance matrix R
            
        Returns:
            Updated state vector
        """
        H = measurement_matrix
        R = measurement_noise
        z = measurement.unsqueeze(1) if measurement.dim() == 1 else measurement
        
        # Innovation: y = z - H * x
        innovation = z - torch.matmul(H, self.state)
        
        # Innovation covariance: S = H * P * H^T + R
        S = torch.matmul(torch.matmul(H, self.P), H.T) + R
        
        # Kalman gain: K = P * H^T * S^-1
        try:
            K = torch.matmul(torch.matmul(self.P, H.T), torch.inverse(S))
        except RuntimeError:
            # Handle singular matrix by using pseudoinverse
            K = torch.matmul(torch.matmul(self.P, H.T), torch.pinverse(S))
        
        # Update state: x = x + K * y
        self.state = self.state + torch.matmul(K, innovation)
        
        # Update covariance: P = (I - K * H) * P
        I = torch.eye(self.state_dim, device=self.device)
        self.P = torch.matmul(I - torch.matmul(K, H), self.P)
        
        return self.state.clone()
    
    def get_position(self) -> Tuple[float, float, float]:
        """
        Get current position estimate (x, y, z).
        
        Returns:
            Tuple of (x, y, z) coordinates
        """
        return (
            self.state[0].item(),
            self.state[1].item(),
            self.state[2].item()
        )
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """
        Get current velocity estimate (vx, vy, vz).
        
        Returns:
            Tuple of (vx, vy, vz) velocities
        """
        return (
            self.state[3].item(),
            self.state[4].item(),
            self.state[5].item()
        )
    
    def get_orientation(self) -> Tuple[float, float, float]:
        """
        Get current orientation estimate (roll, pitch, yaw).
        
        Returns:
            Tuple of (roll, pitch, yaw) angles in radians
        """
        return (
            self.state[6].item(),
            self.state[7].item(),
            self.state[8].item()
        )
    
    def get_angular_velocity(self) -> Tuple[float, float, float]:
        """
        Get current angular velocity estimate.
        
        Returns:
            Tuple of (roll_rate, pitch_rate, yaw_rate) in rad/s
        """
        return (
            self.state[9].item(),
            self.state[10].item(),
            self.state[11].item()
        )
    
    def get_wheel_speed(self) -> float:
        """
        Get current wheel speed estimate.
        
        Returns:
            Wheel speed in m/s
        """
        return self.state[12].item()
    
    def get_full_state(self) -> Dict[str, Any]:
        """
        Get complete state information.
        
        Returns:
            Dictionary containing all state variables
        """
        position = self.get_position()
        velocity = self.get_velocity()
        orientation = self.get_orientation()
        angular_velocity = self.get_angular_velocity()
        wheel_speed = self.get_wheel_speed()
        
        return {
            'position': {'x': position[0], 'y': position[1], 'z': position[2]},
            'velocity': {'vx': velocity[0], 'vy': velocity[1], 'vz': velocity[2]},
            'orientation': {'roll': orientation[0], 'pitch': orientation[1], 'yaw': orientation[2]},
            'angular_velocity': {'roll_rate': angular_velocity[0], 'pitch_rate': angular_velocity[1], 'yaw_rate': angular_velocity[2]},
            'wheel_speed': wheel_speed,
            'uncertainty': torch.diag(self.P).cpu().numpy().tolist()
        }
    
    def reset(self):
        """Reset the filter to initial state."""
        self.state = torch.zeros(self.state_dim, 1, device=self.device)
        self.P = torch.eye(self.state_dim, device=self.device)
        self.last_update_time = None
    
    def to(self, device: str):
        """Move filter to specified device."""
        super().to(device)
        self.device = torch.device(device)
        self.state = self.state.to(self.device)
        self.P = self.P.to(self.device)
        self.Q = self.Q.to(self.device)
        self.F = self.F.to(self.device)
        return self