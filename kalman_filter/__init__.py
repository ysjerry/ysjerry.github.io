"""
PyTorch Kalman Filter for Multi-Sensor Fusion

A modular implementation for fusing data from multiple sensors:
- IMU (gyroscope and accelerometer)
- GPS (heading, velocity, position coordinates, altitude) 
- Wheel speed sensor
- Steering angle sensor

The system is designed for GPU acceleration using PyTorch and outputs
optimized GPS coordinates (latitude, longitude).
"""

__version__ = "1.0.0"
__author__ = "Kalman Filter Team"

from .core.kalman import KalmanFilter
from .fusion import MultiSensorFusion

__all__ = ['KalmanFilter', 'MultiSensorFusion']