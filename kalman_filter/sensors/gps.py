"""
GPS Sensor integration module.

Handles GPS data including position, velocity, heading, and altitude.
Converts GPS coordinates to local coordinate system for Kalman filter processing.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
import math


class GPSSensor:
    """GPS sensor integration for Kalman filter."""
    
    def __init__(self, 
                 reference_lat: float = 0.0,
                 reference_lon: float = 0.0,
                 position_noise_std: float = 5.0,  # GPS accuracy in meters
                 velocity_noise_std: float = 0.5,
                 heading_noise_std: float = 0.1,
                 altitude_noise_std: float = 10.0):
        """
        Initialize GPS sensor.
        
        Args:
            reference_lat: Reference latitude for coordinate conversion
            reference_lon: Reference longitude for coordinate conversion
            position_noise_std: Standard deviation of position measurements
            velocity_noise_std: Standard deviation of velocity measurements  
            heading_noise_std: Standard deviation of heading measurements
            altitude_noise_std: Standard deviation of altitude measurements
        """
        self.reference_lat = reference_lat
        self.reference_lon = reference_lon
        self.position_noise_std = position_noise_std
        self.velocity_noise_std = velocity_noise_std
        self.heading_noise_std = heading_noise_std
        self.altitude_noise_std = altitude_noise_std
        
        # Earth's radius in meters (approximate)
        self.earth_radius = 6371000.0
        
    def set_reference_point(self, lat: float, lon: float):
        """Set reference point for coordinate conversion."""
        self.reference_lat = lat
        self.reference_lon = lon
    
    def gps_to_local_coords(self, lat: float, lon: float, alt: float = 0.0) -> Tuple[float, float, float]:
        """
        Convert GPS coordinates to local Cartesian coordinates.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters
            
        Returns:
            Tuple of (x, y, z) in meters relative to reference point
        """
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        ref_lat_rad = math.radians(self.reference_lat)
        ref_lon_rad = math.radians(self.reference_lon)
        
        # Calculate differences
        dlat = lat_rad - ref_lat_rad
        dlon = lon_rad - ref_lon_rad
        
        # Convert to local coordinates (approximate for small distances)
        x = self.earth_radius * dlon * math.cos(ref_lat_rad)
        y = self.earth_radius * dlat
        z = alt
        
        return x, y, z
    
    def local_to_gps_coords(self, x: float, y: float, z: float = 0.0) -> Tuple[float, float, float]:
        """
        Convert local Cartesian coordinates back to GPS coordinates.
        
        Args:
            x: X coordinate in meters
            y: Y coordinate in meters
            z: Z coordinate (altitude) in meters
            
        Returns:
            Tuple of (latitude, longitude, altitude)
        """
        ref_lat_rad = math.radians(self.reference_lat)
        ref_lon_rad = math.radians(self.reference_lon)
        
        # Convert back to GPS coordinates
        dlat = y / self.earth_radius
        dlon = x / (self.earth_radius * math.cos(ref_lat_rad))
        
        lat = math.degrees(ref_lat_rad + dlat)
        lon = math.degrees(ref_lon_rad + dlon)
        alt = z
        
        return lat, lon, alt
    
    def process_gps_data(self, gps_data: Dict[str, Any], device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process raw GPS data for Kalman filter update.
        
        Args:
            gps_data: Dictionary containing GPS measurements
            device: PyTorch device
            
        Returns:
            Tuple of (measurement_vector, observation_matrix, measurement_noise)
        """
        measurements = []
        observation_rows = []
        noise_variances = []
        
        # Convert GPS position to local coordinates
        if all(key in gps_data for key in ['latitude', 'longitude']):
            lat = gps_data['latitude']
            lon = gps_data['longitude']
            alt = gps_data.get('altitude', 0.0)
            
            # Set reference point on first measurement if not set
            if self.reference_lat == 0.0 and self.reference_lon == 0.0:
                self.set_reference_point(lat, lon)
            
            x, y, z = self.gps_to_local_coords(lat, lon, alt)
            
            # Add position measurements
            measurements.extend([x, y, z])
            
            # Observation matrix for position (maps to state indices 0, 1, 2)
            pos_obs = torch.zeros(3, 13, device=torch.device(device))
            pos_obs[0, 0] = 1.0  # x position
            pos_obs[1, 1] = 1.0  # y position
            pos_obs[2, 2] = 1.0  # z position (altitude)
            observation_rows.append(pos_obs)
            
            # Position noise
            noise_variances.extend([
                self.position_noise_std**2,
                self.position_noise_std**2,
                self.altitude_noise_std**2
            ])
        
        # Process velocity data
        if 'speed' in gps_data and 'heading' in gps_data and gps_data['speed'] is not None and gps_data['heading'] is not None:
            speed = gps_data['speed']  # m/s
            heading = math.radians(gps_data['heading'])  # Convert to radians
            
            # Convert speed and heading to velocity components
            vx = speed * math.cos(heading)
            vy = speed * math.sin(heading)
            vz = 0.0  # Assume no vertical velocity from GPS
            
            measurements.extend([vx, vy])
            
            # Observation matrix for velocity (maps to state indices 3, 4)
            vel_obs = torch.zeros(2, 13, device=torch.device(device))
            vel_obs[0, 3] = 1.0  # vx
            vel_obs[1, 4] = 1.0  # vy
            observation_rows.append(vel_obs)
            
            # Velocity noise
            noise_variances.extend([
                self.velocity_noise_std**2,
                self.velocity_noise_std**2
            ])
        
        # Process heading data (maps to yaw)
        if 'heading' in gps_data and gps_data['heading'] is not None:
            heading_rad = math.radians(gps_data['heading'])
            measurements.append(heading_rad)
            
            # Observation matrix for heading (maps to state index 8 - yaw)
            heading_obs = torch.zeros(1, 13, device=torch.device(device))
            heading_obs[0, 8] = 1.0  # yaw angle
            observation_rows.append(heading_obs)
            
            # Heading noise
            noise_variances.append(self.heading_noise_std**2)
        
        if not measurements:
            return None, None, None
        
        # Combine measurements and observation matrices
        measurement_vector = torch.tensor(measurements, dtype=torch.float32, device=torch.device(device))
        observation_matrix = torch.cat(observation_rows, dim=0)
        measurement_noise = torch.diag(torch.tensor(noise_variances, dtype=torch.float32, device=torch.device(device)))
        
        return measurement_vector, observation_matrix, measurement_noise
    
    def get_measurement_info(self) -> Dict[str, Any]:
        """Get information about GPS measurements."""
        return {
            'sensor_type': 'GPS',
            'measurements': ['position_x', 'position_y', 'altitude', 'velocity_x', 'velocity_y', 'heading'],
            'noise_std': {
                'position': self.position_noise_std,
                'velocity': self.velocity_noise_std,
                'heading': self.heading_noise_std,
                'altitude': self.altitude_noise_std
            },
            'reference_point': {
                'latitude': self.reference_lat,
                'longitude': self.reference_lon
            }
        }