"""
Drone Object Detection with Distance Estimation
"""

from .distance_detector import DroneObjectDistanceDetector
from . import models_config

__version__ = "0.1.0"
__all__ = ['DroneObjectDistanceDetector', 'models_config']
