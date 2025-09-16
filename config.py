"""
Configuration settings for Face Mask Detection System
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

class Config:
    """Base configuration class"""
    
    # Model settings
    FACE_DETECTION_CONFIDENCE = float(os.getenv('FACE_DETECTION_CONFIDENCE', '0.5'))
    MASK_DETECTION_CONFIDENCE = float(os.getenv('MASK_DETECTION_CONFIDENCE', '0.5'))
    FACE_DETECTION_MODEL = BASE_DIR / "face_detector" / "deploy.prototxt"
    FACE_DETECTION_WEIGHTS = BASE_DIR / "face_detector" / "res10_300x300_ssd_iter_140000.caffemodel"
    MASK_DETECTOR_MODEL = BASE_DIR / "mask_detector.keras"
    
    # Camera settings
    CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
    CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '640'))
    CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '480'))
    CAMERA_FPS = int(os.getenv('CAMERA_FPS', '30'))
    
    # Detection settings
    MAX_FACES = int(os.getenv('MAX_FACES', '10'))
    FACE_RESIZE_SIZE = (224, 224)
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
    
    # Web app settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', '5000'))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Logging settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = LOG_DIR / "face_mask_detection.log"
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5
    
    # Performance settings
    ENABLE_GPU = os.getenv('ENABLE_GPU', 'True').lower() == 'true'
    FRAME_SKIP = int(os.getenv('FRAME_SKIP', '1'))  # Process every Nth frame
    
    # UI settings
    UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL', '1000'))  # ms
    SHOW_CONFIDENCE = os.getenv('SHOW_CONFIDENCE', 'True').lower() == 'true'
    SHOW_FPS = os.getenv('SHOW_FPS', 'True').lower() == 'true'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
