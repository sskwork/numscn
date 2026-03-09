# config.py - Configuration settings for the grid scanner

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Model settings
MODEL_PATHS = [
    os.path.join(MODEL_FOLDER, "digit_model_enhanced.h5"),
    os.path.join(MODEL_FOLDER, "digit_model_fallback.h5"),
    os.path.join(BASE_DIR, "digit_model_enhanced.h5"),
    os.path.join(BASE_DIR, "digit_model_fallback.h5")
]

# OCR settings
OCR_CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence to accept a digit
OCR_ENABLE_FALLBACK = True  # Use rule-based fallback if model fails

# Grid detection settings
MIN_GRID_SIZE = 3  # Minimum grid size (rows/cols)
MAX_GRID_SIZE = 15  # Maximum grid size
DEFAULT_GRID_SIZE = 5  # Default if detection fails

# Cell extraction settings
CELL_SIZE = 100  # Standard cell size for processing
CELL_PADDING = 5  # Padding around cells

# Image processing settings
ENABLE_PERSPECTIVE_CORRECTION = True
ENABLE_GRID_DETECTION = True
ENABLE_IMAGE_ENHANCEMENT = True

# Server settings
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(BASE_DIR, "scanner.log")

# File settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Export settings
EXCEL_TEMPLATE = os.path.join(BASE_DIR, "templates", "excel_template.xlsx")
EXCEL_INCLUDE_CONFIDENCE = True
EXCEL_INCLUDE_SUMMARY = True

# Camera settings
DEFAULT_CAMERA_RESOLUTION = (1920, 1080)
CAMERA_FACING_MODE = "environment"  # Use back camera
ENABLE_TORCH = True
ENABLE_AUTO_FOCUS = True
ENABLE_ZOOM = True

# Performance settings
USE_BATCH_PROCESSING = True
BATCH_SIZE = 32
ENABLE_MULTITHREADING = True
MAX_WORKERS = 4

# Cache settings
ENABLE_CACHE = True
CACHE_SIZE = 100  # Number of images to cache
CACHE_TTL = 3600  # Time to live in seconds

# Feature flags
ENABLE_CONFIDENCE_SCORING = True
ENABLE_SUGGESTIONS = True
ENABLE_KEYBOARD_SHORTCUTS = True
ENABLE_PROGRESS_BAR = True

# API settings
API_TIMEOUT = 30  # seconds
API_RETRY_COUNT = 3
API_RETRY_DELAY = 1  # seconds

# UI settings
THEME_COLOR = "#667eea"
SHOW_GRID_OVERLAY = True
GRID_OVERLAY_OPACITY = 0.2
AUTO_HIDE_STATUS = True
STATUS_DISPLAY_TIME = 5000  # milliseconds

# Export settings
EXCEL_FILENAME_PREFIX = "scan_result"
EXCEL_INCLUDE_TIMESTAMP = True
EXCEL_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"