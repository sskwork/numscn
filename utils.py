# utils.py - Utility functions for the grid scanner

import os
import cv2
import numpy as np
import logging
import json
import hashlib
import time
from datetime import datetime
from functools import lru_cache
import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

def timer_decorator(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

@lru_cache(maxsize=config.CACHE_SIZE)
def get_image_hash(image_path):
    """Get hash of image for caching"""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def generate_filename(prefix=config.EXCEL_FILENAME_PREFIX):
    """Generate unique filename with timestamp"""
    if config.EXCEL_INCLUDE_TIMESTAMP:
        timestamp = datetime.now().strftime(config.EXCEL_TIMESTAMP_FORMAT)
        return f"{prefix}_{timestamp}"
    return prefix

def resize_image(image, max_size=1920):
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    
    return image

def enhance_contrast(image):
    """Enhance image contrast using CLAHE"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    if len(image.shape) == 3:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced

def denoise_image(image):
    """Apply denoising to image"""
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def sharpen_image(image):
    """Sharpen image using kernel"""
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

def preprocess_for_ocr(image):
    """Preprocess image specifically for OCR"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply series of preprocessing steps
    enhanced = enhance_contrast(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def calculate_confidence_score(predictions):
    """Calculate confidence score from model predictions"""
    if not predictions:
        return 0
    
    # Get top 2 predictions
    sorted_pred = sorted(predictions, reverse=True)
    
    if len(sorted_pred) >= 2:
        # Confidence based on margin between top 2 predictions
        margin = sorted_pred[0] - sorted_pred[1]
        return min(sorted_pred[0] + margin * 0.5, 1.0)
    
    return sorted_pred[0] if sorted_pred else 0

def draw_grid_on_image(image, grid_size=5, color=(0,255,0), thickness=2):
    """Draw grid lines on image for visualization"""
    result = image.copy()
    h, w = result.shape[:2]
    
    # Calculate cell sizes
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    # Draw vertical lines
    for i in range(1, grid_size):
        x = i * cell_w
        cv2.line(result, (x, 0), (x, h), color, thickness)
    
    # Draw horizontal lines
    for i in range(1, grid_size):
        y = i * cell_h
        cv2.line(result, (0, y), (w, y), color, thickness)
    
    return result

def save_debug_image(image, prefix="debug"):
    """Save image for debugging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(config.UPLOAD_FOLDER, filename)
    cv2.imwrite(filepath, image)
    logger.info(f"Debug image saved: {filepath}")
    return filepath

def load_json_config(config_path):
    """Load JSON configuration file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def save_json_config(config_path, config_data):
    """Save configuration to JSON file"""
    try:
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False

def create_thumbnail(image, size=(200,200)):
    """Create thumbnail of image"""
    return cv2.resize(image, size)

def get_image_info(image):
    """Get information about image"""
    info = {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "min_value": np.min(image),
        "max_value": np.max(image),
        "mean_value": np.mean(image),
        "std_value": np.std(image)
    }
    
    if len(image.shape) == 3:
        info["channels"] = image.shape[2]
    
    return info

def validate_grid_data(grid_data):
    """Validate extracted grid data"""
    if not grid_data:
        return False
    
    # Check if it's a 2D list
    if not isinstance(grid_data, list):
        return False
    
    if not all(isinstance(row, list) for row in grid_data):
        return False
    
    # Check consistency
    row_lengths = [len(row) for row in grid_data]
    if len(set(row_lengths)) > 1:
        return False
    
    return True

def merge_results(results_list):
    """Merge multiple scan results"""
    merged = {}
    
    for results in results_list:
        for cell, value in results.items():
            if cell in merged:
                # If cell already has value, keep the one with more digits
                if len(value) > len(merged[cell]):
                    merged[cell] = value
            else:
                merged[cell] = value
    
    return merged

def format_processing_time(seconds):
    """Format processing time for display"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"

def calculate_statistics(results):
    """Calculate statistics from results"""
    total_cells = len(results)
    filled_cells = sum(1 for v in results.values() if v)
    empty_cells = total_cells - filled_cells
    
    # Calculate digit statistics
    all_digits = ''.join(results.values())
    digit_count = len(all_digits)
    
    # Per-digit frequency
    digit_freq = {}
    for d in range(10):
        digit_freq[str(d)] = all_digits.count(str(d))
    
    return {
        "total_cells": total_cells,
        "filled_cells": filled_cells,
        "empty_cells": empty_cells,
        "fill_rate": filled_cells / total_cells if total_cells > 0 else 0,
        "total_digits": digit_count,
        "digits_per_cell": digit_count / filled_cells if filled_cells > 0 else 0,
        "digit_frequency": digit_freq
    }