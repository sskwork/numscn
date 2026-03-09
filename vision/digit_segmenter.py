import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def segment_digits(cell):
    """
    Enhanced digit segmentation with multiple methods and confidence scoring
    """
    if cell is None or cell.size == 0:
        return []
    
    try:
        # Convert to grayscale
        if len(cell.shape) == 3:
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell.copy()
        
        # Preprocess cell
        processed = preprocess_cell(gray)
        
        # Try multiple segmentation methods
        all_digits = []
        
        # Method 1: Connected components analysis
        digits1 = segment_by_components(processed)
        all_digits.extend([(x, img, 0.7) for x, img in digits1])  # Base confidence 0.7
        
        # Method 2: Contour analysis
        digits2 = segment_by_contours(processed)
        all_digits.extend([(x, img, 0.8) for x, img in digits2])  # Base confidence 0.8
        
        # Method 3: MSER (Maximally Stable Extremal Regions)
        digits3 = segment_by_mser(processed)
        all_digits.extend([(x, img, 0.6) for x, img in digits3])  # Base confidence 0.6
        
        # Method 4: Projection analysis
        digits4 = segment_by_projection(processed)
        all_digits.extend([(x, img, 0.5) for x, img in digits4])  # Base confidence 0.5
        
        # Remove duplicates and sort
        unique_digits = remove_duplicate_digits(all_digits)
        
        logger.info(f"Found {len(unique_digits)} digits in cell")
        return unique_digits
        
    except Exception as e:
        logger.error(f"Error in segment_digits: {e}")
        return []

def preprocess_cell(gray):
    """Enhanced cell preprocessing"""
    try:
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(sharpened, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological cleaning
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return gray

def segment_by_components(binary):
    """Segment using connected components analysis"""
    digits = []
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
    
    for i in range(1, num_labels):  # Skip background
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter based on size and aspect ratio
        if filter_component(w, h, area, binary.shape):
            digit = binary[y:y+h, x:x+w]
            cleaned = clean_digit_enhanced(digit)
            if cleaned is not None:
                digits.append((x, cleaned))
    
    return digits

def filter_component(w, h, area, img_shape):
    """Filter connected components that might be digits"""
    img_h, img_w = img_shape
    
    # Size constraints
    if w < 5 or h < 8 or w > img_w * 0.8 or h > img_h * 0.8:
        return False
    
    # Area constraints
    if area < 20 or area > img_w * img_h * 0.5:
        return False
    
    # Aspect ratio (digits are roughly 0.3 to 2.0 aspect ratio)
    aspect_ratio = h / w if w > 0 else 0
    if aspect_ratio < 0.3 or aspect_ratio > 3.0:
        return False
    
    # Solid