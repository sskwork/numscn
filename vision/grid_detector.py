import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_grid(image):
    """
    Enhanced grid detection with multiple methods
    Returns cropped grid image or original if detection fails
    """
    if image is None:
        return None
    
    try:
        # Try multiple detection methods
        methods = [
            detect_by_contour,
            detect_by_lines,
            detect_by_grid_pattern,
            detect_by_morphology
        ]
        
        best_grid = None
        best_score = 0
        
        for method in methods:
            try:
                grid = method(image)
                if grid is not None:
                    score = evaluate_grid_detection(grid, image)
                    logger.debug(f"Method {method.__name__} score: {score}")
                    
                    if score > best_score:
                        best_grid = grid
                        best_score = score
                        
                        if score > 0.8:  # Good enough
                            break
                            
            except Exception as e:
                logger.debug(f"Method {method.__name__} failed: {e}")
                continue
        
        if best_grid is not None and best_score > 0.5:
            logger.info(f"✅ Grid detection successful (score: {best_score:.2f})")
            return best_grid
        else:
            logger.warning("⚠️ Grid detection failed, using original")
            return image
            
    except Exception as e:
        logger.error(f"Grid detection error: {e}")
        return image

def detect_by_contour(image):
    """Detect grid by finding largest rectangular contour"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Try multiple threshold methods
    for thresh_method in [cv2.THRESH_BINARY + cv2.THRESH_OTSU, 
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU]:
        _, thresh = cv2.threshold(blur, 0, 255, thresh_method)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest)
        
        # Check if detected area is reasonable
        if (w > 100 and h > 100 and 
            w < image.shape[1] * 0.95 and 
            h < image.shape[0] * 0.95 and
            w * h > image.shape[0] * image.shape[1] * 0.1):  # At least 10% of image
            
            # Add padding
            pad = 20
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(image.shape[1] - x, w + 2*pad)
            h = min(image.shape[0] - y, h + 2*pad)
            
            grid = image[y:y+h, x:x+w]
            return grid
    
    return None

def detect_by_lines(image):
    """Detect grid by finding horizontal and vertical lines"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                            minLineLength=100, maxLineGap=10)
    
    if lines is None or len(lines) < 4:
        return None
    
    # Separate horizontal and vertical lines
    h_lines = []
    v_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if angle < 20 and length > 100:  # Horizontal
            h_lines.append((y1, length))
        elif angle > 70 and length > 100:  # Vertical
            v_lines.append((x1, length))
    
    if len(h_lines) < 2 or len(v_lines) < 2:
        return None
    
    # Get the most prominent lines (by length)
    h_lines.sort(key=lambda x: x[1], reverse=True)
    v_lines.sort(key=lambda x: x[1], reverse=True)
    
    # Take top lines
    h_positions = sorted([pos for pos, _ in h_lines[:10]])
    v_positions = sorted([pos for pos, _ in v_lines[:10]])
    
    if len(h_positions) < 2 or len(v_positions) < 2:
        return None
    
    # Get grid boundaries
    top = min(h_positions)
    bottom = max(h_positions)
    left = min(v_positions)
    right = max(v_positions)
    
    # Add padding
    pad = 20
    top = max(0, top - pad)
    bottom = min(image.shape[0], bottom + pad)
    left = max(0, left - pad)
    right = min(image.shape[1], right + pad)
    
    if bottom > top and right > left:
        grid = image[top:bottom, left:right]
        return grid
    
    return None

def detect_by_grid_pattern(image):
    """Detect grid by looking for repetitive patterns"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use morphological operations to emphasize grid
    # Horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combine
    grid_lines = cv2.bitwise_or(horizontal, vertical)
    
    # Threshold
    _, thresh = cv2.threshold(grid_lines, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours of grid
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Find the largest contour that might be the grid
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Grid should cover significant portion of image
        if area > image.shape[0] * image.shape[1] * 0.2:
            valid_contours.append((cnt, area))
    
    if not valid_contours:
        return None
    
    # Get the largest valid contour
    largest = max(valid_contours, key=lambda x: x[1])[0]
    x, y, w, h = cv2.boundingRect(largest)
    
    # Add padding
    pad = 20
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(image.shape[1] - x, w + 2*pad)
    h = min(image.shape[0] - y, h + 2*pad)
    
    grid = image[y:y+h, x:x+w]
    return grid

def detect_by_morphology(image):
    """Detect grid using morphological operations"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find grid cells using morphological operations
    # Horizontal line detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Vertical line detection
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combine to get grid
    grid_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    # Dilate to connect grid lines
    kernel = np.ones((3,3), np.uint8)
    grid_mask = cv2.dilate(grid_mask, kernel, iterations=2)
    
    # Find contours of the grid
    contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Find the largest contour that forms a rectangle
    best_rect = None
    best_area = 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Check if it's reasonably sized
        if (area > image.shape[0] * image.shape[1] * 0.1 and
            w > 100 and h > 100):
            
            # Check if contour is roughly rectangular
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            
            if len(approx) == 4:
                if area > best_area:
                    best_area = area
                    best_rect = (x, y, w, h)
    
    if best_rect is not None:
        x, y, w, h = best_rect
        
        # Add padding
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image.shape[1] - x, w + 2*pad)
        h = min(image.shape[0] - y, h + 2*pad)
        
        grid = image[y:y+h, x:x+w]
        return grid
    
    return None

def evaluate_grid_detection(grid, original):
    """Evaluate quality of grid detection"""
    try:
        if grid is None or grid.size == 0:
            return 0
        
        # Check size (should be significant portion of original)
        size_ratio = (grid.shape[0] * grid.shape[1]) / (original.shape[0] * original.shape[1])
        if size_ratio < 0.1:
            return 0.1
        elif size_ratio > 0.9:
            size_score = 0.5  # Might be too large (no cropping)
        else:
            size_score = 0.8  # Good size
        
        # Check for grid-like patterns
        gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30)
        
        if lines is not None:
            h_lines = 0
            v_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if angle < 20 and length > 50:
                    h_lines += 1
                elif angle > 70 and length > 50:
                    v_lines += 1
            
            # Good grid should have multiple horizontal and vertical lines
            if h_lines >= 3 and v_lines >= 3:
                line_score = 1.0
            elif h_lines >= 2 and v_lines >= 2:
                line_score = 0.7
            elif h_lines >= 1 and v_lines >= 1:
                line_score = 0.4
            else:
                line_score = 0.1
        else:
            line_score = 0
        
        # Combine scores
        final_score = 0.3 * size_score + 0.7 * line_score
        
        return final_score
        
    except Exception as e:
        logger.error(f"Error evaluating grid detection: {e}")
        return 0