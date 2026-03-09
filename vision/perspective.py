import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def correct_perspective(image):
    """
    Enhanced perspective correction with multiple methods
    Returns corrected image or original if correction fails
    """
    if image is None:
        return None
    
    try:
        # Try multiple methods
        methods = [
            correct_by_contours,
            correct_by_lines,
            correct_by_corners,
            correct_by_grid
        ]
        
        best_result = None
        best_score = 0
        
        for method in methods:
            try:
                result = method(image)
                if result is not None:
                    # Score the result
                    score = evaluate_correction(result, image)
                    logger.debug(f"Method {method.__name__} score: {score}")
                    
                    if score > best_score:
                        best_result = result
                        best_score = score
                        
                        if score > 0.8:  # Good enough, break early
                            break
                            
            except Exception as e:
                logger.debug(f"Method {method.__name__} failed: {e}")
                continue
        
        if best_result is not None and best_score > 0.5:
            logger.info(f"✅ Perspective correction successful (score: {best_score:.2f})")
            return best_result
        else:
            logger.warning("⚠️ Perspective correction failed, using original")
            return image
            
    except Exception as e:
        logger.error(f"Perspective correction error: {e}")
        return image

def correct_by_contours(image):
    """Correct perspective using largest contour"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Try multiple threshold methods
    for thresh_method in [cv2.THRESH_BINARY + cv2.THRESH_OTSU, cv2.THRESH_BINARY]:
        _, thresh = cv2.threshold(blur, 0, 255, thresh_method)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Approximate polygon
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            
            # Get dimensions
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            
            maxWidth = max(int(widthA), int(widthB))
            maxHeight = max(int(heightA), int(heightB))
            
            # Ensure minimum size
            if maxWidth < 100 or maxHeight < 100:
                continue
            
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")
            
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            
            return warped
    
    return None

def correct_by_lines(image):
    """Correct perspective using line detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                            minLineLength=100, maxLineGap=10)
    
    if lines is None or len(lines) < 4:
        return None
    
    # Find intersection points of lines
    points = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            x3, y3, x4, y4 = lines[j][0]
            
            # Calculate intersection
            intersection = line_intersection((x1, y1, x2, y2), (x3, y3, x4, y4))
            if intersection is not None:
                x, y = intersection
                # Check if within image bounds
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    points.append((x, y))
    
    if len(points) < 4:
        return None
    
    # Find convex hull of intersection points
    points = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(points)
    
    if len(hull) >= 4:
        # Get the four extreme points
        rect = cv2.approxPolyDP(hull, 10, True)
        if len(rect) == 4:
            rect = rect.reshape(4, 2)
            rect = order_points(rect)
            
            # Get dimensions
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            
            maxWidth = max(int(widthA), int(widthB))
            maxHeight = max(int(heightA), int(heightB))
            
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")
            
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            
            return warped
    
    return None

def correct_by_corners(image):
    """Correct perspective using corner detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect corners
    corners = cv2.goodFeaturesToTrack(gray, 20, 0.01, 10)
    
    if corners is None or len(corners) < 4:
        return None
    
    corners = np.int0(corners)
    
    # Find the four extreme corners
    points = corners.reshape(-1, 2)
    
    # Get convex hull
    hull = cv2.convexHull(points)
    hull = hull.reshape(-1, 2)
    
    if len(hull) >= 4:
        # Find the four corners that are most likely the document corners
        # by looking at distance from image corners
        img_h, img_w = image.shape[:2]
        img_corners = np.array([[0,0], [img_w-1,0], [img_w-1,img_h-1], [0,img_h-1]])
        
        # Find hull points closest to image corners
        selected_points = []
        for img_corner in img_corners:
            distances = [np.linalg.norm(point - img_corner) for point in hull]
            closest_idx = np.argmin(distances)
            selected_points.append(hull[closest_idx])
        
        selected_points = np.array(selected_points, dtype=np.float32)
        rect = order_points(selected_points)
        
        # Get dimensions
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped
    
    return None

def correct_by_grid(image):
    """Correct perspective by detecting grid pattern"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive threshold to find grid
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours that might be grid cells
    cell_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 5000:  # Typical cell size range
            cell_contours.append(cnt)
    
    if len(cell_contours) < 4:
        return None
    
    # Get bounding rectangles of cells
    rects = [cv2.boundingRect(cnt) for cnt in cell_contours]
    
    # Find the overall bounding box
    x_coords = [x for x, y, w, h in rects]
    y_coords = [y for x, y, w, h in rects]
    x2_coords = [x + w for x, y, w, h in rects]
    y2_coords = [y + h for x, y, w, h in rects]
    
    if not x_coords:
        return None
    
    min_x = min(x_coords)
    min_y = min(y_coords)
    max_x = max(x2_coords)
    max_y = max(y2_coords)
    
    # Add padding
    pad = 20
    min_x = max(0, min_x - pad)
    min_y = max(0, min_y - pad)
    max_x = min(image.shape[1], max_x + pad)
    max_y = min(image.shape[0], max_y + pad)
    
    # Crop to grid
    cropped = image[min_y:max_y, min_x:max_x]
    
    return cropped

def line_intersection(line1, line2):
    """Find intersection point of two lines"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Line equations
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if denom == 0:
        return None  # Lines are parallel
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (int(x), int(y))
    
    return None

def order_points(pts):
    """Order points in consistent order: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and diff for ordering
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum
    rect[1] = pts[np.argmin(diff)]  # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest difference
    
    return rect

def evaluate_correction(corrected, original):
    """Evaluate quality of perspective correction"""
    try:
        # Check if corrected image is too small
        if corrected.shape[0] < 100 or corrected.shape[1] < 100:
            return 0
        
        # Check aspect ratio (should be reasonable)
        aspect_ratio = corrected.shape[1] / corrected.shape[0]
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return 0.3
        
        # Check for straight lines (grid should have mostly horizontal/vertical lines)
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50)
        
        if lines is not None:
            h_lines = 0
            v_lines = 0
            total_lines = len(lines)
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 20:
                    h_lines += 1
                elif angle > 70:
                    v_lines += 1
            
            # Good correction should have many horizontal/vertical lines
            if total_lines > 0:
                line_score = (h_lines + v_lines) / total_lines
            else:
                line_score = 0
        else:
            line_score = 0
        
        # Check for text alignment (should be roughly horizontal)
        # This is a simplified check
        text_score = 0.5  # Default
        
        # Combine scores
        final_score = 0.3 * (1 - abs(aspect_ratio - 1)) + 0.4 * line_score + 0.3 * text_score
        
        return final_score
        
    except Exception as e:
        logger.error(f"Error evaluating correction: {e}")
        return 0