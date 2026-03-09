import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_cells(image):
    """
    Enhanced extract cells from grid image
    Returns 2D list of cell images
    """
    if image is None or image.size == 0:
        logger.error("❌ Error: Empty image")
        return []
    
    logger.info(f"\n🔍 Processing image: {image.shape}")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Try multiple methods in order of preference
    methods = [
        ("Contour Detection", detect_by_contours_enhanced),
        ("Line Detection", detect_by_lines_enhanced),
        ("Morphological Detection", detect_by_morphology),
        ("Adaptive Division", divide_into_grid_enhanced)
    ]
    
    best_cells = []
    best_score = 0
    
    for method_name, method_func in methods:
        logger.info(f"🔄 Trying {method_name}...")
        try:
            cells = method_func(image, gray)
            
            if cells and len(cells) > 0:
                # Score this method's result
                score = evaluate_grid_quality(cells, image)
                logger.info(f"  Score: {score:.2f}")
                
                if score > best_score:
                    best_cells = cells
                    best_score = score
                    
                    # If we get a very good score, break early
                    if score > 0.8:
                        logger.info(f"✅ Excellent result with {method_name}")
                        break
                        
        except Exception as e:
            logger.error(f"  Error in {method_name}: {e}")
            continue
    
    if best_cells and len(best_cells) > 0:
        logger.info(f"✅ Best result: {len(best_cells)} rows, {len(best_cells[0])} columns")
    else:
        logger.info("❌ No cells detected with any method")
        # Ultimate fallback
        best_cells = ultimate_fallback(image)
    
    return best_cells

def evaluate_grid_quality(cells, original_image):
    """
    Evaluate the quality of detected grid
    Returns score between 0 and 1
    """
    if not cells or len(cells) < 2 or len(cells[0]) < 2:
        return 0
    
    try:
        rows = len(cells)
        cols = len(cells[0])
        
        # Check consistency (all rows should have same number of columns)
        col_consistency = all(len(row) == cols for row in cells)
        if not col_consistency:
            return 0.3  # Low score for inconsistent grid
        
        # Check cell sizes (should be roughly uniform)
        cell_sizes = []
        for row in cells:
            for cell in row:
                if cell is not None:
                    cell_sizes.append(cell.shape[0] * cell.shape[1])
        
        if cell_sizes:
            size_variance = np.var(cell_sizes) / (np.mean(cell_sizes) ** 2) if np.mean(cell_sizes) > 0 else 1
            size_score = max(0, 1 - size_variance)
        else:
            size_score = 0
        
        # Check if cells contain content (non-zero pixels)
        content_score = 0
        total_cells = rows * cols
        
        for row in cells:
            for cell in row:
                if cell is not None and cell.size > 0:
                    if len(cell.shape) == 3:
                        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = cell.copy()
                    
                    # Count pixels that might be digits
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    non_zero = np.count_nonzero(thresh)
                    
                    if non_zero > 50:  # Likely contains a digit
                        content_score += 1
        
        content_score = content_score / total_cells if total_cells > 0 else 0
        
        # Combine scores
        final_score = (0.4 * size_score) + (0.6 * content_score)
        
        return final_score
        
    except Exception as e:
        logger.error(f"Error evaluating grid quality: {e}")
        return 0

def detect_by_contours_enhanced(image, gray):
    """Enhanced contour-based detection"""
    try:
        # Apply multiple threshold methods
        thresh_images = []
        
        # Method 1: Otsu
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh_images.append(thresh1)
        
        # Method 2: Adaptive
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        thresh_images.append(thresh2)
        
        # Method 3: Combined with Canny
        edges = cv2.Canny(gray, 50, 150)
        thresh3 = cv2.bitwise_or(thresh1, edges)
        thresh_images.append(thresh3)
        
        best_cells = []
        best_rect_count = 0
        
        for thresh in thresh_images:
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            cell_rects = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Filter based on size and aspect ratio
                if (w > 20 and h > 20 and 
                    w < image.shape[1] * 0.9 and 
                    h < image.shape[0] * 0.9 and
                    w * h > 400):  # Minimum area
                    
                    # Check aspect ratio (should be roughly square for grid cells)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    if aspect_ratio < 3:  # Not too elongated
                        cell_rects.append((x, y, w, h))
            
            if len(cell_rects) > best_rect_count:
                best_rect_count = len(cell_rects)
                cells = organize_cells_enhanced(image, cell_rects)
                if cells:
                    best_cells = cells
        
        return best_cells
        
    except Exception as e:
        logger.error(f"Enhanced contour detection error: {e}")
        return []

def detect_by_lines_enhanced(image, gray):
    """Enhanced line-based detection"""
    try:
        # Edge detection with different parameters
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges3 = cv2.Canny(gray, 70, 200)
        
        edges = cv2.bitwise_or(edges1, edges2)
        edges = cv2.bitwise_or(edges, edges3)
        
        # Dilate edges to connect broken lines
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                                minLineLength=30, maxLineGap=20)
        
        if lines is None:
            return []
        
        # Separate horizontal and vertical lines
        h_lines = []
        v_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if angle < 20 and length > 30:  # Horizontal
                h_lines.append((y1, length))
            elif angle > 70 and length > 30:  # Vertical
                v_lines.append((x1, length))
        
        # Cluster line positions
        h_positions = cluster_weighted_positions(h_lines, image.shape[0] * 0.05)
        v_positions = cluster_weighted_positions(v_lines, image.shape[1] * 0.05)
        
        # Need at least 2 lines in each direction
        if len(h_positions) < 2 or len(v_positions) < 2:
            return []
        
        # Sort positions
        h_positions = sorted(h_positions)
        v_positions = sorted(v_positions)
        
        # Remove outliers (lines that are too close to edges)
        h_positions = [p for p in h_positions if 5 < p < image.shape[0] - 5]
        v_positions = [p for p in v_positions if 5 < p < image.shape[1] - 5]
        
        if len(h_positions) < 2 or len(v_positions) < 2:
            return []
        
        # Extract cells
        cells = []
        for i in range(len(h_positions) - 1):
            row_cells = []
            for j in range(len(v_positions) - 1):
                y1 = max(0, h_positions[i] - 2)  # Small padding
                y2 = min(image.shape[0], h_positions[i+1] + 2)
                x1 = max(0, v_positions[j] - 2)
                x2 = min(image.shape[1], v_positions[j+1] + 2)
                
                if y2 > y1 and x2 > x1:
                    cell = image[y1:y2, x1:x2]
                    if cell.size > 0:
                        # Resize to standard size
                        cell = cv2.resize(cell, (100, 100))
                        row_cells.append(cell)
            
            if len(row_cells) > 0:
                cells.append(row_cells)
        
        return cells
        
    except Exception as e:
        logger.error(f"Enhanced line detection error: {e}")
        return []

def cluster_weighted_positions(lines, threshold):
    """Cluster lines with weights (length)"""
    if not lines:
        return []
    
    # Sort by position
    lines.sort(key=lambda x: x[0])
    
    clusters = []
    current_cluster = [lines[0]]
    
    for line in lines[1:]:
        if abs(line[0] - current_cluster[-1][0]) <= threshold:
            current_cluster.append(line)
        else:
            # Calculate weighted average
            total_weight = sum(w for _, w in current_cluster)
            weighted_sum = sum(p * w for p, w in current_cluster)
            clusters.append(int(weighted_sum / total_weight))
            current_cluster = [line]
    
    if current_cluster:
        total_weight = sum(w for _, w in current_cluster)
        weighted_sum = sum(p * w for p, w in current_cluster)
        clusters.append(int(weighted_sum / total_weight))
    
    return clusters

def detect_by_morphology(image, gray):
    """Detect grid using morphological operations"""
    try:
        # Apply morphological operations to find grid lines
        # Horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        grid_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # Threshold
        _, thresh = cv2.threshold(grid_lines, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours of grid cells
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        cell_rects = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter for potential cells
            if (w > 30 and h > 30 and 
                w < image.shape[1] * 0.5 and 
                h < image.shape[0] * 0.5):
                cell_rects.append((x, y, w, h))
        
        if len(cell_rects) > 4:
            return organize_cells_enhanced(image, cell_rects)
        
        return []
        
    except Exception as e:
        logger.error(f"Morphological detection error: {e}")
        return []

def organize_cells_enhanced(image, rects):
    """Enhanced cell organization with grid normalization"""
    if len(rects) < 4:
        return []
    
    try:
        # Sort by y-coordinate
        rects.sort(key=lambda r: r[1])
        
        # Group into rows
        rows = []
        current_row = [rects[0]]
        current_y = rects[0][1]
        threshold = image.shape[0] * 0.1
        
        for rect in rects[1:]:
            x, y, w, h = rect
            if abs(y - current_y) <= threshold:
                current_row.append(rect)
            else:
                current_row.sort(key=lambda r: r[0])
                rows.append(current_row)
                current_row = [rect]
                current_y = y
        
        if current_row:
            current_row.sort(key=lambda r: r[0])
            rows.append(current_row)
        
        # Determine expected number of columns (mode of row lengths)
        col_counts = [len(row) for row in rows]
        if not col_counts:
            return []
        
        from scipy import stats
        expected_cols = stats.mode(col_counts)[0][0] if len(col_counts) > 1 else col_counts[0]
        
        # Extract cell images
        cells = []
        for row in rows:
            row_cells = []
            for x, y, w, h in row:
                # Add adaptive padding
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.1)
                
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(image.shape[1], x + w + pad_x)
                y2 = min(image.shape[0], y + h + pad_y)
                
                cell = image[y1:y2, x1:x2]
                if cell.size > 0:
                    cell = cv2.resize(cell, (100, 100))
                    row_cells.append(cell)
            
            # Pad row if needed to match expected columns
            while len(row_cells) < expected_cols:
                row_cells.append(np.zeros((100, 100, 3), dtype=np.uint8))
            
            cells.append(row_cells)
        
        return cells
        
    except Exception as e:
        logger.error(f"Enhanced organization error: {e}")
        return []

def divide_into_grid_enhanced(image, gray=None):
    """Enhanced grid division with content analysis"""
    try:
        h, w = image.shape[:2]
        
        if gray is None:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
        
        # Analyze image to determine optimal grid size
        # Look for patterns that suggest grid lines
        edges = cv2.Canny(gray, 50, 150)
        
        # Count edge pixels in horizontal and vertical directions
        h_edge_count = np.sum(edges, axis=1)  # Horizontal projection
        v_edge_count = np.sum(edges, axis=0)  # Vertical projection
        
        # Find peaks that might be grid lines
        h_peaks = find_peaks(h_edge_count, height=np.mean(h_edge_count) * 1.5)
        v_peaks = find_peaks(v_edge_count, height=np.mean(v_edge_count) * 1.5)
        
        # Determine grid size based on peaks
        if len(h_peaks) > 3 and len(v_peaks) > 3:
            rows = len(h_peaks) - 1
            cols = len(v_peaks) - 1
        else:
            # Fallback to content-based size
            # Count potential digits to estimate grid size
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            digit_count = sum(1 for cnt in contours if cv2.contourArea(cnt) > 50)
            
            # Estimate grid size based on digit count
            if digit_count > 50:
                rows, cols = 10, 10
            elif digit_count > 30:
                rows, cols = 8, 8
            elif digit_count > 15:
                rows, cols = 6, 6
            else:
                rows, cols = 5, 5
        
        # Ensure reasonable grid size
        rows = min(max(rows, 2), 15)
        cols = min(max(cols, 2), 15)
        
        cell_h = h // rows
        cell_w = w // cols
        
        cells = []
        for i in range(rows):
            row_cells = []
            for j in range(cols):
                y1 = i * cell_h
                y2 = (i + 1) * cell_h if i < rows - 1 else h
                x1 = j * cell_w
                x2 = (j + 1) * cell_w if j < cols - 1 else w
                
                cell = image[y1:y2, x1:x2]
                cell = cv2.resize(cell, (100, 100))
                row_cells.append(cell)
            
            cells.append(row_cells)
        
        return cells
        
    except Exception as e:
        logger.error(f"Enhanced division error: {e}")
        return []

def find_peaks(data, height=None, distance=10):
    """Find peaks in 1D data"""
    peaks = []
    n = len(data)
    
    for i in range(1, n - 1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            if height is None or data[i] > height:
                # Check distance from previous peak
                if not peaks or i - peaks[-1] > distance:
                    peaks.append(i)
    
    return peaks

def ultimate_fallback(image):
    """Ultimate fallback method when everything else fails"""
    try:
        h, w = image.shape[:2]
        
        # Make a reasonable guess for grid size
        # Based on typical handwritten grids
        if h > 1000 or w > 1000:
            rows, cols = 10, 10
        elif h > 600 or w > 600:
            rows, cols = 8, 8
        elif h > 400 or w > 400:
            rows, cols = 6, 6
        else:
            rows, cols = 5, 5
        
        cell_h = h // rows
        cell_w = w // cols
        
        cells = []
        for i in range(rows):
            row_cells = []
            for j in range(cols):
                y1 = i * cell_h
                y2 = (i + 1) * cell_h if i < rows - 1 else h
                x1 = j * cell_w
                x2 = (j + 1) * cell_w if j < cols - 1 else w
                
                cell = image[y1:y2, x1:x2]
                cell = cv2.resize(cell, (100, 100))
                row_cells.append(cell)
            
            cells.append(row_cells)
        
        logger.info(f"⚠️ Using ultimate fallback: {rows}x{cols} grid")
        return cells
        
    except Exception as e:
        logger.error(f"Ultimate fallback error: {e}")
        return []