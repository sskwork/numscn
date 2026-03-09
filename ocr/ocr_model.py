import cv2
import numpy as np
import os
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model with fallback options
model = None
model_paths = [
    "digit_model_fallback.h5",
    "digit_model.h5",
    "models/digit_model.h5",
    "ocr/digit_model.h5"
]

def load_best_model():
    """Load the best available model"""
    global model
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                # Try different import methods
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(path)
                    logger.info(f"✅ Model loaded from {path} (TensorFlow)")
                    return True
                except:
                    try:
                        from tensorflow.keras.models import load_model
                        model = load_model(path)
                        logger.info(f"✅ Model loaded from {path} (Keras)")
                        return True
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load {path}: {e}")
                        continue
            except Exception as e:
                logger.warning(f"⚠️ Error loading {path}: {e}")
                continue
    
    logger.warning("⚠️ No model found, using rule-based fallback")
    return False

# Load model on import
load_best_model()

def recognize_digit(digit_img, return_confidence=False):
    """
    Enhanced digit recognition with confidence scoring
    Returns digit as string or (digit, confidence) if return_confidence=True
    """
    if digit_img is None or digit_img.size == 0:
        return ("", 0) if return_confidence else ""
    
    try:
        # Prepare image with multiple preprocessing options
        prepared_images = prepare_digit_multiple(digit_img)
        
        best_digit = ""
        best_confidence = 0
        all_predictions = []
        
        for prepared in prepared_images:
            if model is not None:
                # Use neural network
                digit, confidence = predict_with_model(prepared)
            else:
                # Use rule-based fallback
                digit, confidence = recognize_rule_based(prepared)
            
            if confidence > best_confidence:
                best_digit = digit
                best_confidence = confidence
            
            if confidence > 0.5:
                all_predictions.append(digit)
        
        # If we have multiple predictions, use voting
        if len(all_predictions) > 1:
            vote_result = Counter(all_predictions).most_common(1)[0]
            if vote_result[1] > len(all_predictions) / 2:  # Majority vote
                best_digit = vote_result[0]
                best_confidence = max(best_confidence, 0.9)
        
        logger.debug(f"  Recognized: {best_digit} (confidence: {best_confidence:.2f})")
        
        if return_confidence:
            return best_digit, best_confidence
        else:
            return best_digit if best_confidence > 0.4 else ""  # Lower threshold for better recall
            
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        return ("", 0) if return_confidence else ""

def prepare_digit_multiple(digit_img):
    """Prepare digit image using multiple methods"""
    prepared_images = []
    
    try:
        # Convert to grayscale if needed
        if len(digit_img.shape) == 3:
            gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = digit_img.copy()
        
        # Method 1: Standard resize
        img1 = cv2.resize(gray, (28, 28))
        img1 = img1.astype(np.float32) / 255.0
        prepared_images.append(img1.reshape(1, 28, 28, 1))
        
        # Method 2: Inverted (for white on black)
        img2 = 1.0 - img1
        prepared_images.append(img2.reshape(1, 28, 28, 1))
        
        # Method 3: Thresholded
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img3 = cv2.resize(thresh, (28, 28))
        img3 = img3.astype(np.float32) / 255.0
        prepared_images.append(img3.reshape(1, 28, 28, 1))
        
        # Method 4: Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        img4 = cv2.resize(enhanced, (28, 28))
        img4 = img4.astype(np.float32) / 255.0
        prepared_images.append(img4.reshape(1, 28, 28, 1))
        
    except Exception as e:
        logger.error(f"Error preparing digit: {e}")
        # Fallback to simple preparation
        if len(digit_img.shape) == 3:
            gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = digit_img.copy()
        img = cv2.resize(gray, (28, 28))
        img = img.astype(np.float32) / 255.0
        prepared_images = [img.reshape(1, 28, 28, 1)]
    
    return prepared_images

def predict_with_model(prepared_img):
    """Predict digit using neural network model"""
    try:
        predictions = model.predict(prepared_img, verbose=0)[0]
        digit = np.argmax(predictions)
        confidence = np.max(predictions)
        
        # Apply confidence calibration
        if confidence > 0.9:
            confidence = min(confidence * 1.05, 1.0)  # Boost high confidence
        elif confidence < 0.3:
            confidence = confidence * 0.8  # Reduce low confidence
        
        return str(digit), confidence
        
    except Exception as e:
        logger.error(f"Model prediction error: {e}")
        return "", 0

def recognize_rule_based(digit_img):
    """Rule-based digit recognition as fallback"""
    try:
        # Extract features
        if len(digit_img.shape) == 3:
            gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = digit_img.copy()
        
        # Ensure 28x28
        if gray.shape != (28, 28):
            gray = cv2.resize(gray, (28, 28))
        
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Extract features
        features = extract_features(binary)
        
        # Simple rule-based classification
        return classify_by_features(features)
        
    except Exception as e:
        logger.error(f"Rule-based recognition error: {e}")
        return "", 0

def extract_features(binary):
    """Extract features from binary digit image"""
    features = {}
    
    # 1. Pixel density
    features['density'] = np.sum(binary > 0) / (28 * 28)
    
    # 2. Vertical projection (sum per column)
    features['v_proj'] = np.sum(binary > 0, axis=0)
    
    # 3. Horizontal projection (sum per row)
    features['h_proj'] = np.sum(binary > 0, axis=1)
    
    # 4. Center of mass
    y_coords, x_coords = np.where(binary > 0)
    if len(x_coords) > 0:
        features['center_x'] = np.mean(x_coords) / 28
        features['center_y'] = np.mean(y_coords) / 28
    else:
        features['center_x'] = 0.5
        features['center_y'] = 0.5
    
    # 5. Aspect ratio of bounding box
    if len(x_coords) > 0:
        features['width'] = (np.max(x_coords) - np.min(x_coords)) / 28
        features['height'] = (np.max(y_coords) - np.min(y_coords)) / 28
        features['aspect'] = features['height'] / features['width'] if features['width'] > 0 else 1
    else:
        features['width'] = 0
        features['height'] = 0
        features['aspect'] = 1
    
    # 6. Symmetry scores
    features['h_symmetry'] = compute_horizontal_symmetry(binary)
    features['v_symmetry'] = compute_vertical_symmetry(binary)
    
    # 7. Number of holes (for digits like 8, 0)
    features['holes'] = count_holes(binary)
    
    # 8. Endpoints and junctions
    features['endpoints'], features['junctions'] = count_endpoints_junctions(binary)
    
    return features

def compute_horizontal_symmetry(binary):
    """Compute horizontal symmetry score"""
    h, w = binary.shape
    left = binary[:, :w//2]
    right = np.fliplr(binary[:, w//2:])
    
    # Make sure shapes match
    min_w = min(left.shape[1], right.shape[1])
    if min_w > 0:
        left = left[:, :min_w]
        right = right[:, :min_w]
        return np.sum(left == right) / (h * min_w)
    return 0

def compute_vertical_symmetry(binary):
    """Compute vertical symmetry score"""
    h, w = binary.shape
    top = binary[:h//2, :]
    bottom = np.flipud(binary[h//2:, :])
    
    # Make sure shapes match
    min_h = min(top.shape[0], bottom.shape[0])
    if min_h > 0:
        top = top[:min_h, :]
        bottom = bottom[:min_h, :]
        return np.sum(top == bottom) / (min_h * w)
    return 0

def count_holes(binary):
    """Count number of holes in digit (for 0, 8, etc.)"""
    # Invert and find contours
    inverted = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count internal holes
    holes = 0
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 20:  # Minimum hole size
            holes += 1
    
    return max(0, holes - 1)  # Subtract 1 for outer contour

def count_endpoints_junctions(binary):
    """Count endpoints and junctions using skeletonization"""
    # Skeletonize
    skeleton = cv2.ximgproc.thinning(binary)
    
    # Define kernels for endpoint and junction detection
    endpoint_kernel = np.array([[1,1,1],
                                 [1,0,1],
                                 [1,1,1]], dtype=np.uint8)
    
    # Count endpoints
    endpoints = 0
    junctions = 0
    
    h, w = skeleton.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if skeleton[i, j] > 0:
                # Count neighbors
                neighbors = np.sum(skeleton[i-1:i+2, j-1:j+2] > 0) - 1
                
                if neighbors == 1:
                    endpoints += 1
                elif neighbors >= 3:
                    junctions += 1
    
    return endpoints, junctions

def classify_by_features(features):
    """Classify digit based on extracted features"""
    
    # Rule-based classification
    if features['holes'] >= 2:
        return "8", 0.8
    elif features['holes'] == 1:
        if features['aspect'] > 1.2:
            return "0", 0.7
        else:
            return "0", 0.6
    
    # Based on aspect ratio
    if features['aspect'] > 2.0:
        return "1", 0.6
    elif features['aspect'] < 0.5:
        # Could be 1 written horizontally or something else
        pass
    
    # Based on symmetry
    if features['h_symmetry'] > 0.8 and features['v_symmetry'] > 0.8:
        return "0", 0.5
    
    # Based on density
    if features['density'] < 0.1:
        return "1", 0.5
    elif features['density'] > 0.6:
        return "8", 0.4
    
    # Based on endpoints
    if features['endpoints'] == 2:
        if features['center_y'] < 0.4:
            return "7", 0.4
        else:
            return "4", 0.3
    elif features['endpoints'] == 1:
        return "6", 0.3
    elif features['endpoints'] == 0:
        return "8", 0.3
    
    # Default
    return "", 0

def batch_recognize_digits(digit_images):
    """
    Recognize multiple digits in batch for efficiency
    Returns list of (digit, confidence) tuples
    """
    results = []
    
    for img in digit_images:
        digit, conf = recognize_digit(img, return_confidence=True)
        results.append((digit, conf))
    
    return results