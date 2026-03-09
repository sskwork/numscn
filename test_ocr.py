import cv2
import numpy as np
from ocr.ocr_model import recognize_digit

# Create test digits
for digit in range(10):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(img, str(digit), (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imwrite(f"test_{digit}.jpg", img)
    
    result = recognize_digit(img)
    print(f"Digit {digit}: Recognized as '{result}'")