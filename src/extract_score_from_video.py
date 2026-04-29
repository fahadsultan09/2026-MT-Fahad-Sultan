import cv2
import pytesseract
from PIL import Image
import os

def extract_text(img, config='--psm 7'):
    if img is None or img.size == 0:
        return ""
    # Upscale 3x to help Tesseract with low-res text
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    pil_img = Image.fromarray(img)
    return pytesseract.image_to_string(pil_img, config=config).strip()

image = cv2.imread('./scoreboard_log/point_78.png')
if image is None:
    exit("Error: Image not found")

# Preprocessing: Clean up noise
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY) 

# COORDINATES FOR 314x56 
# Format: [y_start:y_end, x_start:x_end]
# We split the 56px height into two rows (~28px each)
p1_name_roi   = thresh[0:28, 20:230]   # Skips the '5', gets 'JABEUR TUN'
p1_games_roi  = thresh[0:28, 250:285]  # The '0'
p1_points_roi = thresh[0:28, 285:314]  # The '30'

p2_name_roi   = thresh[28:56, 20:230]   # 'TOMLJANOVIC AUS'
p2_games_roi  = thresh[28:56, 250:285]  # The '1'
p2_points_roi = thresh[28:56, 285:314]  # The '15'

# OPTIONAL: Save crops to disk to verify coordinates
# cv2.imwrite('debug_p1_name.png', p1_name_roi)
# cv2.imwrite('debug_p1_games.png', p1_games_roi)

results = {
    "p1_name": extract_text(p1_name_roi),
    "p1_games": extract_text(p1_games_roi, config='--psm 10 -c tessedit_char_whitelist=0123456789'),
    "p1_points": extract_text(p1_points_roi, config='--psm 10 -c tessedit_char_whitelist=0123456789AD'),
    "p2_name": extract_text(p2_name_roi),
    "p2_games": extract_text(p2_games_roi, config='--psm 10 -c tessedit_char_whitelist=0123456789'),
    "p2_points": extract_text(p2_points_roi, config='--psm 10 -c tessedit_char_whitelist=0123456789AD')
}

print(results)