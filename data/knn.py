import cv2

import pytesseract


image = cv2.imread('./1.jpeg', cv2.IMREAD_COLOR)

print(pytesseract.image_to_string(image))


