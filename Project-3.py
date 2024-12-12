import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the image
image_path = "C:/Users/Saira/Desktop/Uni/Year 4/Sem 1/AER-850/Project-3/motherboard_image.JPEG"  # Replace with the actual image file path
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area (remove smaller noise)
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

# Create a mask for the detected object
mask = np.zeros_like(gray)
cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

# Apply the mask to the original image
extracted_image = cv2.bitwise_and(image, image, mask=mask)

# Display the result
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 2)
plt.title("Thresholded Image")
plt.imshow(thresholded, cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Extracted Image")
plt.imshow(cv2.cvtColor(extracted_image, cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()

