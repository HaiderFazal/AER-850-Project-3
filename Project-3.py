import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

#Step 1: Object Masking

# Load the image from the specified path
image_path = "/content/drive/MyDrive/AER-850-Project-3/motherboard_image.JPEG"
image_real = cv2.imread(image_path)

# Rotate the image for better orientation
image = cv2.rotate(image_real, cv2.ROTATE_90_CLOCKWISE)

# Convert to grayscale for processing
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to smooth out the image, reducing noise and details
image_blur = cv2.GaussianBlur(gray_image, (9, 9), 0)  # Adjusted kernel size for less aggressive blurring

# Apply thresholding to segment the motherboard from the background
_, threshold_image = cv2.threshold(image_blur, 120, 255, cv2.THRESH_BINARY)  # Adjusted threshold for better differentiation

# Detect edges using the Canny method
edge_detection = cv2.Canny(threshold_image, 50, 150)  # Adjusted thresholds for edge detection
edge_detection = cv2.dilate(edge_detection, np.ones((5, 5), np.uint8), iterations=1)  # Reduced dilation

# Detect contours from the edge-detected image
contour_detection, _ = cv2.findContours(edge_detection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Check if contours were found and select the largest contour
if contour_detection:
    largest_contour = max(contour_detection, key=cv2.contourArea)
    mask = np.zeros(gray_image.shape, dtype="uint8")  # Ensure mask is the same size as the grayscale image
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)  # Fill the contour

    # Apply the mask to the original (color) image to isolate the motherboard
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Extract the motherboard using the mask
    extracted_pcb = cv2.bitwise_and(image, image, mask=mask)
    extracted_path = "/content/drive/MyDrive/AER-850-Project-3/extracted_pcb_clean.JPEG"
    cv2.imwrite(extracted_path, extracted_pcb)

else:
    raise Exception("No contours found - check edge detection parameters and input image quality.")

# Visualizing the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Edge Detection")
plt.imshow(edge_detection, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Isolated Motherboard")
plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()

# Step 2: YOLOv8 Training

# Path to your dataset's YAML file which contains training and validation set paths
data_yaml_path = "/content/drive/MyDrive/AER-850-Project-3/data/data.yaml"

# Define the training parameters
model = YOLO('yolov8n.pt')  # Load the YOLOv8 Nano pre-trained model

# Start training the model
train_results = model.train(
    data=data_yaml_path,   # Path to dataset configuration file
    epochs=2,            # Maximum number of training epochs
    batch=4,              # Number of images per training batch
    imgsz=1000,             # Minimum recommended input size for YOLOv8 Nano
    name='haider_model'  # Name for the trained model
)
train_results=model.val(data=data_yaml_path)
print("Training complete. Model saved and ready for deployment or further evaluation.")

# Step 3: YOLOv8 Evaluation

train_results=model.predict(source="/content/drive/MyDrive/AER-850-Project-3/data/evaluation",save=True)