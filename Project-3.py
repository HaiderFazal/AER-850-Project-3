import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

#Step 1: Object Masking

img_path = "/content/drive/MyDrive/AER-850-Project-3/motherboard_image.JPEG"
img_real = cv2.imread(img_path)

img = cv2.rotate(img_real, cv2.ROTATE_90_CLOCKWISE)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur_img = cv2.GaussianBlur(gray_img, (9, 9), 0) 

_, threshold_img = cv2.threshold(blur_img, 120, 255, cv2.THRESH_BINARY) 

edge_detect = cv2.Canny(threshold_img, 50, 150)  
edge_detect = cv2.dilate(edge_detect, np.ones((5, 5), np.uint8), iterations=1) 

contour_detection, _ = cv2.findContours(edge_detect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contour_detection:
    largest_contour = max(contour_detection, key=cv2.contourArea)
    mask = np.zeros(gray_img.shape, dtype="uint8")  
    cv2.drawContours(mask, [largest_contour], -1, 255, -1) 

    masked_img = cv2.bitwise_and(img, img, mask=mask)

    extracted_pcb = cv2.bitwise_and(img, img, mask=mask)
    extracted_path = "/content/drive/MyDrive/AER-850-Project-3/extracted_pcb_clean.JPEG"
    cv2.imwrite(extracted_path, extracted_pcb)

else:
    raise Exception("No contours found - check edge detection parameters and input image quality.")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Edge Detection")
plt.imshow(edge_detect, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Isolated Motherboard")
plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()

# Step 2: YOLOv8 Training

data_yaml_path = "/content/drive/MyDrive/AER-850-Project-3/data/data.yaml"
model = YOLO('yolov8n.pt') 

train_results = model.train(
    data=data_yaml_path,   
    epochs=125,        # <200 recommended
    batch=4,              # 4-5 recommended 
    imgsz=1000,             
    name='haider_model' 
)

train_results=model.val(data=data_yaml_path)
print("Model Successfully Trained")

# Step 3: YOLOv8 Evaluation

train_results=model.predict(source="/content/drive/MyDrive/AER-850-Project-3/data/evaluation",save=True)