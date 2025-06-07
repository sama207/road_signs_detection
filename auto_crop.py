from ultralytics import YOLO
from PIL import Image
import os
import shutil

# --- Configuration ---
yolo_model_path = "YOLO8m-Experiments/left_right_signs_train/weights/best.pt"
input_images_dir = "left_right_signs/images/train"  # Folder with full-size images
output_crops_dir = "cropped_cnn_dataset/unlabeled/"  # We'll manually label left/right later

# Create output directory
os.makedirs(output_crops_dir, exist_ok=True)

# Load YOLO model
model = YOLO(yolo_model_path)

# Loop through all images
image_filenames = [f for f in os.listdir(input_images_dir) if f.lower().endswith(('.jpg', '.png'))]

for filename in image_filenames:
    image_path = os.path.join(input_images_dir, filename)
    image = Image.open(image_path).convert("RGB")

    # Inference
    results = model(image)

    for i, box in enumerate(results[0].boxes):
        class_id = int(box.cls.item())
        class_name = model.names[class_id]

        if class_name == "right_right":  # Only crop detected turn signs
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = image.crop((x1, y1, x2, y2))

            # Save crop for manual labeling
            save_path = os.path.join(output_crops_dir, f"{filename[:-4]}_crop{i}.jpg")
            cropped.save(save_path)

print("✅ Cropping done. Please manually sort into 'left/' and 'right/' folders.")
