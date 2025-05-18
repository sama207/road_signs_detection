from ultralytics import YOLO
import os
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def checking_gpu():
    print(torch.cuda.is_available())  # Should be True
    print(torch.cuda.get_device_name(0))  # Prints GPU name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    torch.cuda.is_available()

    return device

def model_training(model,device):
    results = model.train(
        project='YOLOv8-Experiments',
        name="road_signs_train2",
        data="config.yaml",        
        optimizer='AdamW',
        augment=True,
        fliplr=0.0,  
        patience=10,
        epochs=50,                 # Number of epochs
        imgsz=800,                 # Image size
        batch=16,  
        device=device,  # disable left-right flipsoo the left and right images wont get confused
        exist_ok=True,
        degrees=5,  # small tilt, like camera shake
        translate=0.1,  # slight shift
        scale=0.5,     
    )

    results = model.val()

    # Print specific metrics
    print("Class indices with average precision:", results.ap_class_index)
    print("Average precision for all classes:", results.box.all_ap)
    print("Average precision:", results.box.ap)
    print("Average precision at IoU=0.50:", results.box.ap50)
    print("Class indices for average precision:", results.box.ap_class_index)
    print("Class-specific results:", results.box.class_result)
    print("F1 score:", results.box.f1)
    print("F1 score curve:", results.box.f1_curve)
    print("Overall fitness score:", results.box.fitness)
    print("Mean average precision:", results.box.map)
    print("Mean average precision at IoU=0.50:", results.box.map50)
    print("Mean average precision at IoU=0.75:", results.box.map75)
    print("Mean average precision for different IoU thresholds:", results.box.maps)
    print("Mean results for different metrics:", results.box.mean_results)
    print("Mean precision:", results.box.mp)
    print("Mean recall:", results.box.mr)
    print("Precision:", results.box.p)
    print("Precision curve:", results.box.p_curve)
    print("Precision values:", results.box.prec_values)
    print("Specific precision metrics:", results.box.px)
    print("Recall:", results.box.r)
    print("Recall curve:", results.box.r_curve)
    return results


def verify_yolo_labels_random(image_dir, label_dir, class_names, sample_limit=5):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if len(image_files) == 0:
        print("No images found in the directory.")
        return

    # Select random sample
    random_images = random.sample(image_files, min(sample_limit, len(image_files)))

    for img_file in random_images:
        image_path = os.path.join(image_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            print(f"No label for {img_file}, skipping.")
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        w, h = image.size

        with open(label_path, "r") as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.strip().split())
                cls = int(cls)
                x1 = (x - bw / 2) * w
                y1 = (y - bh / 2) * h
                x2 = (x + bw / 2) * w
                y2 = (y + bh / 2) * h
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1), class_names[cls], fill="white")

        plt.figure(figsize=(6, 6))
        plt.title(f"Labeled: {img_file}")
        plt.imshow(image)
        plt.axis("off")
        plt.show()


def main():
    # verify_yolo_labels_random(
    # image_dir="C:/Users/ai_wo/OneDrive/Desktop/road_signs_detection/Data2/images/train",
    # label_dir="C:/Users/ai_wo/OneDrive/Desktop/road_signs_detection/Data2/labels/train",
    # class_names=["stop", "right", "left"],
    # sample_limit=5
    # )
    model = YOLO("yolov11s.pt")  
    device=checking_gpu()

    results = model_training(model,device)


if __name__ == "__main__":
    main()

# results = model("C:/Users/ai_wo/OneDrive/Desktop/car_detection_project/Data/images/test/4aa2611e-eb30-40ce-9f8c-56bd19c510b7.jpg")

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk


