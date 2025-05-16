from ultralytics import YOLO
import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def checking_gpu():
    print(torch.cuda.is_available())  # Should be True
    print(torch.cuda.get_device_name(0))  # Prints GPU name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    torch.cuda.is_available()

    return device

def model_training(model,device):
    results = model.train(
        data="config.yaml",  
        epochs=150,
        batch=15,
        name="road_signs_train2",
        device=device
    )

    results = model.val()

    return results

def main():
    model = YOLO("yolov8n.pt")  
    device=checking_gpu()

    results = model_training(model,device)

    print('results : \n', results)

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


