# Road Signs Detection 🚦🛑  
Repo: `sama207/road_signs_detection`

A student-friendly computer vision project that builds an **end-to-end road sign pipeline**:

1) **Detect** road signs in images using **YOLOv8** (bounding boxes).  
2) **Crop** detected signs automatically to build a clean classification dataset.  
3) **Classify** cropped signs (e.g., **Left vs Right** direction signs) using a **CNN (ResNet18 fine-tuning)**.

This repo contains notebooks for experiments + scripts for automation, along with trained weights for quick testing.

---

## ✨ What’s inside (high-level)

- **YOLO detection** (notebook): `BB_detection_YOLO.ipynb`  
- **Two-stage detect → crop → classify pipeline**: `classification_using_YOLO_CNN.ipynb`  
- **CNN training / evaluation (Left vs Right)**: `left_right_classification_CNN.ipynb`  
- **Auto cropping utility**: `auto_crop.py`  
- **Config**: `config.yaml`  
- **Trained models**:
  - `yolov8m.pt` (YOLO base weights)
  - `resnet18_left_right_best.pth` (best CNN weights for left/right)
- **Folders** (may be large):
  - `runs/detect/` (YOLO results)
  - `classified_output/` (classification outputs)
  - `cropped_cnn_dataset/` (cropped dataset)
  - `left_right_signs/` and `YOLO8m-Experiments/left_right_signs_train/` (dataset/experiments)

> If you plan to push changes to GitHub, avoid committing large generated folders (`runs/`, datasets, model weights) unless you really need them.

---

## ✅ Requirements

### Option A — pip (simple)
Create and activate a virtual environment, then:

```bash
pip install ultralytics opencv-python torch torchvision numpy pandas matplotlib pillow scikit-learn pyyaml
```

### Option B — use your existing environment files
If your repo includes them later, you can install from:

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

---

## 📦 Quick Start (run the pipeline)

### 1) Detect road signs (YOLOv8)
You can run detection using Ultralytics directly:

```bash
yolo detect predict model=yolov8m.pt source=PATH_TO_IMAGES
```

This creates outputs under `runs/detect/...`.

If you trained a custom detector, replace `yolov8m.pt` with your trained weights.

---

### 2) Crop detected signs (build classification dataset)
This repo includes an automation script:

```bash
python auto_crop.py
```

Typical behavior (conceptually):
- Reads YOLO predictions (bounding boxes)
- Crops each sign region from the original image
- Saves crops into a dataset folder (e.g., `cropped_cnn_dataset/`)

> If your `auto_crop.py` expects specific folders/paths, edit `config.yaml` (or variables inside the script).

---

### 3) Classify cropped signs (Left vs Right)
If you want to **run inference** using the trained CNN:

- Make sure the weight file exists:
  - `resnet18_left_right_best.pth`

Then use the notebook:
- `classification_using_YOLO_CNN.ipynb`

or add a small inference script (recommended if you want CLI usage).

---

## 🧪 Notebooks (how to use)

### `BB_detection_YOLO.ipynb`
Use this to:
- Load YOLOv8
- Run predictions
- Visualize bounding boxes
- Save results

### `left_right_classification_CNN.ipynb`
Use this to:
- Prepare cropped dataset
- Fine-tune ResNet18 for left/right
- Evaluate accuracy
- Save best checkpoint (`.pth`)

### `classification_using_YOLO_CNN.ipynb`
Use this when you want the **full pipeline**:
- Detect → crop → classify
- Produce organized outputs under `classified_output/`

---

## 🗂️ Suggested data layout

You can organize your input images like:

```
road_signs_detection/
  Data2/
    images/
      img1.jpg
      img2.jpg
```

Cropped dataset example:

```
cropped_cnn_dataset/
  train/
    left/
    right/
  val/
    left/
    right/
  test/
    left/
    right/
```

If your dataset folders are different, that’s totally fine — just update paths in the notebooks / `config.yaml`.

---

## 📊 Outputs

- **YOLO detections**: `runs/detect/`
- **Cropped signs**: `cropped_cnn_dataset/`
- **Final classified results**: `classified_output/`

---

## 🔥 Tips (real-world)
- If detection is good but classification is weak:
  - balance the cropped classes (left/right)
  - add augmentation (flip, blur, brightness)
  - verify crop quality (tight boxes matter a lot)
- If you get many wrong crops:
  - increase YOLO confidence threshold
  - clean labels / retrain detector with more data

---

## 🧹 GitHub push safety (avoid huge files)
Add something like this to `.gitignore`:

```gitignore
# datasets / outputs
Data2/
cropped_cnn_dataset/
classified_output/
runs/

# large weights (optional)
*.pt
*.pth

# node (if any UI experiments)
node_modules/
```

---

## 📌 Credits
- YOLOv8 by Ultralytics (for object detection)
- ResNet18 (transfer learning backbone for classification)

---

## 👩‍💻 Author
Created by **sama207** — built for learning + experimenting with a full road-sign CV pipeline.
