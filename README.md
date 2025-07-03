# 🛑 Indian Traffic Sign Detection using Faster R-CNN
This project implements a deep learning model for detecting Indian traffic signboards using Faster R-CNN. The model is trained on a custom dataset annotated in COCO format and aims to accurately localize and classify various traffic signs under real-world conditions.

# 🚦 Project Overview
Model: Faster R-CNN with ResNet-50 + FPN backbone

Framework: PyTorch (Torchvision Models)

Dataset: Indian Traffic Signboards v2 (COCO format, 56 classes)

Goal: Object detection (bounding box + class)

Use Case: Intelligent Transport Systems, Driver Assistance, Traffic Monitoring

🗂️ Dataset Summary
📦 Total Images: 7,833

📁 Splits: train/, valid/, test/

🏷️ Classes: 56 Indian traffic sign types (e.g., No Entry, U-Turn, Speed Limits, School Ahead)

Annotation Format:
Format: COCO JSON

Each image has multiple objects annotated with bounding boxes and class labels

🧠 Model Architecture
📌 Base: fasterrcnn_resnet50_fpn(pretrained=True)

🔄 Transfer Learning: COCO weights fine-tuned on traffic sign dataset

🎯 Detection Head: Updated to match 56 traffic sign classes

🧱 FPN (Feature Pyramid Network): Helps detect signs of various scales

⚙️ Training Configuration
Parameter	Value
Optimizer	SGD
Learning Rate	0.005
Momentum	0.9
Weight Decay	0.0005
Epochs	25–30
Batch Size	4
Image Size	512×512
Augmentations	Flip, Rotation, Blur, Brightness

💾 Checkpoints saved after every epoch to /checkpoints/
✅ Final model stored at Trained_Models/final_frcnn_model.pth

📈 Evaluation Results
Evaluation was done on the test set (15% split). 

📊 Visual results show consistent detection even with occlusion, blur, and small-scale signs.


# 🚀 How to Run

## Clone the repo:
git clone https://github.com/Hemu-29/Indian-Traffic-Sign-Detection.git
cd Indian-Traffic-Sign-Detection

## Install dependencies:
pip install -r requirements.txt

## Run training:
python train.py

## Run inference:
python predict.py --image your_image.jpg

## 📦 Directory Structure
├── train.py                     # Model training script

├── predict.py                   # Inference on single image
├── metrics.py                   # Validate the trained model
├── dataset/                     # train/valid/test + annotations
├── checkpoints/                 # Intermediate saved models
├── Trained_Models/
│   └── final_frcnn_model.pth    # Final trained model
├── README.md

## 📜 License
This project is licensed under the MIT License.
The dataset is also MIT-licensed and publicly available on Roboflow & Kaggle.

## 🙋‍♂️ Author
Hemanth Ande
B.Tech CSE (AI & ML)
DRK Institute of Science and Technology

GitHub: [@Hemu-29](https://github.com/Hemu-29)

Kaggle: [@HemanthAnde](https://www.kaggle.com/hemanthande24)

Roboflow Dataset: [Indian Traffic Signboards](https://app.roboflow.com/hemanth-dtwzo/indian-traffic-signboards-zcmku-pug8l/2)

# DATASET
# 🛑 Indian Traffic Signboards Dataset (v2) - COCO Format

This dataset contains **7,833 labeled images** of **Indian traffic signboards**, prepared for object detection tasks and exported in **COCO format** using [Roboflow](https://roboflow.com). It’s ideal for training models such as **Faster R-CNN**, **YOLOv8 (with COCO loader)**, and **Detectron2**.

---
## Get Dataset
Kaggle Dataset Link:
- https://www.kaggle.com/datasets/hemanthande24/custom-indian-traffic-sign-dataset-coco

## 📁 Dataset Structure

The dataset includes:

- `train/`: Training images
- `valid/`: Validation images
- `test/`: Testing images

Along with:
- `_annotations.coco.json` files for each split, containing annotations in standard [COCO format](https://cocodataset.org/#format-data)

Each JSON annotation file includes:
- `images`: list of image metadata (filename, dimensions, id)
- `annotations`: bounding box data, `category_id`, image id, etc.
- `categories`: all traffic sign classes with their IDs

---

## 🏷️ Classes (56 Total)

See full distribution below.

---

## 📊 Class Distribution

The dataset includes the following traffic sign classes and the number of annotations per class:

| #  | Class Name                      | Count |
|----|--------------------------------|-------|
| 1  | All Motor Vehicle Prohibited   | 73    |
| 2  | Axle Load Limit                | 65    |
| 3  | Compulsory Ahead               | 75    |
| 4  | Compulsory Keep Left           | 76    |
| 5  | Compulsory Keep Right          | 71    |
| 6  | Compulsory Turn Left Ahead     | 70    |
| 7  | Compulsory Turn Right Ahead    | 67    |
| 8  | Cross Road                     | 77    |
| 9  | Dangerous Dip                  | 52    |
| 10 | Falling Rocks                  | 75    |
| 11 | Gap in Median                  | 60    |
| 12 | Give Way                       | 83    |
| 13 | Guarded Level Crossing         | 62    |
| 14 | Height Limit                   | 63    |
| 15 | Horn Prohibited                | 83    |
| 16 | Hospital Ahead                 | 66    |
| 17 | Hump or Rough Road             | 77    |
| 18 | Left Hand Curve                | 78    |
| 19 | Left Reverse Bend              | 70    |
| 20 | Left Turn Prohibited           | 57    |
| 21 | Loose Gravel                   | 59    |
| 22 | Men at Work                    | 77    |
| 23 | Narrow Bridge Ahead            | 81    |
| 24 | Narrow Road Ahead              | 68    |
| 25 | No Entry                       | 80    |
| 26 | No Parking                     | 86    |
| 27 | No Stopping or Standing        | 60    |
| 28 | Overtaking Prohibited          | 66    |
| 29 | Pass Either Side               | 69    |
| 30 | Pedestrian Crossing            | 79    |
| 31 | Petrol Pump Ahead              | 57    |
| 32 | Quay Side or River Bank        | 80    |
| 33 | Restriction Ends               | 75    |
| 34 | Right Hand Curve               | 77    |
| 35 | Right Reverse Bend             | 70    |
| 36 | Right Turn Prohibited          | 68    |
| 37 | Road Widens Ahead              | 40    |
| 38 | Roundabout                     | 62    |
| 39 | School Ahead                   | 68    |
| 40 | Side Road Left                 | 55    |
| 41 | Side Road Right                | 42    |
| 42 | Slippery Road                  | 63    |
| 43 | Speed Limit 100                | 59    |
| 44 | Speed Limit 30                 | 68    |
| 45 | Speed Limit 50                 | 64    |
| 46 | Staggered Intersection         | 49    |
| 47 | Steep Ascent                   | 44    |
| 48 | Steep Descent                  | 48    |
| 49 | Stop                           | 72    |
| 50 | Straight Prohibited            | 22    |
| 51 | T Intersection                 | 46    |
| 52 | U Turn                         | 59    |
| 53 | U Turn Prohibited              | 61    |
| 54 | Unguarded Level Crossing       | 53    |
| 55 | Width Limit                    | 42    |
| 56 | Y Intersection                 | 54    |

---

## 🧪 Augmentations Used (via Roboflow)

To improve model generalization, each image was augmented with the following:

- ✅ 50% probability of horizontal flip  
- ✅ Random 90° rotation (none, clockwise, counter-clockwise)  
- ✅ Random rotation between −15° and +15°  
- ✅ Brightness variation between −15% and +15%  
- ✅ Gaussian blur (0 to 2.5 px)

Each original image was augmented to create 3 versions, resulting in **7,833 total images**.

---

## ⚙️ Preprocessing

- Auto-orientation (EXIF stripped)
- All images resized to **512×512 pixels** using stretch scaling
- Exported in **COCO JSON format**, compatible with PyTorch, TensorFlow, Detectron2, MMDetection, and more.

---

## 📅 Dataset Details

- Version: v2  
- Created: June 16, 2025  
- Exported: June 17, 2025 via Roboflow  
- Format: COCO  
- Total images: 7,833  
- Author: Hemanth Ande  

---

## 📜 License

**MIT License**  
This dataset is free for personal, academic, and commercial use. You may use, modify, and redistribute it under the terms of the MIT license.

[Read the full license →](https://opensource.org/licenses/MIT)

---

## 🙋‍♂️ About the Author

**Hemanth Ande**  
- B.Tech CSE (AI & ML)  
- DRK Institute of Science and Technology  
- GitHub: [@Hemu-29](https://github.com/Hemu-29)  
- Roboflow: [Indian Traffic Signboards](https://universe.roboflow.com/hemanth-dtwzo/indian-traffic-signboards-zcmku-pug8l)

---

📦 *This dataset was built, augmented, and exported using [Roboflow](https://roboflow.com) — the leading platform for computer vision datasets.*
