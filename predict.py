import sys
import cv2
import time
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QHBoxLayout
)
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# -------------------------------
# CLASS NAMES
# -------------------------------
CLASS_NAMES = [ "__background__", "all motor vehicle prohibited", "axle load limit", "compulsary ahead",
    "compulsary keep left", "compulsary keep right", "compulsary turn left ahead", "compulsary turn right ahead",
    "cross road", "dangerous dip", "falling rocks", "gap in median", "give way", "guarded level crossing",
    "height limit", "horn prohibited", "hospital ahead", "hump or rough road", "left hand curve", "left reverse bend",
    "left turn prohibited", "loose gravel", "men at work", "narrow bridge ahead", "narrow road ahead", "no entry",
    "no parking", "no stopping or standing", "overtaking prohibited", "pass either side", "pedestrian crossing",
    "petrol pump ahead", "quay side or river bank", "restriction ends", "right hand curve", "right reverse bend",
    "right turn prohibited", "road widens ahead", "roundabout", "school ahead", "side road left", "side road right",
    "slippery road", "speed limit 100", "speed limit 30", "speed limit 50", "staggered intersection", "steep ascent",
    "steep descent", "stop", "straight prohibited", "t intersection", "u turn", "u turn prohibited",
    "unguarded level crossing", "width limit", "y intersection"
]

# -------------------------------
# Load Model
# -------------------------------
def load_model(path, num_classes=58):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    state_dict = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict)
    model.eval()
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# Main Application
# -------------------------------
class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Sign Detection")
        self.setStyleSheet("background-color: #f0f0f0;")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model("final_frcnn_model.pth", num_classes=58)

        self.video_cap = None
        self.video_playing = False

        self.heading = QLabel("INDIAN TRAFFIC SIGN DETECTION AND RECOGNITION")
        self.heading.setFont(QFont("Arial", 20, QFont.Bold))
        self.heading.setAlignment(Qt.AlignCenter)
        self.heading.setStyleSheet("color: darkblue; margin: 20px;")

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid gray; margin: 10px;")

        self.predicted_label = QLabel("")
        self.predicted_label.setFont(QFont("Arial", 16))
        self.predicted_label.setAlignment(Qt.AlignCenter)

        # Buttons
        self.upload_button = QPushButton("UPLOAD IMAGE")
        self.upload_button.clicked.connect(self.upload_image)

        self.webcam_button = QPushButton("START WEBCAM")
        self.webcam_button.clicked.connect(self.start_webcam)

        self.stop_button = QPushButton("STOP WEBCAM")
        self.stop_button.clicked.connect(self.stop_webcam)

        self.video_button = QPushButton("UPLOAD VIDEO")
        self.video_button.clicked.connect(self.upload_video)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.upload_button)
        btn_layout.addWidget(self.webcam_button)
        btn_layout.addWidget(self.stop_button)
        btn_layout.addWidget(self.video_button)

        layout = QVBoxLayout()
        layout.addWidget(self.heading)
        layout.addWidget(self.image_label)
        layout.addWidget(self.predicted_label)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.last_fps_time = time.time()

        self.showMaximized()

    def stop_video(self):
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.video_playing = False
        self.image_label.clear()
        self.predicted_label.setText("")  # Clear label

    def upload_image(self):
        self.stop_webcam()
        self.stop_video()
        self.image_label.clear()

        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            image = Image.open(path).convert("RGB")
            draw, top_class = self.detect_image(image)
            qimage = QImage(draw.tobytes(), draw.width, draw.height, draw.width * 3, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qimage).scaled(1000, 800, Qt.KeepAspectRatio))
            self.predicted_label.setText(f'Predicted class: "{top_class}"' if top_class else "No class detected.")
            QApplication.processEvents()

    def start_webcam(self):
        self.stop_video()
        self.stop_webcam()
        self.image_label.clear()
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_webcam(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.image_label.clear()
        self.predicted_label.setText("")  # Clear label
        QApplication.processEvents()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_fps_time)
        self.last_fps_time = current_time

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw, top_class = self.detect_image(pil_img)

        draw1 = ImageDraw.Draw(draw)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        draw1.text((10, 10), f"FPS: {fps:.1f}", fill="yellow", font=font)

        qimage = QImage(draw.tobytes(), draw.width, draw.height, draw.width * 3, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimage).scaled(1000, 800, Qt.KeepAspectRatio))
        self.predicted_label.setText(f'Detected: "{top_class}"' if top_class else "No class detected.")
        QApplication.processEvents()

    def upload_video(self):
        self.stop_webcam()
        self.stop_video()
        self.image_label.clear()

        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.avi)")
        if not path:
            return

        self.video_cap = cv2.VideoCapture(path)
        self.video_playing = True
        width, height = int(self.video_cap.get(3)), int(self.video_cap.get(4))
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter("output_detected.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while self.video_playing and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            draw, _ = self.detect_image(pil_img)
            out.write(cv2.cvtColor(np.array(draw), cv2.COLOR_RGB2BGR))
            qimage = QImage(draw.tobytes(), draw.width, draw.height, draw.width * 3, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qimage).scaled(1000, 800, Qt.KeepAspectRatio))
            QApplication.processEvents()

        self.video_playing = False
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        out.release()
        self.predicted_label.setText("Video saved as output_detected.mp4")
        QApplication.processEvents()

    def detect_image(self, image):
        tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)[0]

        draw = image.copy()
        draw1 = ImageDraw.Draw(draw)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        top_score = 0
        top_class = None

        for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
            if score >= 0.5:
                x1, y1, x2, y2 = map(int, box.tolist())
                draw1.rectangle([x1, y1, x2, y2], outline="red", width=3)
                label_text = f"{CLASS_NAMES[label]} ({score:.2f})"
                draw1.text((x1, y1 - 25), label_text, fill="red", font=font)
                if score > top_score:
                    top_score = score
                    top_class = CLASS_NAMES[label]

        return draw, top_class

    def closeEvent(self, event):
        self.stop_webcam()
        self.stop_video()
        event.accept()

# -------------------------------
# Launch App
# -------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())
