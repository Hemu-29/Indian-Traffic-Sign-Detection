
CHECKPOINT_DIR = "checkpoints"
FINAL_MODEL_PATH = "Trained_Models/final_frcnn_model.pth"

import os
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms import functional as F
import torchvision.transforms as T
from tqdm import tqdm
import os
DATASET_ROOT = "traffic_dataset"
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "train")
VAL_IMG_DIR = os.path.join(DATASET_ROOT, "valid")
TRAIN_ANN_FILE = os.path.join(TRAIN_IMG_DIR, "_annotations.coco.json")
VAL_ANN_FILE = os.path.join(VAL_IMG_DIR, "_annotations.coco.json")


class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class CocoDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super(CocoDataset, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        anns = self.coco.loadAnns(self.coco.getAnnIds(image_id))

        boxes, labels = [], []
        for obj in anns:
            xmin, ymin, width, height = obj["bbox"]
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(obj["category_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([image_id])

        target = {"boxes": boxes, "labels": labels, "image_id": image_id}
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target



transform = ComposeTransform([ToTensor()])

train_dataset = CocoDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, transforms=transform)
val_dataset = CocoDataset(VAL_IMG_DIR, VAL_ANN_FILE, transforms=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(train_dataset.coco.getCatIds()) + 1  # +1 for background

model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
if torch.cuda.is_available():
    print("🚀 GPU is available and will be used.")
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("⚠️ GPU not available, using CPU.")


start_epoch = 0

if os.path.exists(CHECKPOINT_DIR):
    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
    if ckpts:
        ckpts.sort()
        latest_ckpt = os.path.join(CHECKPOINT_DIR, ckpts[-1])
        checkpoint = torch.load(latest_ckpt)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resuming training from epoch {start_epoch}")
    else:
        print("⚠️ No checkpoint found, starting from scratch.")
else:
    print("⚠️ Checkpoint directory not found, starting from scratch.")


NUM_EPOCHS = 20

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for images, targets in pbar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        pbar.set_postfix(loss=losses.item())

    avg_loss = running_loss / len(train_loader)
    print(f"📉 Epoch {epoch+1} average loss: {avg_loss:.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, ckpt_path)
    print(f"💾 Saved checkpoint: {ckpt_path}")
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f"🎉 Final model saved to: {FINAL_MODEL_PATH}")
