import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from tqdm import tqdm


def load_trained_model(path, num_classes=58, device='cpu'):
    """
    Load your trained Faster R-CNN model from .pth checkpoint
    """
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, data_loader, iou_threshold=0.5, device='cpu'):
    """
    Evaluate object detection model on a given DataLoader
    """
    model.eval()
    model.to(device)

    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for pred, target in zip(outputs, targets):
                gt_boxes = target["boxes"]
                gt_labels = target["labels"]

                pred_boxes = pred["boxes"]
                pred_scores = pred["scores"]
                pred_labels = pred["labels"]

                # Apply score threshold
                keep = pred_scores >= 0.5
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                if len(pred_boxes) == 0:
                    total_fn += len(gt_boxes)
                    continue

                ious = box_iou(pred_boxes, gt_boxes)
                matched_gt = set()
                tp, fp = 0, 0

                for i in range(len(pred_boxes)):
                    iou_row = ious[i]
                    max_iou, max_idx = iou_row.max(0)
                    max_idx = max_idx.item()

                    if max_iou >= iou_threshold and max_idx not in matched_gt:
                        if pred_labels[i] == gt_labels[max_idx]:  # class match
                            tp += 1
                            matched_gt.add(max_idx)
                        else:
                            fp += 1  # IoU match but class mismatch
                    else:
                        fp += 1

                fn = len(gt_boxes) - len(matched_gt)
                total_tp += tp
                total_fp += fp
                total_fn += fn

    # Final metrics
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)

    print("\n📊 Evaluation Results:")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1 Score:  {f1_score:.4f}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score
    }
from evaluate_model import load_trained_model, evaluate_model
from your_dataset_loader import val_loader  # Replace this with your actual DataLoader import

model = load_trained_model("final_frcnn_model.pth", num_classes=58, device='cpu')
metrics = evaluate_model(model, val_loader, device='cpu')

