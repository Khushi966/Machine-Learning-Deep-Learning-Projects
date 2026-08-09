import cv2
import time
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from statistics import mean

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# ----------------------------------------------------
# 1) LOAD MODELS
# ----------------------------------------------------

# GoogLeNet (classification)
googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
googlenet.fc = nn.Linear(1024, 2)
googlenet = googlenet.to(device).eval()

# ResNet-50 (classification)
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.fc = nn.Linear(2048, 2)
resnet = resnet.to(device).eval()

# Fast R-CNN (using FasterRCNN backbone but same FC head logic)
fastrcnn = fasterrcnn_resnet50_fpn(weights="DEFAULT")
fastrcnn = fastrcnn.to(device).eval()

# --------------------------------------------
# 2) IMAGE PREPROCESSING
# --------------------------------------------

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def classify_patch(model, patch):
    """Classification for GoogLeNet / ResNet using sliding window."""
    img_tensor = preprocess(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor)
        prob = torch.softmax(out, dim=1)[0]
        label = torch.argmax(prob).item()
        return label, float(prob[label])

# --------------------------------------------
# 3) SLIDING WINDOW DETECTOR FOR GoogLeNet & ResNet
# --------------------------------------------

def detect_sliding_window(img, model, window=150, stride=50, threshold=0.85):
    h, w = img.shape[:2]
    detections = []

    for y in range(0, h - window, stride):
        for x in range(0, w - window, stride):

            patch = img[y:y+window, x:x+window]
            label, prob = classify_patch(model, patch)

            if label == 1 and prob >= threshold:
                detections.append([x, y, x+window, y+window, prob])

    return detections

# --------------------------------------------
# 4) EVALUATION METRICS (mAP, F1, Accuracy)
# --------------------------------------------

def compute_f1(true_labels, pred_labels):
    return f1_score(true_labels, pred_labels, average='binary')

def compute_accuracy(true_labels, pred_labels):
    return accuracy_score(true_labels, pred_labels)

# Dummy mAP calculation (replace with your annotation data)
def compute_map(pred_count, gt_count):
    return pred_count / (gt_count + 1e-6)

# --------------------------------------------
# 5) RUN COMPARISON ON ONE IMAGE OR DATASET
# --------------------------------------------

image_path = "C:/Users/DELL/OneDrive/DL/Helmet Detection/images/BikesHelmets173.png"
img = cv2.imread(image_path)

# ---------------- GoogleNet ----------------
start_gn = time.time()
gn_boxes = detect_sliding_window(img, googlenet)
time_gn = time.time() - start_gn
fps_gn = 1 / time_gn

# ---------------- ResNet-50 ----------------
start_res = time.time()
res_boxes = detect_sliding_window(img, resnet)
time_res = time.time() - start_res
fps_res = 1 / time_res

# ---------------- Fast R-CNN ----------------
start_fr = time.time()
img_tensor = [torch.tensor(img).permute(2, 0, 1).float().to(device)]
outputs = fastrcnn(img_tensor)[0]
time_fr = time.time() - start_fr
fps_fr = 1 / time_fr

fr_boxes = []
for box, score in zip(outputs["boxes"], outputs["scores"]):
    if score > 0.8:
        b = box.detach().cpu().numpy()
        fr_boxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(score)])

# ----------------------------------------------
# 6) PRINT RESULTS
# ----------------------------------------------

print("\n=== COMPARISON RESULTS ===")
print(f"GoogLeNet FPS: {fps_gn:.2f}")
print(f"ResNet-50 FPS: {fps_res:.2f}")
print(f"Fast R-CNN FPS: {fps_fr:.2f}")

# Dummy true labels for example (replace with your dataset values)
true_labels = [1]  # helmet present

pred_gn = [1 if len(gn_boxes) > 0 else 0]
pred_res = [1 if len(res_boxes) > 0 else 0]
pred_fr = [1 if len(fr_boxes) > 0 else 0]

print(f"\nAccuracy:")
print("GoogLeNet:", compute_accuracy(true_labels, pred_gn))
print("ResNet-50:", compute_accuracy(true_labels, pred_res))
print("Fast R-CNN:", compute_accuracy(true_labels, pred_fr))

print(f"\nF1 Score:")
print("GoogLeNet:", compute_f1(true_labels, pred_gn))
print("ResNet-50:", compute_f1(true_labels, pred_res))
print("Fast R-CNN:", compute_f1(true_labels, pred_fr))

print("\nmAP (dummy values):")
print("GoogLeNet:", compute_map(len(gn_boxes), 3))
print("ResNet-50:", compute_map(len(res_boxes), 3))
print("Fast R-CNN:", compute_map(len(fr_boxes), 3))

# ----------------------------------------------
# 7) VISUALIZE RESULTS
# ----------------------------------------------

def draw_boxes(img, boxes, color):
    for (x1, y1, x2, y2, prob) in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, f"{prob:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

cv2.imshow("GoogLeNet", draw_boxes(img.copy(), gn_boxes, (0,255,0)))
cv2.imshow("ResNet-50", draw_boxes(img.copy(), res_boxes, (255,0,0)))
cv2.imshow("Fast R-CNN", draw_boxes(img.copy(), fr_boxes, (0,0,255)))

cv2.waitKey(0)
cv2.destroyAllWindows()
