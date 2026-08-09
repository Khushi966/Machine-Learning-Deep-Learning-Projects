import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

# ---------------------------
# 1) LOAD PRETRAINED MODELS
# ---------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# GoogLeNet
googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
googlenet.fc = nn.Linear(1024, 2)
googlenet = googlenet.to(device)
googlenet.eval()

# ResNet50
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.fc = nn.Linear(2048, 2)
resnet = resnet.to(device)
resnet.eval()

# Load your trained weights here:
# googlenet.load_state_dict(torch.load("gn_helmet.pth", map_location=device))
# resnet.load_state_dict(torch.load("resnet_helmet.pth", map_location=device))


# ---------------------------
# 2) PREPROCESSING FUNCTION
# ---------------------------

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def classify_patch(model, patch):
    img_tensor = preprocess(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor)
        prob = torch.softmax(out, dim=1)[0]
        label = torch.argmax(prob).item()
    return label, prob[label].item()


# ---------------------------
# 3) HELMET DETECTOR
# ---------------------------

def detect_helmets(img, model, window_size=150, stride=50, threshold=0.50):
    h, w = img.shape[:2]
    boxes = []

    for y in range(0, h - window_size, stride):
        for x in range(0, w - window_size, stride):

            patch = img[y:y + window_size, x:x + window_size]
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

            label, prob = classify_patch(model, patch_rgb)

            if label == 1 and prob >= threshold:  # 1 = Helmet class
                boxes.append((x, y, x + window_size, y + window_size, prob))

    return boxes


# ---------------------------
# 4) DRAW RESULTS
# ---------------------------

def draw_boxes(img, boxes, color=(0, 255, 0)):
    for (x1, y1, x2, y2, prob) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"Helmet {prob:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


# ---------------------------
# 5) RUN ON IMAGE
# ---------------------------

image_path = r"C:\Users\DELL\OneDrive\DL\Helmet Detection\images\BikesHelmets173.png"
print("Loading image:", image_path)

img = cv2.imread(image_path)

if img is None:
    print("ERROR: Image not found. Check path.")
    exit()

print("Image loaded:", img.shape)

# Convert image to RGB for model input
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# GoogLeNet detection
print("Running GoogLeNet detection...")
gn_boxes = detect_helmets(img, googlenet)
print("GoogLeNet boxes:", gn_boxes)
img_gn = draw_boxes(img.copy(), gn_boxes, color=(0, 255, 0))

# ResNet detection
print("Running ResNet50 detection...")
res_boxes = detect_helmets(img, resnet)
print("ResNet boxes:", res_boxes)
img_res = draw_boxes(img.copy(), res_boxes, color=(255, 0, 0))

# ---------------------------
# 6) SHOW OR SAVE OUTPUT
# ---------------------------

try:
    cv2.imshow("GoogLeNet Helmet Detection", img_gn)
    cv2.imshow("ResNet50 Helmet Detection", img_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    print("GUI not available. Saving results instead...")
    cv2.imwrite("output_gn.jpg", img_gn)
    cv2.imwrite("output_resnet.jpg", img_res)
    print("Saved output_gn.jpg and output_resnet.jpg")

