import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D # type: ignore
import seaborn as sns

# Print progress
def print_progress(current, total):
    percent = int((current / total) * 100)
    print(f"\rProgress: {current}/{total} ({percent}%) complete", end='')

# Set path and categories
base_path = "C:/Users/DELL/OneDrive/AML/chest_xray"
splits = ['train', 'test', 'val']
categories = ['NORMAL', 'PNEUMONIA']  # Update if you have more categories

image_paths = []
labels = []

# Load image paths and labels
print("Loading dataset...")
for split in splits:
    for label, category in enumerate(categories):
        category_path = os.path.join(base_path, split, category)
        if not os.path.exists(category_path):
            continue
        for img_name in os.listdir(category_path):
            if img_name.lower().endswith((".jpeg", ".jpg", ".png")):
                image_paths.append(os.path.join(category_path, img_name))
                labels.append(label)

# Load VGG16 model for feature extraction
print("\nLoading VGG16 model...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
model = Model(inputs=base_model.input, outputs=x)

# Extract features with error handling
print("Extracting features...")
features = []
valid_labels = []
skipped = 0
total_images = len(image_paths)

for i, (img_path, label) in enumerate(zip(image_paths, labels)):
    print_progress(i + 1, total_images)
    try:
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        feature = model.predict(img_array, verbose=0)
        features.append(feature.flatten())
        valid_labels.append(label)
    except Exception as e:
        skipped += 1
        print(f"\n[Skipping] {img_path} — Error: {e}")

features = np.array(features)

# Apply PCA
print("\nApplying PCA...")
pca = PCA(n_components=50)
reduced_features = pca.fit_transform(features)

# Apply KMeans clustering
print("Performing KMeans clustering...")
n_clusters = len(categories)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(reduced_features)

print("Visualizing clusters...")
plt.figure(figsize=(10, 5))
for i in range(n_clusters):
    cluster_indices = np.where(cluster_labels == i)[0]
    if len(cluster_indices) == 0:
        continue
    img_path = image_paths[cluster_indices[0]]
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, n_clusters, i + 1)
        plt.imshow(img)
        plt.title(f"Cluster {i + 1}")
        plt.axis('off')
    except:
        continue
plt.tight_layout()
plt.show()

# Create confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(valid_labels, cluster_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f"Cluster {i+1}" for i in range(n_clusters)],
            yticklabels=categories)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Clusters")
plt.ylabel("True Labels")
plt.show()

print("Calculating clustering accuracy...")
correct_predictions = 0
for cluster in range(n_clusters):
    cluster_indices = np.where(cluster_labels == cluster)[0]
    if len(cluster_indices) == 0:
        continue
    cluster_true_labels = [valid_labels[i] for i in cluster_indices]
    majority_label = max(set(cluster_true_labels), key=cluster_true_labels.count)
    correct_predictions += cluster_true_labels.count(majority_label)

accuracy = correct_predictions / len(valid_labels)
print(f"\n Clustering accuracy: {accuracy:.2%}")
print(f" Skipped {skipped} image(s) due to load errors.")
