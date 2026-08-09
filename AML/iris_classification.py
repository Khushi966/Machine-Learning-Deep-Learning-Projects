import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Load data
file_path = "Iris.csv"
iris_data = pd.read_csv(file_path)

# Display data info
print(iris_data.head())
print(iris_data.info())

X = iris_data.drop(columns=['Id', 'Species'], errors='ignore')
Y = iris_data['Species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(6, 4))
sns.barplot(x=feature_importance, y=features, color="skyblue")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Greens", values_format='d')
plt.title("Confusion Matrix")
plt.show()
