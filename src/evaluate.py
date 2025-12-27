import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# === Load Data and Model ===
print("📂 Loading balanced dataset and model...")
X, y = joblib.load("data/features_balanced.pkl")  # 13 MFCCs only
le = joblib.load("models/label_encoder.pkl")
model = load_model("models/infant_cry_model.h5")

print(f"✅ Loaded X shape: {X.shape}, Model expects: {model.input_shape[1]} features")

# === Predict ===
print("🔍 Running predictions...")
y_true = le.transform(y)
y_pred = np.argmax(model.predict(X), axis=1)

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Balanced Infant Cry Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# === Detailed Metrics ===
print("\n📈 Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

acc = np.mean(y_true == y_pred)
print(f"\n✅ Overall Accuracy: {acc * 100:.2f}%")
