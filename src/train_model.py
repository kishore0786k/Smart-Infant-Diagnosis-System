import joblib, numpy as np, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from collections import Counter

# ============================================================
# Load the balanced features dataset
# ============================================================
print("📂 Loading balanced dataset...")
X, y = joblib.load("data/features_balanced.pkl")

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
os.makedirs("models", exist_ok=True)
joblib.dump(le, "models/label_encoder.pkl")

print("\n📊 Class distribution:", Counter(y))

# Compute class weights (for safety, even if already balanced)
classes = np.unique(y_enc)
class_weights_vals = compute_class_weight(class_weight='balanced', classes=classes, y=y_enc)
class_weight = dict(zip(classes, class_weights_vals))
print("\⚖️ Computed Class Weights:", class_weight)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ============================================================
# Build Neural Network Model
# ============================================================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(np.unique(y_enc)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ============================================================
# Train the Model
# ============================================================
print("\n🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=16,
    verbose=1,
    class_weight=class_weight
)

# ============================================================
# Evaluate the Model
# ============================================================
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

model.save("models/infant_cry_model.h5")
print("\n✅ Model trained and saved successfully as 'models/infant_cry_model.h5'")
