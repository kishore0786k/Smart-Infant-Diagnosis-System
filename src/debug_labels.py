import joblib
from collections import Counter
X, y = joblib.load("data/features.pkl")
print("Total samples:", len(y))
print("Class counts:", Counter(y))

# If you used label encoder before:
import joblib as jb
le = jb.load("models/label_encoder.pkl")
print("Label classes (encoder):", list(le.classes_))
