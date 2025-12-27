import joblib, numpy as np

# Load the existing MFCC features
X, y = joblib.load("data/features.pkl")

# Simulate biological data for each sample
# These are just random, realistic values for infants
hr = np.random.normal(140, 10, len(X))       # heart rate
temp = np.random.normal(37.0, 0.3, len(X))   # temperature

# Stack the biological features alongside the audio features
bio = np.vstack((hr, temp)).T
X_bio = np.hstack((X, bio))

# Save the new combined dataset
joblib.dump((X_bio, y), "data/features_bio.pkl")

print(f"✅ Simulated biological data added successfully.")
print(f"New feature shape: {X_bio.shape}")
