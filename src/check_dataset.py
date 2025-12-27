import os

base_dir = "data/Baby Cry Sence Dataset"
for folder in os.listdir(base_dir):
    path = os.path.join(base_dir, folder)
    if os.path.isdir(path):
        count = len([f for f in os.listdir(path) if f.endswith(".wav")])
        print(f"{folder}: {count} files")
