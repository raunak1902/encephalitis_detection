import tarfile
import os

tar_path = r"C:\Users\rauna\Downloads\encephalitis_detection\training\NINS_Dataset.tar"
extract_path = r"C:\Users\rauna\Downloads\encephalitis_detection\training\NINS_Dataset"

print("Checking if file exists:", os.path.exists(tar_path))

os.makedirs(extract_path, exist_ok=True)

with tarfile.open(tar_path, "r:") as tar:
    tar.extractall(path=extract_path)

print("✅ Dataset extracted successfully!")
