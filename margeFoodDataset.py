import os
import shutil
from glob import glob

# Set your base path
source_root = r"C:\Users\s5323655\Downloads\Food"
destination_root = os.path.join(source_root, "combined")

# Source folders
source_folders = ["training", "validation", "evaluation"]

# Make sure the combined folder exists
os.makedirs(destination_root, exist_ok=True)

for folder in source_folders:
    folder_path = os.path.join(source_root, folder)
    for class_name in os.listdir(folder_path):
        class_src_path = os.path.join(folder_path, class_name)
        class_dst_path = os.path.join(destination_root, class_name)
        os.makedirs(class_dst_path, exist_ok=True)

        # Copy all files from source to destination
        for file_path in glob(os.path.join(class_src_path, "*")):
            filename = os.path.basename(file_path)
            dst_file_path = os.path.join(class_dst_path, filename)

            # Handle duplicate filenames by appending folder name
            if os.path.exists(dst_file_path):
                base, ext = os.path.splitext(filename)
                filename = f"{base}_{folder}{ext}"
                dst_file_path = os.path.join(class_dst_path, filename)

            shutil.copy(file_path, dst_file_path)

print("✅ All images combined into: ", destination_root)
