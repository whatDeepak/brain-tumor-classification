import zipfile
import os

# Define paths
zip_folder = r'C:\_D\Uni\Minor Project\dataset'  # Change this to where you downloaded the .zip files
output_folder = r'C:\_D\Uni\Minor Project\brain-tumor-classification\dataset\original'

# Unzip each .zip file in the zip folder
for zip_file in os.listdir(zip_folder):
    if zip_file.endswith('.zip'):
        zip_path = os.path.join(zip_folder, zip_file)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
        print(f"Extracted: {zip_file}")
