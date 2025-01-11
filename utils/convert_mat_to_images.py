import os
import scipy.io
import h5py
import numpy as np
import cv2

# Paths
input_folder = r'C:\_D\Uni\Minor Project\brain-tumor-classification\dataset\original'
output_folder = r'C:\_D\Uni\Minor Project\brain-tumor-classification\dataset\images'

def load_mat_file(mat_path):
    try:
        # Try to read as regular .mat file
        data = scipy.io.loadmat(mat_path)
        return data, False
    except NotImplementedError:
        # If the file is in v7.3 format, use h5py
        try:
            with h5py.File(mat_path, 'r') as f:
                # For v7.3 files, we need to handle the data differently
                cjdata = f.get('cjdata')
                if cjdata is None:
                    print(f"Error: 'cjdata' not found in {mat_path}")
                    return None, True
                
                data = {
                    'cjdata': {
                        'image': np.array(cjdata.get('image')),
                        'label': np.array(cjdata.get('label'))
                    }
                }
                return data, True
        except Exception as e:
            print(f"Error opening {mat_path} with h5py: {e}")
            return None, True

def extract_image_and_label(data, is_v73):
    try:
        if is_v73:
            # For v7.3 files
            image = data['cjdata']['image']
            label = int(data['cjdata']['label'][0])
        else:
            # For regular .mat files
            cjdata = data['cjdata'][0, 0]
            image = cjdata['image']
            label = int(cjdata['label'][0, 0])
        
        return image, label
    except Exception as e:
        print(f"Error extracting data: {e}")
        return None, None

def process_image(image):
    # Handle different image formats and normalize
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]
    
    # Ensure image is 2D
    if image.ndim != 2:
        print(f"Unexpected image dimensions: {image.shape}")
        return None
    
    # Normalize to 0-255 range
    image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
    return image

# Process each .mat file
for mat_file in os.listdir(input_folder):
    if mat_file.endswith('.mat'):
        mat_path = os.path.join(input_folder, mat_file)
        print(f"Processing {mat_file}...")
        
        try:
            # Load the data
            data, is_v73 = load_mat_file(mat_path)
            if data is None:
                print(f"Skipping {mat_file} due to read error.")
                continue
            
            # Extract image and label
            image, label = extract_image_and_label(data, is_v73)
            if image is None or label is None:
                print(f"Skipping {mat_file} due to data extraction error.")
                continue
            
            # Process the image
            processed_image = process_image(image)
            if processed_image is None:
                print(f"Skipping {mat_file} due to image processing error.")
                continue
            
            # Create label folder if it doesn't exist
            label_folder = os.path.join(output_folder, str(label))
            os.makedirs(label_folder, exist_ok=True)
            
            # Save image as JPG
            output_path = os.path.join(label_folder, mat_file.replace('.mat', '.jpg'))
            cv2.imwrite(output_path, processed_image)
            print(f"Successfully saved: {output_path}")
        
        except KeyboardInterrupt:
            print("\nProcess interrupted by user.")
            break
        except Exception as e:
            print(f"Error processing {mat_file}: {e}")
            continue

print("Processing complete!")