import os
import scipy.io
import numpy as np
import cv2
import zipfile
import tqdm
import h5py

"""
Downloaded from 
https://figshare.com/articles/dataset/brain_tumor_dataset/1512427?file=51340418
"""

def extract(zip_file_path, extract_folder):
        # Ensure the extraction folder exists
    if os.path.exists(extract_folder):
        return
    os.makedirs(extract_folder, exist_ok=True)

    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

def process(input_folder, output_folder,name):
    # Ensure output folders exist
    image_output_folder = os.path.join(output_folder, "images")
    mask_output_folder = os.path.join(output_folder, "masks")
    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(mask_output_folder, exist_ok=True)

    # List all .mat files in the input folder
    file_list = [f for f in os.listdir(input_folder) if f.endswith(".mat")]

    for file_name in tqdm.tqdm(file_list):
        file_path = os.path.join(input_folder, file_name)

        # Load .mat file
        mat_data = h5py.File(file_path, "r")
        cjdata = mat_data["cjdata"]

        # Extract image data
        image = np.array(cjdata["image"], dtype=np.float32)
        
        # Normalize and convert image to uint8
        image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)

        # Extract tumor border coordinates
        tumor_border = cjdata["tumorBorder"][0]

        # Convert tumor border to a list of (x, y) points
        tumor_border = np.array(tumor_border, dtype=np.int32).reshape(-1, 2)

        # Create an empty mask with the same shape as the image
        mask = np.zeros_like(image, dtype=np.uint8)

        # Draw the tumor region as a filled polygon on the mask
        cv2.fillPoly(mask, [tumor_border], 255)

        # Get the label 

        label_names = {1: "meningioma", 2: "glioma", 3: "pituitary tumor"}
        label = label_names[int(np.array(cjdata["label"]))]

        # Create label-based subfolders
        label_image_folder = os.path.join(image_output_folder, str(label))
        label_mask_folder = os.path.join(mask_output_folder, str(label))
        os.makedirs(label_image_folder, exist_ok=True)
        os.makedirs(label_mask_folder, exist_ok=True)

        # Save image and mask
        image_filename = os.path.join(label_image_folder, f"{name}_{os.path.splitext(file_name)[0]}.png")
        mask_filename = os.path.join(label_mask_folder, f"{name}_{os.path.splitext(file_name)[0]}.png")

        cv2.imwrite(image_filename, image)
        cv2.imwrite(mask_filename, mask)

    print("Image and mask extraction completed successfully!")

extract_folder = "mat_data"
output_data_path = 'segmentation_data'

for file in tqdm.tqdm(os.listdir()):
    if file.endswith('.zip'):
        file_output = os.path.join(extract_folder, os.path.basename(file.replace('.zip','')) )
        print('Extracting to',file_output)
        extract(file,file_output)

        process(file_output,output_data_path,
                os.path.basename(file.replace('.zip','')))
