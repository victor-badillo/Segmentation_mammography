import os
import cv2
import numpy as np
import pandas as pd

def calculate_interregion_contrast(image, mask):
    """
    Calculate interregion contrast (GLCM) between the segmented region and the background.
    
    Args:
        image (np.ndarray): Original grayscale image.
        mask (np.ndarray): Binary segmentation mask (1 for segmented region, 0 for background).
    
    Returns:
        float: Interregion contrast (GLCM).
    """
    # Ensure the mask is binary
    mask = mask > 0

    # Extract pixels from the segmented region and the background
    region_segmented = image[mask]
    region_background = image[~mask]

    # Calculate mean intensities
    mean_segmented = np.mean(region_segmented) if region_segmented.size > 0 else 0
    mean_background = np.mean(region_background) if region_background.size > 0 else 0

    # Calculate interregion contrast (GLCM)
    numerator = abs(mean_segmented - mean_background)
    denominator = mean_segmented + mean_background

    contrast = numerator / denominator if denominator > 0 else 0
    return contrast


def validate_segmentations(data_dir, segmentation_dir):
    """
    Validate segmentations by calculating interregion contrast for each image.
    
    Args:
        data_dir (str): Directory containing subdirectories with original images.
        segmentation_dir (str): Directory containing the segmented images.
    
    Returns:
        pd.DataFrame: DataFrame with image ID, tissue type, and interregion contrast.
    """
    results = []

    #Iterate through tissue type subdirectories
    for tissue_type in os.listdir(data_dir):
        tissue_path = os.path.join(data_dir, tissue_type)
        if not os.path.isdir(tissue_path):
            continue

        #Iterate through images in the tissue subdirectory
        for filename in os.listdir(tissue_path):
            if filename.endswith('.jpg'):
                #Get paths for the original and segmented images
                image_id = filename.split('.')[0]
                original_path = os.path.join(tissue_path, filename)
                segmented_path = os.path.join(segmentation_dir, f"{image_id}_msk.jpg")

                #Check if the corresponding segmented image exists
                if not os.path.exists(segmented_path):
                    print(f"Warning: Segmentation not found for {filename}")
                    continue

                #Load the original and segmented images
                original_image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
                segmented_mask = cv2.imread(segmented_path, cv2.IMREAD_GRAYSCALE)


                #Calculate interregion contrast
                contrast = calculate_interregion_contrast(original_image, segmented_mask)

                #Append results
                results.append({
                    'Image_ID': image_id,
                    'Tissue_Type': tissue_type,
                    'Interregion_Contrast': contrast
                })

    #Create a DataFrame with the results
    results_df = pd.DataFrame(results)
    return results_df


def print_results_table(results_df):
    """
    Print the results of segmentation validation in a formatted table.
    
    Args:
        results_df (pd.DataFrame): DataFrame with validation results.
    """
    print(results_df.to_string(index=False))