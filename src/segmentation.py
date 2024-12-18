import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label, find_objects
from utilities import save_image, visualize_image



def keep_largest_object(binary_image):
    """
    Retains the largest connected component in a binary image.

    Parameters:
        binary_image (numpy.ndarray): Input binary image where objects have values > 0.

    Returns:
        numpy.ndarray: Binary mask containing only the largest object, 
                       with values 0 (background) and 255 (largest object).
    """

    #Label components
    labeled_image, num_features = label(binary_image)
    
    # Area of each component
    object_slices = find_objects(labeled_image)
    areas = [np.sum(labeled_image[obj_slice] == label_id + 1) for label_id, obj_slice in enumerate(object_slices)]
    
    #Identify index of biggest component
    largest_component_idx = np.argmax(areas) + 1  #Labeled objects index start at 1

    #Mask with biggest object
    largest_object_mask = (labeled_image == largest_component_idx).astype(np.uint8)
    
    return largest_object_mask * 255


def remove_labels(image):
    """
    Removes labels (unwanted small regions or artifacts) from an image, keeping only the largest connected component. 
    Smooths the resulting region to produce cleaner borders.

    Parameters:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Processed grayscale image with labels removed and borders smoothed.
    """

    #Binarize image
    #_, binarized = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binarized = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)

    #Separate regions that could be joined
    kernel_separate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binarized = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel_separate)

    #Keep biggest binary object from image
    without_labels_binary = keep_largest_object(binarized)

    #Image without labels
    final =  cv2.bitwise_and(image, without_labels_binary)

    #Smooth borders
    radius = 10 
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    without_labels_binary_smooth = cv2.morphologyEx(without_labels_binary, cv2.MORPH_OPEN, kernel_opening)

    #Image without labels smooth
    final_smooth =  cv2.bitwise_and(image, without_labels_binary_smooth)

    # plt.figure(figsize=(20, 20))
    # plt.subplot(2, 3, 1), plt.title("Original"), plt.imshow(image, cmap='gray')
    # plt.subplot(2, 3, 2), plt.title("Binarized"), plt.imshow(binarized, cmap='gray')
    # plt.subplot(2, 3, 3), plt.title("Without Labels Binary"), plt.imshow(without_labels_binary, cmap='gray')
    # plt.subplot(2, 3, 4), plt.title("Final"), plt.imshow(final, cmap='gray')
    # plt.subplot(2, 3, 5), plt.title("Binary Smooth"), plt.imshow(without_labels_binary_smooth, cmap='gray')
    # plt.subplot(2, 3, 6), plt.title("Final Smooth"), plt.imshow(final_smooth, cmap='gray')
    # plt.show()

    return final_smooth



def adjust_image_orientation(image, original):
    """
    Adjusts the orientation of the image based on the position of the first white pixel starting from the top-left corner.
    If the white pixel is found in the second half of the columns, the image is flipped horizontally.

    Args:
        image (np.ndarray): Binary grayscale image.
        original (np.ndarray): Original image to adjust.

    Returns:
        np.ndarray: Adjusted image.
        bool: A flag indicating whether the image was mirrored (flipped horizontally).
    """
    rows, cols = image.shape
    mirrored = False

    #Traverse the image from top to bottom, left to right
    for row in range(rows):
        for col in range(cols):
            if image[row, col] == 255:  # Find the first white pixel
                print(f"First white pixel found at ({row}, {col})")
                #If the pixel is in the second half of the columns, flip the image
                if col >= cols // 2:
                    print("The pixel is in the second half, flipping the image.")
                    mirrored = True
                    return cv2.flip(original, 1), mirrored  #Flip horizontally
                else:
                    print("The pixel is in the first half, keeping the image as it is.")
                    return original, mirrored

    print("No white pixel was found, returning the image unchanged.")
    return original, mirrored


def breast_orientate(without_labels):
    """
    Orients a mammogram image by detecting the main breast region and adjusting its position.

    Args:
        without_labels (np.ndarray): Grayscale mammogram image without annotations.

    Returns:
        tuple:
            - breast_oriented (np.ndarray): Oriented breast image.
            - mirrored (bool): True if the image was flipped horizontally, False otherwise.
    """

    #Binarize and find contours
    _, binary_image = cv2.threshold(without_labels, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("No contours were found.")
    else:
        #Choose biggest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        #Draw contour
        contoured_image = np.zeros_like(without_labels)
        cv2.drawContours(contoured_image, [largest_contour], -1, 255, 2)


    #Keep vertical line next to muscle in mammography
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 400))
    vertical_lines = cv2.erode(contoured_image, kernel_vertical)

    #Obtain breast oriented and a flag indicating if the image was mirrored
    breast_oriented, mirrored = adjust_image_orientation(vertical_lines, without_labels)

    # plt.figure(figsize=(20, 20))
    # plt.subplot(2, 4, 1), plt.title("Without labels"), plt.imshow(without_labels, cmap='gray')
    # plt.subplot(2, 4, 2), plt.title("Binarized"), plt.imshow(binary_image, cmap='gray')
    # plt.subplot(2, 4, 3), plt.title("Contours"), plt.imshow(contoured_image, cmap='gray')
    # plt.subplot(2, 4, 4), plt.title("Vertical line"), plt.imshow(vertical_lines, cmap='gray')
    # plt.subplot(2, 4, 5), plt.title("Oriented breast"), plt.imshow(breast_oriented, cmap='gray')
    # plt.show()

    return breast_oriented, mirrored



def remove_empty_columns(image):
    """
    Removes empty (completely black) columns on the left side of the image.

    Args:
        image (np.ndarray): Binary image.

    Returns:
        tuple: (np.ndarray, int) Cropped image and the number of columns removed.
    """
    # Find columns that are not completely black
    col_sums = np.sum(image, axis=0)
    first_nonzero_col = np.argmax(col_sums > 0)  # First non-empty column

    # Crop the image from the first non-empty column
    cropped_image = image[:, first_nonzero_col:]

    return cropped_image, first_nonzero_col


def region_growing(image, seed_point, threshold=10):
    """
    Performs region growing segmentation starting from a seed point.
    
    Args:
        image (np.ndarray): Grayscale image.
        seed_point (tuple): Initial point (row, column).
        threshold (int): Maximum intensity difference to consider homogeneity.
    
    Returns:
        np.ndarray: Binary mask with the segmented region.
    """
    rows, cols = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=bool)
    seed_value = image[seed_point]
    
    #Use a stack for the iterative algorithm
    stack = [seed_point]
    
    while stack:
        x, y = stack.pop()
        
        if visited[x, y]:
            continue
        
        #Mark as visited
        visited[x, y] = True
        
        #Add to the region if it meets the criteria
        if abs(int(image[x, y]) - int(seed_value)) <= threshold:
            segmented[x, y] = 255
            
            #Add valid neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                    stack.append((nx, ny))
    
    return segmented



def restore_columns(image, mask, removed_columns):
    """
    Restores the removed columns on the left side of the mask to return it to the original image dimensions.

    Args:
        image (np.ndarray): Original image without labels.
        mask (np.ndarray): Mask generated by region growing.
        columns_removed (int): Number of columns initially removed.

    Returns:
        np.ndarray: Mask with restored original dimensions.
    """

    restored_mask = np.zeros_like(image, dtype=np.uint8)

    #Insert the generated mask into the corresponding region
    restored_mask[:, removed_columns:] = mask

    return restored_mask




def remove_muscle(without_labels, breast_orientated):
    """
    Removes the muscle region from a mammogram image using filtering, region growing, and masking techniques.

    Args:
        without_labels (np.ndarray): Original grayscale mammogram image without annotations.
        breast_orientated (np.ndarray): Oriented breast image for processing.

    Returns:
        np.ndarray: Image with the muscle region removed.
    """

    #Obtain cropped image without columns in the left part
    cropped_breast, columns_removed = remove_empty_columns(breast_orientated)

    #Apply bilateral filter to image to obtain a more homogeneous muscle
    breast_filtered = cv2.bilateralFilter(cropped_breast, d=50, sigmaColor=40, sigmaSpace=10)

    #Initialization point for region growing
    seed_point = (30, 30) 
    muscle_mask = region_growing(breast_filtered, seed_point, threshold=35)

    #Image with muscle mask with original dimensions
    muscle_mask_restored = restore_columns(without_labels, muscle_mask, columns_removed)

    #FALTA HACER UN TOP HAT, RELLENAR HUECOS Y APLANAR PICOS, (a lo mejor no es en esta parte y es en el siguiente)

    #Breast mask
    _, binary_orientated = cv2.threshold(breast_orientated, 1, 255, cv2.THRESH_BINARY)
    breast_mask = np.clip(binary_orientated - muscle_mask_restored, 0, 255).astype(np.uint8)

    #Obtain image containing only the breast
    breast = cv2.bitwise_and(breast_orientated, breast_mask)

    # plt.figure(figsize=(20, 20))
    # plt.subplot(2, 4, 1), plt.title("Without labels"), plt.imshow(without_labels, cmap='gray')
    # plt.subplot(2, 4, 2), plt.title("Cropedd"), plt.imshow(cropped_breast, cmap='gray')
    # plt.subplot(2, 4, 3), plt.title("Bilateral filter"), plt.imshow(breast_filtered, cmap='gray')
    # plt.subplot(2, 4, 4), plt.title("Muscle mask"), plt.imshow(muscle_mask, cmap='gray')
    # plt.subplot(2, 4, 5), plt.title("Muscle mask restored"), plt.imshow(muscle_mask_restored, cmap='gray')
    # plt.subplot(2, 4, 6), plt.title("Binarized"), plt.imshow(binary_orientated, cmap='gray')
    # plt.subplot(2, 4, 7), plt.title("Breast mask"), plt.imshow(breast_mask, cmap='gray')
    # plt.subplot(2, 4, 8), plt.title("Result"), plt.imshow(breast, cmap='gray')
    # plt.show()


    return breast



def smooth_muscle(without_muscle, mirrored, without_labels):
    """
    Smooths the muscle-removed mammogram image by applying morphological operations 
    to clean noise, fill gaps, and ensure a single connected region.

    Args:
        without_muscle (np.ndarray): Grayscale image with the muscle region removed.
        mirrored (bool): Indicates if the original image was mirrored during preprocessing.
        without_labels (np.ndarray): Original mammogram image without annotations.

    Returns:
        tuple:
            - np.ndarray: Smoothed grayscale image with the muscle region removed.
            - np.ndarray: Binary mask of the smoothed breast region.
    """
    
    #Binarize
    _, binary = cv2.threshold(without_muscle, 1, 255, cv2.THRESH_BINARY)
    binary = keep_largest_object(binary)

    #Opening for eliminating noise and separate regions
    radius = 10
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    opening_smooth = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_opening)

    #Closing for fill some regions
    radius = 30
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    closing_smooth = cv2.morphologyEx(opening_smooth, cv2.MORPH_CLOSE, kernel)

    #Keep largest binary object after applying morphological operators
    without_muscle_smooth_binary = keep_largest_object(closing_smooth)

    #Original
    if mirrored == True:
        without_muscle_smooth_binary = cv2.flip(without_muscle_smooth_binary, 1)

    without_muscle_smooth =  cv2.bitwise_and(without_labels, without_muscle_smooth_binary)


    plt.figure(figsize=(20, 20))
    plt.subplot(2, 3, 1), plt.title("Original"), plt.imshow(without_muscle, cmap='gray')
    plt.subplot(2, 3, 2), plt.title("Binarized"), plt.imshow(binary, cmap='gray')
    plt.subplot(2, 3, 3), plt.title("Opening"), plt.imshow(opening_smooth, cmap='gray')
    plt.subplot(2, 3, 4), plt.title("Closing"), plt.imshow(closing_smooth, cmap='gray')
    plt.subplot(2, 3, 5), plt.title("Biggest object"), plt.imshow(without_muscle_smooth_binary, cmap='gray')
    plt.subplot(2, 3, 6), plt.title("Result"), plt.imshow(without_muscle_smooth, cmap='gray')
    plt.show()

    return without_muscle_smooth, without_muscle_smooth_binary









def process_images_in_directory(directory_path):

    for filename in os.listdir(directory_path):

        if filename.endswith(".jpg"):
            image_path = os.path.join(directory_path, filename)

            mammography = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if mammography is None:
                print(f"Could not load image: {image_path}")
                continue
            

            #Remove labels from mammography
            without_labels = remove_labels(mammography)

            #Orientate all breasts in the same direction
            breast_orientated, mirrored = breast_orientate(without_labels)

            #Remove muscle from mammography
            without_muscle = remove_muscle(without_labels, breast_orientated)

            #Get a smoother mask
            without_muscle_smooth, without_muscle_smooth_binary  = smooth_muscle(without_muscle, mirrored,without_labels)

            surrounded_breast, _ = cv2.findContours(without_muscle_smooth_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(surrounded_breast, key=cv2.contourArea)

            contoured_image  = cv2.cvtColor(mammography, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contoured_image , [largest_contour], -1, (255, 0, 0), 2) 

            #GUARDAR IMAGENES CON CONTORNO E IMAGENES SEGMENTADAS

            image_name = os.path.basename(image_path)
            image_name_ext = os.path.splitext(image_name)[0]

            #save_image('results/segmentations/' + f'{image_name_ext}_sgmt.jpg',without_muscle_smooth )
            #save_image('results/contourns/' + f'{image_name_ext}_cnt.jpg',contoured_image )

            # plt.figure(figsize=(20, 20))
            # plt.subplot(2, 4, 1), plt.title("Original"), plt.imshow(mammography, cmap='gray')
            # plt.subplot(2, 4, 2), plt.title("Sin etiquetas"), plt.imshow(without_labels, cmap='gray')
            # plt.subplot(2, 4, 3), plt.title("Orientada"), plt.imshow(breast_orientated, cmap='gray')
            # plt.subplot(2, 4, 4), plt.title("Sin musculo"), plt.imshow(without_muscle, cmap='gray')
            # plt.subplot(2, 4, 5), plt.title("Sin musculo smooth"), plt.imshow(without_muscle_smooth, cmap='gray')
            # plt.subplot(2, 4, 6), plt.title("Binaria"), plt.imshow(bin_contour, cmap='gray')
            # plt.subplot(2, 4, 7), plt.title("Contorno"), plt.imshow(contoured_image, cmap='gray')
            # plt.show()

            

            

def process_all_directories():
    
    base_directory = "data"
    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)
        
        if os.path.isdir(subdir_path):
            print(f"Processing images from: {subdir_path}")
            process_images_in_directory(subdir_path)


 