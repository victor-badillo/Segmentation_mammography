import numpy as np
import cv2
from scipy.stats import kurtosis, skew
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt
import re
import csv
import os
import pandas as pd
from utilities import visualize_image, plot_histogram


def extract_ground_truth(input_file='data/Info.txt', output_file='ground_truth.csv'):
    """
    Extracts relevant information from an Info.txt file and saves it in a CSV file.

    Args:
        input_file (str): Path to the input file (Info.txt).
        output_file (str): Path to the CSV file where the processed information will be saved.
    """
    #Regular expression to extract fields
    pattern = r"(mdb\d+)\s+([FGD])\s+(\w+)\s*([BM])?\s*(\d+)?\s*(\d+)?\s*(\d+)?"

    #Process the file
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['ID', 'Tissue_Type', 'Abnormality', 'Severity', 'X', 'Y', 'Radius'])  # Encabezado
        
        for line in infile:
            match = re.match(pattern, line)
            if match:
                id_, tissue, abnormality, severity, x, y, radius = match.groups()
                writer.writerow([id_, tissue, abnormality, severity, x, y, radius])

    print(f"Ground truth saved to {output_file}")



def measure_features(img_segment):
    """
    Computes features of a segmented image.

    Parameters:
        img_segment (numpy.ndarray): Segmented image (breast area).

    Returns:
        list: Feature vector [average_brightness, kurtosis, skewness,prop].
    """
    #Average brightness
    non_zero_pixels = img_segment[img_segment > 0]
    average_brightness = np.mean(non_zero_pixels) if non_zero_pixels.size > 0 else 0


    #Kurtosis and skewness
    kurt = kurtosis(non_zero_pixels, fisher=True) if non_zero_pixels.size > 0 else 0
    skewness = skew(non_zero_pixels) if non_zero_pixels.size > 0 else 0

    #Proportion of high-intensity pixels
    threshold = 0.7 * np.max(non_zero_pixels)
    prop = np.sum(non_zero_pixels > threshold) / non_zero_pixels.size

    return [average_brightness,kurt,skewness,prop]



def classify_image(features):
    """
    Classifies an image based on its features.

    Parameters:
        features (list): Feature vector [average_brightness, kurtosis, skewness, prop].

    Returns:
        str: Image class ('Fatty', 'Fatty-glandular', 'Dense-glandular').
    """
    average_brightness, kurt, skewness,prop = features


    if average_brightness < 130:        #Darker image -> Fatty
        if kurt > 0 and skewness < 0:   #Low dispersion and more uniform distribution
            return 'Fatty'
        else:                           #Higher variability or intensity concentration
            return 'Fatty-glandular'

    elif 130 <= average_brightness <= 140: #Intermediate brightness -> Fatty-glandular
        if prop < 0.32:
            return 'Fatty-glandular'
        else:
            return 'Dense-glandular'
    else:                                  #Brighter image -> Fibro-dense
            return 'Dense-glandular'
    

        

def evaluate_classification(y_true, y_pred):
    """
    Computes evaluation metrics for image classification.

    Parameters:
        y_true (list): List of true classes.
        y_pred (list): List of predicted classes.

    Returns:
        None: Prints the evaluation metrics.
    """
    # Display overall accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.2f}")

    # Display detailed report (precision, recall, f1-score)
    report = classification_report(y_true, y_pred, target_names=['Fatty', 'Fatty-glandular', 'Dense-glandular'])
    print("Classification Report:\n")
    print(report)



def classify_all_images(segmentation_dir, ground_truth_file):
    """
    Classifies all segmented images in a directory and compares the results with the ground truth values.

    Parameters:
        segmentation_dir (str): Path to the directory containing segmented images.
        ground_truth_file (str): Path to the ground_truth.csv file.

    Returns:
        None: Displays the confusion matrix.
    """
    # Load ground truth
    ground_truth = pd.read_csv(ground_truth_file)
    ground_truth_dict = dict(zip(ground_truth['ID'], ground_truth['Tissue_Type']))

    # Map Tissue_Type to class names
    class_mapping = {'F': 'Fatty', 'G': 'Fatty-glandular', 'D': 'Dense-glandular'}

    y_true = []  # True values
    y_pred = []  # Predicted values

    # Process each image in the directory
    for filename in os.listdir(segmentation_dir):
        if filename.endswith('_sgmt.jpg'):  # Verify expected file extension

            image_id = filename.replace('_sgmt.jpg', '')  # Extract ID without suffix
            if image_id in ground_truth_dict:  # Check if it exists in the ground truth
                # Read image
                img_path = os.path.join(segmentation_dir, filename)
                img_segment = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # Measure features and classify
                features = measure_features(img_segment)
                predicted_class = classify_image(features)

                # Get true class
                true_class = class_mapping[ground_truth_dict[image_id]]

                print(image_id, '   TRUE CLASS: ' + true_class, '\tPREDICTED CLASS: ' + predicted_class)

                # Store in lists
                y_true.append(true_class)
                y_pred.append(predicted_class)

    # Confusion matrix
    labels = ['Fatty', 'Fatty-glandular', 'Dense-glandular']
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Evaluate classification
    evaluate_classification(y_true, y_pred)




