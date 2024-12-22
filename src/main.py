from segmentation import *
from classification import *


if __name__ == "__main__":


    #Segment mammograms, obtain segmented breast and contoured image
    process_all_directories()

    #Clasify results

    #Generate ground truth file from Info.txt in /data
    #extract_ground_truth()

    #classify_all_images('results/segmentations', 'ground_truth.csv')

