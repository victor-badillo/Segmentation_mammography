from segmentation import *
from classification import *
import pandas as pd



if __name__ == "__main__":


    #Segment mammograms, obtain segmented breast and contoured image
    process_all_directories()

    #Clasify results

    #Generate ground truth file from Info.txt in /data
    #extract_ground_truth()

