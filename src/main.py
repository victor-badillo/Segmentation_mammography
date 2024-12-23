from segmentation import process_all_directories
from classification import classify_all_images, extract_ground_truth
from validation import validate_segmentations, print_results_table


if __name__ == "__main__":


    #Segment mammograms, obtain segmented breast and contoured image
    #process_all_directories()

    #Clasify results

    #Generate ground truth file from Info.txt in /datap
    #extract_ground_truth()

    #classify_all_images('results/segmentations', 'ground_truth.csv')

    data_dir = 'data'
    segmentation_dir = 'results/masks/'

    # Validate segmentations and calculate contrasts
    results_df = validate_segmentations(data_dir, segmentation_dir)

    print_results_table(results_df)

