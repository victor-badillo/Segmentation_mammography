from segmentation import *
from classification import *
import pandas as pd



if __name__ == "__main__":

    # imagen_path = "data/Glandular-graso/mdb072.jpg"

    # image = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

    # if image is None:
    #     raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {imagen_path}")
    

    # without_labels = pre_process(image)
    # #En este punto tengo la imagen sin etiquetas


    # breast_gauss2, mirrored = breast_orientate(without_labels)
    # breast_filter = cv2.bilateralFilter(breast_gauss2, d=50, sigmaColor=40, sigmaSpace=10)

            
    # without_muscle = substract_muscle(without_labels, breast_filter, breast_gauss2)

    # without_muscle_smooth = perfil_muscle(without_muscle, mirrored)
    process_all_directories()