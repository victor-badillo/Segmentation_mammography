import numpy as np
import cv2
from scipy.stats import kurtosis, skew
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import re
import csv
import os
import pandas as pd
from utilities import visualize_image, plot_histogram


def extract_ground_truth(input_file='data/Info.txt', output_file='ground_truth.csv'):
    """
    Extrae información relevante de un archivo Info.txt y la guarda en un archivo CSV.

    Args:
        input_file (str): Ruta al archivo de entrada (Info.txt).
        output_file (str): Ruta al archivo CSV donde se guardará la información procesada.
    """
    # Expresión regular para extraer los campos
    pattern = r"(mdb\d+)\s+([FGD])\s+(\w+)\s*([BM])?\s*(\d+)?\s*(\d+)?\s*(\d+)?"

    # Procesar el archivo
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['ID', 'Tissue_Type', 'Abnormality', 'Severity', 'X', 'Y', 'Radius'])  # Encabezado
        
        for line in infile:
            match = re.match(pattern, line)
            if match:
                id_, tissue, abnormality, severity, x, y, radius = match.groups()
                writer.writerow([id_, tissue, abnormality, severity, x, y, radius])

    print(f"Ground truth guardado en {output_file}")



def measure_features(img_segment):
    """
    Calcula las características de una imagen segmentada.

    Parámetros:
        img_segment (numpy.ndarray): Imagen segmentada (área mamaria).

    Retorna:
        list: Vector con las características [brillo_promedio, contraste, porcentaje_bordes].
    """
    # Brillo promedio y contraste
    non_zero_pixels = img_segment[img_segment > 0]  # Obtener solo los píxeles no cero
    brillo_promedio = np.mean(non_zero_pixels) if non_zero_pixels.size > 0 else 0  # Evitar división por cero

    contraste = np.std(non_zero_pixels) if non_zero_pixels.size > 0 else 0

    # Curtosis y asimetría (comentados)
    curt = kurtosis(non_zero_pixels, fisher=True) if non_zero_pixels.size > 0 else 0
    asym = skew(non_zero_pixels) if non_zero_pixels.size > 0 else 0

    # Porcentaje de bordes
    edges = cv2.Canny(img_segment, threshold1=20, threshold2=50)
    porcentaje_bordes = np.sum(edges > 0) / edges.size * 100



    threshold = 0.7 * np.max(non_zero_pixels)
    prop = np.sum(non_zero_pixels > threshold) / non_zero_pixels.size


    # Solo devolvemos brillo, contraste y porcentaje de bordes
    return [brillo_promedio,curt, asym, porcentaje_bordes, prop]



def classify_image(features, image_id):
    """
    Clasifica una imagen basada en sus características.

    Parámetros:
        features (list): Vector de características [brillo_promedio, contraste, porcentaje_bordes].

    Retorna:
        str: Clase de la imagen ('Graso', 'Glandular-graso', 'Glandular-denso').
    """
    brillo_promedio, curt, asym,porcentaje_bordes , prop = features

    if(brillo_promedio > 140):
        print(f"{image_id} {porcentaje_bordes:.3f}")

        # Clasificación en 3 categorías usando ifs
    if brillo_promedio < 130:  # Imagen más oscura -> Graso
        if curt > 0 and asym < 0:  # Baja dispersión y distribución más uniforme
            return 'Graso'
        else:  # Más variabilidad o concentración en la intensidad
            return 'Glandular-graso'

    elif 130 <= brillo_promedio <= 140:  # Brillo intermedio -> Glandular-graso
        if prop < 0.32:
            return 'Glandular-graso'
        else:
            return 'Glandular-denso'
    else:  # Brillo promedio > 140 -> Imagen más brillante -> Glandular-denso
            return 'Glandular-denso'
    

        

def evaluate_classification(y_true, y_pred):
    """
    Calcula métricas de evaluación para la clasificación de imágenes.

    Parámetros:
        y_true (list): Lista con las clases verdaderas.
        y_pred (list): Lista con las clases predichas.

    Retorna:
        None: Imprime las métricas de evaluación.
    """
    # Mostrar Accuracy general
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.2f}")

    # Mostrar reporte detallado (precision, recall, f1-score)
    report = classification_report(y_true, y_pred, target_names=['Graso', 'Glandular-graso', 'Glandular-denso'])
    print("Reporte de clasificación:\n")
    print(report)



def classify_all_images(segmentation_dir, ground_truth_file):
    """
    Clasifica todas las imágenes segmentadas en un directorio y compara los resultados con los valores reales.

    Parámetros:
        segmentation_dir (str): Ruta al directorio con imágenes segmentadas.
        ground_truth_file (str): Ruta al archivo ground_truth.csv.

    Retorna:
        None: Muestra la matriz de confusión.
    """
    # Cargar ground truth
    ground_truth = pd.read_csv(ground_truth_file)
    ground_truth_dict = dict(zip(ground_truth['ID'], ground_truth['Tissue_Type']))

    # Mapear Tissue_Type a nombres de clase
    class_mapping = {'F': 'Graso', 'G': 'Glandular-graso', 'D': 'Glandular-denso'}

    y_true = []  # Valores reales
    y_pred = []  # Valores predichos


    # Procesar cada imagen en el directorio
    for filename in os.listdir(segmentation_dir):
        if filename.endswith('_sgmt.jpg'):  # Verificar extensión esperada

            image_id = filename.replace('_sgmt.jpg', '')  # Extraer ID sin sufijo
            if image_id in ground_truth_dict:  # Verificar si está en el ground truth
                # Leer imagen
                img_path = os.path.join(segmentation_dir, filename)
                img_segment = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # Medir características y clasificar
                features = measure_features(img_segment)
                predicted_class = classify_image(features, image_id)


                # Obtener clase real
                true_class = class_mapping[ground_truth_dict[image_id]]

                print('TRUE CLASS:'+ true_class , 'PREDICTED:' + predicted_class)

                # Guardar en listas
                y_true.append(true_class)
                y_pred.append(predicted_class)

    # Matriz de confusión
    labels = ['Graso', 'Glandular-graso', 'Glandular-denso']
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Mostrar la matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión")
    plt.show()

    # Evaluar clasificación
    evaluate_classification(y_true, y_pred)



def probar(segmentation_dir):
    for filename in os.listdir(segmentation_dir):
        if filename.endswith('.jpg'):  # Verificar extensión esperada
            img_path = os.path.join(segmentation_dir, filename)
            img_segment = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # edges = cv2.Canny(img_segment, threshold1=20, threshold2=30)
            # porcentaje_bordes = np.sum(edges > 0) / edges.size * 100

            # print(img_path, porcentaje_bordes)
            plot_histogram(img_segment, img_path)


            # plt.figure(figsize=(10, 5))
            # plt.subplot(1, 2, 1)
            # plt.title(img_path)
            # plt.imshow(img_segment, cmap='gray')
            # plt.axis('off')

            # plt.subplot(1, 2, 2)
            # plt.title("Aumentar el contraste")
            # plt.imshow(edges, cmap='gray')
            # plt.axis('off')

            # plt.show()
            

'''
def evaluate_classifier(images, labels):
    """
    Evalúa el clasificador y genera una matriz de confusión.

    Parámetros:
        images (list): Lista de imágenes segmentadas.
        labels (list): Lista de etiquetas reales.

    Retorna:
        None
    """
    # Generar predicciones
    predictions = []
    for img in images:
        features = measure_features(img)
        prediction = classify_image(features)
        predictions.append(prediction)

    # Matriz de confusión
    classes = ['Graso', 'Glandular-graso', 'Glandular-denso']
    cm = confusion_matrix(labels, predictions, labels=classes)

    print("Matriz de Confusión:")
    print(cm)

    # Reporte de clasificación
    print("\nReporte de Clasificación:")
    print(classification_report(labels, predictions, target_names=classes))

    # Visualización de la matriz de confusión
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.show()

# Ejemplo de uso (reemplazar con tus datos reales)
# images = [...]  # Lista de imágenes segmentadas (numpy arrays)
# labels = [...]  # Lista de etiquetas reales (str: 'Graso', 'Glandular-graso', 'Glandular-denso')
# evaluate_classifier(images, labels)
'''
