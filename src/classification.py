import numpy as np
import cv2
from scipy.stats import kurtosis, skew
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import re
import csv


def extract_ground_truth(input_file='data/Info.txt', output_file='groun_truth.csv'):
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
        list: Vector con las características [brillo_promedio, contraste, curtosis, asimetría, porcentaje_bordes].
    """
    # Brillo promedio y contraste
    brillo_promedio = np.mean(img_segment)
    contraste = np.std(img_segment)

    # Curtosis y asimetría
    intensities = img_segment.flatten()
    curt = kurtosis(intensities)
    asym = skew(intensities)

    # Porcentaje de bordes
    edges = cv2.Canny(img_segment, threshold1=50, threshold2=150)
    porcentaje_bordes = np.sum(edges > 0) / edges.size * 100

    return [brillo_promedio, contraste, curt, asym, porcentaje_bordes]

def classify_image(features):
    """
    Clasifica una imagen basada en sus características.

    Parámetros:
        features (list): Vector de características [brillo_promedio, contraste, curtosis, asimetría, porcentaje_bordes].

    Retorna:
        str: Clase de la imagen ('Graso', 'Glandular-graso', 'Glandular-denso').
    """
    brillo_promedio, contraste, curt, asym, porcentaje_bordes = features

    if brillo_promedio < 70 and contraste < 30:
        return 'Graso'
    elif brillo_promedio < 130 and porcentaje_bordes < 15:
        return 'Glandular-graso'
    elif brillo_promedio >= 130 or porcentaje_bordes > 15:
        return 'Glandular-denso'
    else:
        return 'Glandular-graso'  # Caso base intermedio

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

