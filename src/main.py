import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label, find_objects

OUTPUT_IMAGES = "resultados/"

def save_image(image_name, image):
    img_path = OUTPUT_IMAGES + image_name
    success = cv2.imwrite(img_path, image)

    if success:
        print(f"La imagen ==> {image_name} ==> se guardó correctamente.")
    else:
        print(f"Error al guardar la imagen ==> {img_path}")

def plot_histogram(imagen):


    # Calcular el histograma de la imagen
    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])

    # Mostrar el histograma
    plt.figure(figsize=(10, 5))
    plt.title("Histograma de la Imagen")
    plt.xlabel("Intensidad de píxel")
    plt.ylabel("Frecuencia")
    plt.plot(histograma)
    plt.xlim([0, 256])  # El rango de intensidades de píxel es de 0 a 255
    plt.show()



def process_images_in_directory(directory_path):
    # Listar todas las imágenes en el directorio
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg"):  # Solo procesar archivos .jpg
            ruta_imagen = os.path.join(directory_path, filename)

            # Cargar la imagen en escala de grises
            imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

            if imagen is None:
                print(f"No se pudo cargar la imagen en la ruta: {ruta_imagen}")
                continue

            # Umbralización usando Otsu para encontrar el umbral óptimo
            _, binarizada = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Operación de apertura morfológica para eliminar etiquetas
            kernel_apertura = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))  # Ajusta el tamaño según el dataset
            binarizada_sin_etiquetas = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel_apertura)

            # Eliminar etiquetas usando bitwise_and
            sin_etiquetas = cv2.bitwise_and(imagen, binarizada_sin_etiquetas)

            # Mostrar y guardar los resultados
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1), plt.title("Imagen Original"), plt.imshow(imagen, cmap='gray')
            plt.subplot(1, 3, 2), plt.title("Binarizada con Otsu"), plt.imshow(binarizada, cmap='gray')
            plt.subplot(1, 3, 3), plt.title("Resultado Sin Etiquetas"), plt.imshow(sin_etiquetas, cmap='gray')
            plt.show()

            img_sin_musculo, mascara_musculo = eliminar_musculo(sin_etiquetas)

            # Mostrar resultados
            plt.figure(figsize=(15, 10))
            plt.subplot(1, 3, 1), plt.title("Imagen Sin Etiquetas"), plt.imshow(sin_etiquetas, cmap='gray')
            plt.subplot(1, 3, 2), plt.title("Máscara del Músculo"), plt.imshow(mascara_musculo, cmap='gray')
            plt.subplot(1, 3, 3), plt.title("Imagen Sin Músculo"), plt.imshow(img_sin_musculo, cmap='gray')
            plt.show()

def process_all_directories():
    # Recorrer todos los subdirectorios en el directorio base
    base_directory = "data"
    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)
        
        if os.path.isdir(subdir_path):
            print(f"Procesando imágenes en: {subdir_path}")
            process_images_in_directory(subdir_path)


def visualize_image(title, image):
    
    cv2.imshow(title, image )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def erase_labels(imagen):
    # Umbralización usando Otsu para encontrar el umbral óptimo
    _, binarizada = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Operación de apertura morfológica para eliminar etiquetas
    kernel_apertura = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))  # Ajusta el tamaño según el dataset
    binarizada_sin_etiquetas = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel_apertura)

    # Eliminar etiquetas usando bitwise_and
    sin_etiquetas = cv2.bitwise_and(imagen, binarizada_sin_etiquetas)

    # Mostrar y guardar los resultados
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1), plt.title("Imagen Original"), plt.imshow(imagen, cmap='gray')
    plt.subplot(1, 3, 2), plt.title("Binarizada con Otsu"), plt.imshow(binarizada, cmap='gray')
    plt.subplot(1, 3, 3), plt.title("Resultado Sin Etiquetas"), plt.imshow(sin_etiquetas, cmap='gray')
    plt.show()

    return sin_etiquetas


def keep_largest_object(binary_image):
    # Etiquetar componentes conectados
    labeled_image, num_features = label(binary_image)
    
    # Encontrar el área de cada componente conectado
    object_slices = find_objects(labeled_image)
    areas = [np.sum(labeled_image[obj_slice] == label_id + 1) for label_id, obj_slice in enumerate(object_slices)]
    
    # Identificar el índice del componente más grande
    largest_component_idx = np.argmax(areas) + 1  # Los índices empiezan en 1 para objetos etiquetados

    # Crear una máscara con solo el objeto más grande
    largest_object_mask = (labeled_image == largest_component_idx).astype(np.uint8)
    
    return largest_object_mask * 255

def pre_process(image):

    _, binarizada = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sin_etiquetas = keep_largest_object(binarizada)

    clean =  cv2.bitwise_and(image, sin_etiquetas)

    kernel_apertura = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # Ajusta el tamaño según el dataset
    clean_smooth = cv2.morphologyEx(sin_etiquetas, cv2.MORPH_OPEN, kernel_apertura)

    smooth_final =  cv2.bitwise_and(image, clean_smooth)

    plt.figure(figsize=(20, 20))
    plt.subplot(2, 4, 1), plt.title("Original"), plt.imshow(image, cmap='gray')
    plt.subplot(2, 4, 2), plt.title("Binarizada"), plt.imshow(binarizada, cmap='gray')
    plt.subplot(2, 4, 3), plt.title("Sin etiquetas"), plt.imshow(sin_etiquetas, cmap='gray')
    plt.subplot(2, 4, 4), plt.title("Final"), plt.imshow(clean, cmap='gray')
    plt.subplot(2, 4, 5), plt.title("Binary Smooth"), plt.imshow(clean_smooth, cmap='gray')
    plt.subplot(2, 4, 6), plt.title("Final Smooth"), plt.imshow(smooth_final, cmap='gray')
    plt.show()

    return smooth_final


if __name__ == "__main__":

    imagen_path = "data/Graso/mdb009.jpg"

    image = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {imagen_path}")
    

    without_labels = pre_process(image)
    visualize_image('without labels after preprocessing', without_labels)
