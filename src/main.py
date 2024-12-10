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
            
            without_labels = pre_process(imagen)

            breast = breast_orientate(without_labels) 

            # # Umbralización usando Otsu para encontrar el umbral óptimo
            # _, binarizada = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # # Operación de apertura morfológica para eliminar etiquetas
            # kernel_apertura = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))  # Ajusta el tamaño según el dataset
            # binarizada_sin_etiquetas = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel_apertura)

            # # Eliminar etiquetas usando bitwise_and
            # sin_etiquetas = cv2.bitwise_and(imagen, binarizada_sin_etiquetas)

            # # Mostrar y guardar los resultados
            # plt.figure(figsize=(10, 10))
            # plt.subplot(1, 3, 1), plt.title("Imagen Original"), plt.imshow(imagen, cmap='gray')
            # plt.subplot(1, 3, 2), plt.title("Binarizada con Otsu"), plt.imshow(binarizada, cmap='gray')
            # plt.subplot(1, 3, 3), plt.title("Resultado Sin Etiquetas"), plt.imshow(sin_etiquetas, cmap='gray')
            # plt.show()

            # img_sin_musculo, mascara_musculo = eliminar_musculo(sin_etiquetas)

            # # Mostrar resultados
            # plt.figure(figsize=(15, 10))
            # plt.subplot(1, 3, 1), plt.title("Imagen Sin Etiquetas"), plt.imshow(sin_etiquetas, cmap='gray')
            # plt.subplot(1, 3, 2), plt.title("Máscara del Músculo"), plt.imshow(mascara_musculo, cmap='gray')
            # plt.subplot(1, 3, 3), plt.title("Imagen Sin Músculo"), plt.imshow(img_sin_musculo, cmap='gray')
            # plt.show()

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

    #_, binarizada = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binarizada = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)

    kernel_separate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Ajusta el tamaño según el dataset
    binarizada = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel_separate)

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

def adjust_image_orientation(image, original):
    """
    Ajusta la orientación de la imagen basado en la posición del primer píxel blanco desde la esquina superior izquierda.
    Si el píxel blanco se encuentra en la segunda mitad de las columnas, se voltea la imagen horizontalmente.

    Args:
        image (np.ndarray): Imagen binaria en escala de grises.

    Returns:
        np.ndarray: Imagen ajustada.
    """
    rows, cols = image.shape
    
    # Recorrer la imagen desde arriba hacia abajo, izquierda a derecha
    for row in range(rows):
        for col in range(cols):
            if image[row, col] == 255:  # Encontrar el primer píxel blanco
                print(f"Primer píxel blanco encontrado en ({row}, {col})")
                # Si el píxel está en la segunda mitad de las columnas, voltear la imagen
                if col >= cols // 2:
                    print("El píxel está en la segunda mitad, volteando la imagen.")
                    return cv2.flip(original, 1)  # Voltear horizontalmente
                else:
                    print("El píxel está en la primera mitad, dejando la imagen como está.")
                    return original

    print("No se encontró ningún píxel blanco, devolviendo la imagen sin cambios.")
    return original 


def breast_orientate(without_labels):

    # Binarizar y encontrar contornos
    _, binary_image = cv2.threshold(without_labels, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("No se encontraron contornos.")
    else:
        #Elegir el contorno más grande por área
        largest_contour = max(contours, key=cv2.contourArea)
        
        #Crear una copia de la imagen para dibujar el contorno
        contoured_image = np.zeros_like(without_labels)
        cv2.drawContours(contoured_image, [largest_contour], -1, 255, 2)


    # kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    # vertical_lines = cv2.morphologyEx(contoured_image, cv2.MORPH_OPEN, kernel_vertical)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 400))
    vertical_lines = cv2.erode(contoured_image, kernel_vertical)

    breast_oriented = adjust_image_orientation(vertical_lines, without_labels)


    plt.figure(figsize=(20, 20))
    plt.subplot(2, 4, 1), plt.title("Sin etiquetas smooth"), plt.imshow(without_labels, cmap='gray')
    plt.subplot(2, 4, 2), plt.title("Binarizada"), plt.imshow(binary_image, cmap='gray')
    plt.subplot(2, 4, 3), plt.title("Contornos"), plt.imshow(contoured_image, cmap='gray')
    plt.subplot(2, 4, 4), plt.title("Linea vertical"), plt.imshow(vertical_lines, cmap='gray')
    plt.subplot(2, 4, 5), plt.title("Mama derecha"), plt.imshow(breast_oriented, cmap='gray')
    plt.show()

    return breast_oriented





if __name__ == "__main__":

    imagen_path = "data/Glandular-graso/mdb045.jpg"

    image = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {imagen_path}")
    

    without_labels = pre_process(image)
    #En este punto tengo la imagen sin etiquetas

    breast = breast_orientate(without_labels) 
    #Mama en el mismo sentido para todas

    



    


