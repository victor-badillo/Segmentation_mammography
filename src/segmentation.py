import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label, find_objects
from skimage.segmentation import active_contour
from skimage.draw import polygon
from utilities import save_image



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

    plt.figure(figsize=(20, 20))
    plt.subplot(2, 4, 1), plt.title("Original"), plt.imshow(image, cmap='gray')
    plt.subplot(2, 4, 2), plt.title("Binarized"), plt.imshow(binarized, cmap='gray')
    plt.subplot(2, 4, 3), plt.title("Without Labels Binary"), plt.imshow(without_labels_binary, cmap='gray')
    plt.subplot(2, 4, 4), plt.title("Final"), plt.imshow(final, cmap='gray')
    plt.subplot(2, 4, 5), plt.title("Binary Smooth"), plt.imshow(without_labels_binary_smooth, cmap='gray')
    plt.subplot(2, 4, 6), plt.title("Final Smooth"), plt.imshow(final_smooth, cmap='gray')
    plt.show()

    return final_smooth




def perfil_muscle(without_muscle, mirrored, without_labels):
    
    # 1. Binarización
    _, binary = cv2.threshold(without_muscle, 1, 255, cv2.THRESH_BINARY)
    binary = keep_largest_object(binary)

    # 2. Apertura morfológica para eliminar ruido pequeño y separar posibles regiones
    radius = 10  # Ajusta el radio según sea necesario
    kernel_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))  # Ajusta el tamaño según el dataset
    clean_smooth = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_apertura)

    # 3. Cerrar huecos (Closing)
    radius = 30
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    close = cv2.morphologyEx(clean_smooth, cv2.MORPH_CLOSE, kernel)

    #Quedarse con al region mas grande de clean_smooth
    largest_close = keep_largest_object(close)

    if mirrored == True:
        largest_close = cv2.flip(largest_close, 1)

    close_image =  cv2.bitwise_and(without_labels, largest_close)


    # plt.figure(figsize=(20, 20))
    # plt.subplot(2, 4, 1), plt.title("Original"), plt.imshow(without_muscle, cmap='gray')
    # plt.subplot(2, 4, 2), plt.title("Binary"), plt.imshow(binary, cmap='gray')
    # plt.subplot(2, 4, 3), plt.title("Opening"), plt.imshow(clean_smooth, cmap='gray')
    # plt.subplot(2, 4, 4), plt.title("Close"), plt.imshow(close, cmap='gray')
    # plt.subplot(2, 4, 5), plt.title("Mas grande close"), plt.imshow(largest_close, cmap='gray')
    # plt.subplot(2, 4, 6), plt.title("Resultado close"), plt.imshow(close_image, cmap='gray')
    # plt.show()

    return close_image, largest_close



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
    mirrored = False
    
    # Recorrer la imagen desde arriba hacia abajo, izquierda a derecha
    for row in range(rows):
        for col in range(cols):
            if image[row, col] == 255:  # Encontrar el primer píxel blanco
                print(f"Primer píxel blanco encontrado en ({row}, {col})")
                # Si el píxel está en la segunda mitad de las columnas, voltear la imagen
                if col >= cols // 2:
                    print("El píxel está en la segunda mitad, volteando la imagen.")
                    mirrored = True
                    return cv2.flip(original, 1), mirrored  # Voltear horizontalmente
                else:
                    print("El píxel está en la primera mitad, dejando la imagen como está.")
                    return original, mirrored

    print("No se encontró ningún píxel blanco, devolviendo la imagen sin cambios.")
    return original, mirrored


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

    breast_oriented, mirrored = adjust_image_orientation(vertical_lines, without_labels)


    # plt.figure(figsize=(20, 20))
    # plt.subplot(2, 4, 1), plt.title("Sin etiquetas smooth"), plt.imshow(without_labels, cmap='gray')
    # plt.subplot(2, 4, 2), plt.title("Binarizada"), plt.imshow(binary_image, cmap='gray')
    # plt.subplot(2, 4, 3), plt.title("Contornos"), plt.imshow(contoured_image, cmap='gray')
    # plt.subplot(2, 4, 4), plt.title("Linea vertical"), plt.imshow(vertical_lines, cmap='gray')
    # plt.subplot(2, 4, 5), plt.title("Mama derecha"), plt.imshow(breast_oriented, cmap='gray')
    # plt.show()

    return breast_oriented, mirrored



def remove_empty_columns(image):
    """
    Elimina las columnas vacías (completamente negras) a la izquierda de la imagen.

    Args:
        image (np.ndarray): Imagen binaria.

    Returns:
        tuple: (np.ndarray, int) Imagen recortada y el número de columnas eliminadas.
    """
    # Encontrar las columnas que no son completamente negras
    col_sums = np.sum(image, axis=0)
    first_nonzero_col = np.argmax(col_sums > 0)  # Primera columna no vacía

    # Recortar la imagen desde la primera columna no vacía
    cropped_image = image[:, first_nonzero_col:]

    return cropped_image, first_nonzero_col



def region_growing(image, seed_point, threshold=10):
    """
    Realiza segmentación por crecimiento de regiones a partir de un punto semilla.
    
    Args:
        image (np.ndarray): Imagen en escala de grises.
        seed_point (tuple): Punto inicial (fila, columna).
        threshold (int): Diferencia máxima de intensidad para considerar homogeneidad.
    
    Returns:
        np.ndarray: Máscara binaria con la región segmentada.
    """
    rows, cols = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=bool)
    seed_value = image[seed_point]
    
    # Usar una pila para el algoritmo iterativo
    stack = [seed_point]
    
    while stack:
        x, y = stack.pop()
        
        if visited[x, y]:
            continue
        
        # Marcar como visitado
        visited[x, y] = True
        
        # Agregar a la región si cumple el criterio
        if abs(int(image[x, y]) - int(seed_value)) <= threshold:
            segmented[x, y] = 255
            
            # Agregar vecinos válidos
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                    stack.append((nx, ny))
    
    return segmented

def restore_columns(image, mask, columns_removed):
    """
    Restaura las columnas eliminadas a la izquierda de la imagen para devolver la máscara a las dimensiones originales.

    Args:
        image (np.ndarray): Imagen original sin etiquetas.
        mask (np.ndarray): Máscara generada por region growing.
        columns_removed (int): Número de columnas eliminadas inicialmente.

    Returns:
        np.ndarray: Máscara con las dimensiones originales restauradas.
    """
    # Crear una máscara del tamaño original, inicializada en negro
    restored_mask = np.zeros_like(image, dtype=np.uint8)

    # Insertar la máscara generada en la región correspondiente
    restored_mask[:, columns_removed:] = mask

    return restored_mask

def substract_muscle(without_labels, breast, original):


    cropped_breast, columns_removed = remove_empty_columns(breast)

    seed_point = (30, 30) 
    muscle_mask = region_growing(cropped_breast, seed_point, threshold=35)

    muscle_mask = restore_columns(without_labels, muscle_mask, columns_removed)

    #FALTA HACER UN TOP HAT, RELLENAR HUECOS Y APLANAR PICOS
    ############
    _, binary_orientated = cv2.threshold(breast, 1, 255, cv2.THRESH_BINARY)
    without_muscle = np.clip(binary_orientated - muscle_mask, 0, 255).astype(np.uint8)
    


    result = cv2.bitwise_and(original, without_muscle)

    # plt.figure(figsize=(20, 20))
    # plt.subplot(2, 4, 1), plt.title("Sin etiquetas"), plt.imshow(without_labels, cmap='gray')
    # plt.subplot(2, 4, 2), plt.title("Orientada"), plt.imshow(breast, cmap='gray')
    # plt.subplot(2, 4, 3), plt.title("Cropedd"), plt.imshow(cropped_breast, cmap='gray')
    # plt.subplot(2, 4, 4), plt.title("Muscle Mask"), plt.imshow(muscle_mask, cmap='gray')
    # plt.subplot(2, 4, 5), plt.title("Without Muscle"), plt.imshow(without_muscle, cmap='gray')
    # plt.subplot(2, 4, 6), plt.title("Result"), plt.imshow(result, cmap='gray')
    # plt.show()

    return result


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


            breast_gauss2, mirrored = breast_orientate(without_labels)
            breast_filter = cv2.bilateralFilter(breast_gauss2, d=50, sigmaColor=40, sigmaSpace=10)

            
            without_muscle = substract_muscle(without_labels, breast_filter, breast_gauss2)

            without_muscle_smooth, bin_contour = perfil_muscle(without_muscle, mirrored,without_labels)

            surrounded_breast, _ = cv2.findContours(bin_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(surrounded_breast, key=cv2.contourArea)

            contoured_image  = cv2.cvtColor(mammography, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contoured_image , [largest_contour], -1, (255, 0, 0), 2) 

            #GUARDAR IMAGENES CON CONTORNO E IMAGENES SEGMENTADAS

            image_name = os.path.basename(image_path)
            image_name_ext = os.path.splitext(image_name)[0]

            save_image('results/segmentations/' + f'{image_name_ext}_sgmt.jpg',without_muscle_smooth )
            save_image('results/contourns/' + f'{image_name_ext}_cnt.jpg',contoured_image )

            plt.figure(figsize=(20, 20))
            plt.subplot(2, 4, 1), plt.title("Original"), plt.imshow(mammography, cmap='gray')
            plt.subplot(2, 4, 2), plt.title("Sin etiquetas"), plt.imshow(without_labels, cmap='gray')
            plt.subplot(2, 4, 3), plt.title("Orientada"), plt.imshow(breast_gauss2, cmap='gray')
            plt.subplot(2, 4, 4), plt.title("Filtro"), plt.imshow(breast_filter, cmap='gray')
            plt.subplot(2, 4, 5), plt.title("Sin musculo"), plt.imshow(without_muscle, cmap='gray')
            plt.subplot(2, 4, 6), plt.title("Sin musculo smooth"), plt.imshow(without_muscle_smooth, cmap='gray')
            plt.subplot(2, 4, 7), plt.title("Binaria"), plt.imshow(bin_contour, cmap='gray')
            plt.subplot(2, 4, 8), plt.title("Contorno"), plt.imshow(contoured_image, cmap='gray')
            plt.show()

            

            

def process_all_directories():
    
    base_directory = "data"
    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)
        
        if os.path.isdir(subdir_path):
            print(f"Processing images from: {subdir_path}")
            process_images_in_directory(subdir_path)


 