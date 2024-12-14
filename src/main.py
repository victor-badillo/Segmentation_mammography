import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label, find_objects
from skimage.segmentation import active_contour
from skimage.draw import polygon

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


def visualize_image(title, image):
    
    cv2.imshow(title, image )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

            #breast = breast_orientate(without_labels) 

            

def process_all_directories():
    # Recorrer todos los subdirectorios en el directorio base
    base_directory = "data"
    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)
        
        if os.path.isdir(subdir_path):
            print(f"Procesando imágenes en: {subdir_path}")
            process_images_in_directory(subdir_path)




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

    #Aumentar contraste para que otsu capture mejor las regiones

    #_, binarizada = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binarizada = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)

    #SEAPARAR POSIBLES REGIONES UNIDAS
    kernel_separate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Ajusta el tamaño según el dataset
    binarizada = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel_separate)

    sin_etiquetas = keep_largest_object(binarizada)

    clean =  cv2.bitwise_and(image, sin_etiquetas)

    radius = 10  # Ajusta el radio según sea necesario
    kernel_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))  # Ajusta el tamaño según el dataset
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

def apply_snake(image, columns_removed):
    """
    Ajusta un snake en forma de triángulo rectángulo para segmentar el músculo.

    Args:
        image (np.ndarray): Imagen binaria recortada.
        columns_removed (int): Número de columnas eliminadas a la izquierda.

    Returns:
        np.ndarray: Máscara ajustada al músculo con las dimensiones originales.
    """
    # Dimensiones de la imagen recortada
    rows, cols = image.shape

    # Inicializar un triángulo rectángulo pequeño en la esquina superior izquierda
    triangle = np.zeros_like(image, dtype=np.uint8)
    
    # Coordenadas del triángulo
    r = [20, 20, 30]  # Filas: (20, 20) para el lado horizontal, y (30) para el lado vertical
    c = [20, 30, 20]  # Columnas: (20, 30) para el lado horizontal, y (20) para el lado vertical
    
    # Generar la máscara inicial del triángulo
    rr, cc = polygon(r, c)
    triangle[rr, cc] = 1

    # Visualización de la máscara inicial del triángulo (opcional, para depuración)
    plt.imshow(triangle, cmap='gray')
    plt.title("Triángulo inicial")
    plt.show()

    # Aplicar snake (Active Contour)
    snake = active_contour(
        image.astype(float),  # Imagen en escala de grises
        snake=triangle.astype(float),  # Máscara inicial
        alpha=0.1,  # Peso de suavidad
        beta=0.5,  # Peso de rigidez
        gamma=0.01,  # Velocidad de evolución
    )

    # Crear la máscara resultante con el snake
    muscle_mask = np.zeros_like(image, dtype=np.uint8)
    rr = np.round(snake[:, 0]).astype(int)
    cc = np.round(snake[:, 1]).astype(int)

    # Asegurar que los índices están dentro de los límites
    rr = np.clip(rr, 0, rows - 1)
    cc = np.clip(cc, 0, cols - 1)

    muscle_mask[rr, cc] = 255

    visualize_image('mascara',muscle_mask)

    # Restaurar las dimensiones originales añadiendo columnas negras a la izquierda
    original_shape = (image.shape[0], image.shape[1] + columns_removed)
    muscle_mask_full = np.zeros(original_shape, dtype=np.uint8)
    muscle_mask_full[:, columns_removed:] = muscle_mask

    return muscle_mask_full


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



if __name__ == "__main__":

    imagen_path = "data/Glandular-graso/mdb045.jpg"

    image = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {imagen_path}")
    

    without_labels = pre_process(image)
    #En este punto tengo la imagen sin etiquetas

    breast = breast_orientate(without_labels) 
    #Mama en el mismo sentido para todas



    cropped_breast, columns_removed = remove_empty_columns(breast)


    seed_point = (30, 30) 
    muscle_mask = region_growing(cropped_breast, seed_point, threshold=15)

    muscle_mask = restore_columns(without_labels, muscle_mask, columns_removed)

    #FALTA HACER UN TOP HAT, RELLENAR HUECOS Y APLANAR PICOS
    ############
    _, binary_orientated = cv2.threshold(breast, 1, 255, cv2.THRESH_BINARY)
    without_muscle = np.clip(binary_orientated - muscle_mask, 0, 255).astype(np.uint8)
    


    result = cv2.bitwise_and(breast, without_muscle)

    


    plt.figure(figsize=(20, 20))
    plt.subplot(2, 4, 1), plt.title("Sin etiquetas"), plt.imshow(without_labels, cmap='gray')
    plt.subplot(2, 4, 2), plt.title("Orientada"), plt.imshow(breast, cmap='gray')
    plt.subplot(2, 4, 3), plt.title("Cropedd"), plt.imshow(cropped_breast, cmap='gray')
    plt.subplot(2, 4, 4), plt.title("Muscle Mask"), plt.imshow(muscle_mask, cmap='gray')
    plt.subplot(2, 4, 5), plt.title("Without Muscle"), plt.imshow(without_muscle, cmap='gray')
    plt.subplot(2, 4, 6), plt.title("Result"), plt.imshow(result, cmap='gray')
    plt.show()
    


