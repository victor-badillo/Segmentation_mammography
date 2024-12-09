import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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



if __name__ == "__main__":

    imagen_path = "data/Graso/mdb009.jpg"

    image = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {imagen_path}")
    
    img_without_labels = erase_labels(image)

    visualize_image('sin_etiquetas', img_without_labels)

    img_with_same_orientation = cv2.flip(img_without_labels, 1)  # 1 significa flip horizontal

    visualize_image('mirror', img_with_same_orientation)

    plt.figure(figsize=(30, 30))
    plt.subplot(1, 3, 1), plt.title("Imagen sin etiquetas"), plt.imshow(img_without_labels, cmap='gray')
    plt.subplot(1, 3, 2), plt.title("Ecualizada"), plt.imshow(img_with_same_orientation, cmap='gray')
    plt.show()

    # img_eq = cv2.equalizeHist(img_without_labels)

    # visualize_image('eq', img_eq)

    # plt.figure(figsize=(30, 30))
    # plt.subplot(1, 3, 1), plt.title("Imagen sin etiquetas"), plt.imshow(img_without_labels, cmap='gray')
    # plt.subplot(1, 3, 2), plt.title("Ecualizada"), plt.imshow(img_eq, cmap='gray')
    # plt.show()