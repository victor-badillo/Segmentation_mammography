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

            # Guardar la imagen procesada (sin etiquetas)
            save_image(f"sin_etiquetas_{filename}", sin_etiquetas)

def process_all_directories(base_directory):
    # Recorrer todos los subdirectorios en el directorio base
    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)
        
        if os.path.isdir(subdir_path):
            print(f"Procesando imágenes en: {subdir_path}")
            process_images_in_directory(subdir_path)

if __name__ == "__main__":

    base_directory = "data"  # Ruta del directorio principal donde están los subdirectorios
    process_all_directories(base_directory)
