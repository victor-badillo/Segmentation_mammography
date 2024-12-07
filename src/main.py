import cv2
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_IMAGES = "resultados/"

def visualize_image(title, image):
    
    cv2.imshow(title, image )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(image_name, image):

    img_path = OUTPUT_IMAGES + image_name
    success = cv2.imwrite(img_path,image * 255)

    if success:
        print("La imagen ==> " + image_name + " ==> se guardó correctamente.")
    else:
        print("Error al guardar la imagen ==>" + img_path)

def plot_histogram(imagen_path):
    # Cargar la imagen en escala de grises
    imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

    if imagen is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {imagen_path}")

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

if __name__ == "__main__":

    # Cargar la imagen en escala de grises
    ruta_imagen = "data/Graso/mdb028.jpg"  # Ajusta la ruta correcta
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

    if imagen is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {ruta_imagen}")


    plot_histogram(ruta_imagen)
    _, binarizada = cv2.threshold(imagen, 20, 255, cv2.THRESH_BINARY)
    kernel_apertura = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))  # Ajusta el tamaño según el dataset
    binarizada_sin_etiquetas = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel_apertura)
    sin_etiquetas = cv2.bitwise_and(imagen, binarizada_sin_etiquetas)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1), plt.title("Imagen Original"), plt.imshow(imagen, cmap='gray')
    plt.subplot(1, 3, 2), plt.title("Binarizada"), plt.imshow(binarizada, cmap='gray')
    plt.subplot(1, 3, 3), plt.title("Resultado Sin Etiquetas"), plt.imshow(sin_etiquetas, cmap='gray')
    plt.show()
