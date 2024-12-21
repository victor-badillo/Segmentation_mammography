import cv2
import matplotlib.pyplot as plt


def save_image(image_name, image):
    img_path = image_name
    success = cv2.imwrite(img_path, image)

    if success:
        print(f"La imagen ==> {image_name} ==> se guardó correctamente.")
    else:
        print(f"Error al guardar la imagen ==> {img_path}")

def plot_histogram(imagen, title):


    # Calcular el histograma de la imagen
    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])

    # Mostrar el histograma
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel("Intensidad de píxel")
    plt.ylabel("Frecuencia")
    plt.plot(histograma)
    plt.xlim([0, 256])  # El rango de intensidades de píxel es de 0 a 255
    plt.show()


def visualize_image(title, image):
    
    cv2.imshow(title, image )
    cv2.waitKey(0)
    cv2.destroyAllWindows()