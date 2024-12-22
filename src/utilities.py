import cv2
import matplotlib.pyplot as plt


def save_image(image_name, image):
    img_path = image_name
    success = cv2.imwrite(img_path, image)

    if success:
        print(f"Image ==> {image_name} ==> was correctly saved.")
    else:
        print(f"Error saving image ==> {img_path}")

def plot_histogram(imagen, title):

    histogram = cv2.calcHist([imagen], [0], None, [256], [0, 256])

    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel("Intensity from pixel")
    plt.ylabel("Frecuency")
    plt.plot(histogram)
    plt.xlim([0, 256]) 
    plt.show()


def visualize_image(title, image):
    
    cv2.imshow(title, image )
    cv2.waitKey(0)
    cv2.destroyAllWindows()