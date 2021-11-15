import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image


def show4(img1, img2, img3, img4, title=''):
    plt.subplot(2, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.title('Original')
    plt.subplot(2, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.title('Frequency filter low ' + title)
    plt.subplot(2, 2, 3)
    plt.imshow(img3, cmap='gray')
    plt.axis('off')
    plt.title('Spectrum result low ' + title)
    plt.subplot(2, 2, 4)
    plt.imshow(img4, cmap='gray')
    plt.axis('off')
    plt.title('Image result low ' + title)
    plt.show()


def open_image(filename):
    try:
        image = cv2.imread(filename, 0)
        return image
    except FileNotFoundError:
        print("Файл не найден")
        return


def watershed():
    return


def normalization(image):
    Imax = np.max(image)
    Imin = np.min(image)
    Omin, Omax = 0, 255
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    image = a * image + b
    image = image.astype(np.uint8)
    return image



if __name__ == '__main__':
    filename = "watershed.png"
    image_original = open_image(filename)

    # Применение градиентного оператора Собела к оригинальному изображению
    image_sobelx = cv2.Sobel(image_original, cv2.CV_64F, 1, 0, 3)
    image_sobely = cv2.Sobel(image_original, cv2.CV_64F, 0, 1, 3)
    image_sobelx = cv2.convertScaleAbs(image_sobelx)
    image_sobely = cv2.convertScaleAbs(image_sobely)
    image_sobelxy = cv2.addWeighted(image_sobelx, 0.5, image_sobely, 0.5, 0)
    image_sobelxy = np.uint8(image_sobelxy)
    '''width, height = image_sobelxy.size


    for i in range(0, height):
        for j in range(0, width):
            if image_sobelxy[i, j]:'''


    show4(image_original, image_sobelx, image_sobelxy, image_sobelxy)

