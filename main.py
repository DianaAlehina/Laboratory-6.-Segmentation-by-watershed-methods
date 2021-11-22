import cv2.cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def show4(img1, img2, img3, img4):
    plt.subplot(2, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.title('Original')
    plt.subplot(2, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.title('Gradient')
    plt.subplot(2, 2, 3)
    plt.imshow(img3, cmap='gray')
    plt.axis('off')
    plt.title('Gradient watershed')
    plt.subplot(2, 2, 4)
    plt.imshow(img4, cmap='gray')
    plt.axis('off')
    plt.title('Original watershed')
    plt.show()


def open_image(filename):
    try:
        image = cv.imread(filename)
        return image
    except FileNotFoundError:
        print("Файл не найден")
        return


def watershed(original):
    # Преобразование изображение к полутоновому изображению
    gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
    # Преобразование изображение к обратно-бинарному
    _, thresh = cv.threshold(gray, 70, 255, cv.THRESH_BINARY_INV)

    # Поиск градиента
    # Градиент = разница между дилатацией и эрозией изображения.
    kernel = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)
    erode = cv.erode(thresh, kernel, iterations=2)
    dilate = cv.dilate(thresh, kernel, iterations=3)
    _, dilate_inv = cv.threshold(dilate, 1, 128, cv.THRESH_BINARY_INV)
    gradient = cv.add(erode, dilate_inv)

    gradient_watershed = gradient.copy()
    original_watershed = original.copy()

    # Находим маркеры (линии водораздела)
    gradient32 = np.int32(gradient)
    marker = cv.watershed(original, gradient32)

    # Накладываем линин водораздела на градиентное и исходное изображение
    gradient_watershed[marker == -1] = [255]
    original_watershed[marker == -1] = [255, 255, 255]

    show4(image, gradient, gradient_watershed, original_watershed)
    return


if __name__ == '__main__':
    filename = "watershed.png"
    image = open_image(filename)
    watershed(image)

# hist, bins = np.histogram(gradient.ravel(), 256, [0, 256])
# print("hist", hist)

# Есть внутренний маркер, все, что 100 процентов область входит в то что мы ищем
# Есть внешний маркер, все, что 100 процентов не входит в то что мы ищем
# Метод водораздела работает с тем, что по середине между маркерами,
# он среди этой жирной границы находит точную (нет-потому что плохой алгоритм)