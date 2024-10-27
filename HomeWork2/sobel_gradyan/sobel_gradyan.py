import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle ve gri tonlamaya çevir
image = cv2.imread("for_sobel.png", cv2.IMREAD_GRAYSCALE)

# x ve y doğrultusunda Sobel gradyanları hesapla
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Gradyan büyüklüğünü ve yönünü hesapla
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
gradient_direction = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)  # Dereceye çevir

# Gradyan büyüklüğünü normalize et ve 8 bit olarak sakla
gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

# Görüntüleri göster
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Orijinal Görüntü")
plt.imshow(image, cmap="gray")
plt.subplot(1, 3, 2)
plt.title("Gradyan Büyüklüğü")
plt.imshow(gradient_magnitude, cmap="gray")
plt.subplot(1, 3, 3)
plt.title("Gradyan Yönü (Açı)")
plt.imshow(gradient_direction, cmap="hsv")  # HSV ile görselleştir
plt.colorbar(label="Derece")
plt.show()
