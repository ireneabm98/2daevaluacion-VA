# Importamos las librerías necesarias
#BrionesMagallon IreneAmeyalli
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Leemos la imagen
image = cv2.imread("j.jpg")

# Convertimos la imagen a RGB

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Aplicamos clustering con kmeans para reducir el número de colores en la imagen
valores_pixeles = np.float32(image.reshape((-1, 3)).copy())
cr = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, etiquetas, centros = cv2.kmeans(valores_pixeles, 4, None, cr, 10, cv2.KMEANS_RANDOM_CENTERS)

centro = np.uint8(centros)
image_clustering = centro[etiquetas.flatten()]
image_clustering = image_clustering.reshape((image.shape))

# Convertimos la imagen a escala de grises
gray = cv2.cvtColor(image_clustering, cv2.COLOR_BGR2GRAY)

# Aplicamos threshold para binarizar la imagen
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Aplicamos el algoritmo watershed
# Primero creamos la máscara de fondo
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Buscamos la región más grande del fondo
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Buscamos la región más pequeña de los objetos
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Buscamos la región desconocida
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Aplicamos watershed
ret, markers = cv2.connectedComponents(sure_fg)

#Agregamos un valor de +1 a todos los marcadores para evitar que el fondo quede con vallor 0

markers += 1

#ahora marcamos los objetos desconocidos con el valor 0

markers[unknown == 255] = 0

#aplicamos watershed

markers = cv2.watershed(image_clustering, markers)
image_clustering[markers == -1] = [255, 0, 0]

#Imagen resultante

plt.title("Imagen segmentada con watershed")
plt.imshow(image_clustering)
plt.show()

#Buscamos los contornos de los objetos segmentados

contornos, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

#Dibujamos los contornos encontrados en la imagen Original
cv2.drawContours(image, contornos, -1, (0, 255, 0), 2)
plt.title("Imagen original con contornos dibujados")
plt.imshow(image)
plt.show()



#Seleccionamos los contornos de jitomate2 y jitomate4

jitomates = []
for i, c in enumerate(contornos):
    # Calculamos el área del contorno
    area = cv2.contourArea(c)
    # Seleccionamos solo los contornos con área mayor a 10000
    if area > 10000:
        jitomates.append(c)

        for jitomate in jitomates:
            # Calculamos la bounding box del jitomate
            #x, y, w, h = cv2.boundingRect(jitomate)
            # Calculamos el punto medio de la bounding box
            #x_middle = x + w / 2
            # Dibujamos una línea vertical en el punto medio
            #cv2.line(image, (x_middle, 0), (x_middle, image.shape[0]), (255, 0, 0), 2)

            # Calcula la coordenada Y media de cada contorno
            y_middles = [int(np.mean(contorno[:, 0, 1])) for contorno in contornos]

            # Dibuja una línea horizontal en el centro de cada contorno
            for y_middle in y_middles:
                cv2.line(image, (0, y_middle), (image.shape[1], y_middle), (255, 0, 0), 2)

            plt.title("Imagen final")
            plt.imshow(image)
            plt.show()