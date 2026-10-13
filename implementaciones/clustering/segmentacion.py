from kmeans import Kmeans
import matplotlib.pyplot as plt

#cargar la imagen y normalizarla 
imagen_original = plt.imread('implementaciones/imagenes/402296mt.jpg') / 255.0
altura, anchura, canales = imagen_original.shape 

#aplanar imagen 
pixeles = imagen_original.reshape(-1, 3)

#kmeans 
ks = [2, 5, 10]
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax = ax.flatten()
ax[0].imshow(imagen_original)
ax[0].set_title('Imagen Original')
ax[0].axis('off')
for i, k in enumerate(ks):
    modelo = Kmeans(k)
    modelo.fit(pixeles)
    centroides = modelo.centroides
    labels = modelo.labels_

    pixeles_segmentados = centroides[labels]

    imagen_segmentada = pixeles_segmentados.reshape(altura, anchura, canales)

    ax[i+1].imshow(imagen_segmentada)
    ax[i+1].set_title(f'Imagen Segmentada (K={k})')
    ax[i+1].axis('off')

plt.tight_layout()
plt.show()
