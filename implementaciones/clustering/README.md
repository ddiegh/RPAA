## Clustering 

# K means
K means es un algoritmo de aprendizaje no supervizado que sirve para clustering. 
La idea del algoritmo de forma muy intuitiva es la siguiente: Dada un conjunto de datos $X$ queremos poder
dividirlo en $K$ clusters, entonces hacemos el algoritmo: 

* Obtener K puntos aleatorios del conjunto de datos (centroides)
* Calcular la distancia de punto en los datos a cada centroide
* Asignar cada punto al cluster más cercano
* Calcular la media en cada cluster y ese será el nuevo centroide
* Repetir desde el paso 2 y detener cuando se converge o despues de un numero maximo de iteraciones

## Ejemplo de uso 
```python 
from kmeans import Kmeans

modelo = Kmeans(k = 5)

#entrenar el modelo 
modelo.fit(X)

#etiquetas y centroides 
etiquetas = modelo.labels_
centroides = modelo.centroides
```

## Argumento matematico 

Dado un conjunto de datos $X = \{x_1, \dots, x_n\} \subset \mathbb{R}^d$ y un valor $k$ fijo (número de clusters), el algoritmo busca agrupar los datos en $k$ conjuntos. 

Consideramos:
* $M = \{\mu_1, \dots, \mu_k\}$ donde cada $\mu_k \in \mathbb{R}^d$ es el centroide del cluster $k$-ésimo.
* $C = \{C_1, \dots, C_k\}$ como los clusters (grupos) asociados a cada centroide.

**Función Objetivo**
La meta es encontrar el conjunto de centroides $M$ y una asignación de los puntos $X$ a los clusters $C$ tal que se minimice la suma de los cuadrados de las distancias de cada punto a su $\mu_k$ más cercano. Esta función de costo (o inercia) se modela como:

$$ J(C, M) = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2 $$

**Optimización de Centroides**
Como los clusters $C$ dependen de $\mu_k$, el objetivo es minimizar $J$ en función de $\mu_k$. Fijada una asignación de puntos al grupo $C_k$, buscamos el vector $\mu_k$ que minimiza su contribución a la función $J$:

$$ \frac{\partial}{\partial \mu_k} \sum_{x \in C_k} ||x - \mu_k||^2 = 0 $$

Calculando la derivada respecto al vector $\mu_k$:

$$ -2 \sum_{x \in C_k} (x - \mu_k) = 0 $$
$$ \sum_{x \in C_k} x - \sum_{x \in C_k} \mu_k = 0 $$
$$ \sum_{x \in C_k} x - \mu_k |C_k| = 0 $$

Despejando el centroide $\mu_k$:

$$ \mu_k = \frac{1}{|C_k|} \sum_{x \in C_k} x $$

**Conclusión**
La demostración prueba que el punto que minimiza la función de costo es exactamente la **media** (el centro de masa) de los puntos del grupo. 

*Nota:* Debido a la dependencia de la norma euclidiana al cuadrado en la función $J$, el algoritmo K-means asume intrínsecamente que los grupos a encontrar son compactos y esféricos.

### Breve analisis del codigo
Como se vió en el argumento matematico, el centroide que siempre va a ir minimizando las distancias es el centro de masa y es por eso que en el algoritmo está ese paso. 


## Segmentacíon de imagenes usando K means 
Una aplicación de kmeans (clustering) es que podemos hacer segmentación de imagenes. 
La idea es que a una imagen le encontramos los $K$ colores (clusteres) que más dominan en la imagen y luego remplazamos los pixeles originales por el centroide representante del cluster en el que están.
Con esto nuestra imagen se reduce a solo K colores. 

* Implementación:
![imagen1](/implementaciones/imagenes/Figure_1.png)
