import numpy as np

class KNN:
    def __init__(self, k: int):
        self.k = k
        self.X = None
        self.y = None
        self.clases = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = y
        self.clases = np.unique(y)
        return self

    def predecir(self, x):
        #distancia a todos los puntos
        distancias = np.linalg.norm(self.X - x, axis = 1)

        # k vecinos mas cercanos 
        k_vecinos = np.argpartition(distancias, self.k)[:self.k]

        #etiqueta de los vecinos y contar cuantas veces se repite cada clase
        id_vecinos = self.y[k_vecinos]
        valores, contador = np.unique(id_vecinos, return_counts=True)

        #asignamos la clase que se repite más
        asignacion = np.argmax(contador)
        return asignacion
    

class NearestCentroid:
    def __init__(self):
        self.X = None
        self.y = None
        self.clases = None
        self.centroides = None


    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

        N = len(self.y) #total de datos
        clases, contadores = np.unique(self.y, return_counts=True)  #cuantos elementos tiene cada clase
        self.clases = clases
        

        #encontramos los puntos en cada clase y obtenemos la media (centroides)
        centroides = np.zeros((len(clases), 2))
        for i, id in enumerate(np.unique(y)):
            mascara = y == id
            centroides[i][0], centroides[i][1] = np.mean(self.X[mascara], axis = 0)

        self.centroides = centroides
        return self
    
    def predecir(self, x):
        #calculamos la distancia a todos los centroides y asignamos el minimo
        distancias = np.linalg.norm(self.centroides - x, axis=1)
        etiqueta = np.argmin(distancias) + 1

        return etiqueta

class LDA:
    def __init__(self):
        self.X = None
        self.y = None


    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

        N = len(self.y) #total de datos
        clases, contadores = np.unique(self.y, return_counts=True)  #cuantos elementos tiene cada clase
        
        PI = contadores / N

        #encontramos los puntos en cada clase y obtenemos la media (centroides)
        centroides = np.zeros((len(PI), 2))
        for i, id in enumerate(np.unique(y)):
            mascara = y == id
            centroides[i][0], centroides[i][1] = np.mean(self.X[mascara], axis = 0)

        

        return centroides




def prueba_NC():
    X_train = np.array([
    [1.0, 1.0], [2.0, 2.0], [1.0, 3.0],  # Clase 1
    [6.0, 2.0], [7.0, 1.0], [8.0, 3.0],  # Clase 2
    [3.0, 6.0], [4.0, 8.0], [5.0, 7.0]   # Clase 3
    ])

    # Etiquetas
    y_train = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])

    nc = NearestCentroid()
    nc.fit(X_train, y_train)
    print(nc.predecir([4.5, 6.5]))




def prueba_KNN():
    X_train = np.array([
        [1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0], 
        [8.0, 8.0], [8.0, 9.0], [9.0, 8.0], [9.0, 9.0]  
    ])

    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    x = [3.0, 3.0]

    knn = KNN(k = 3)
    knn.fit(X_train, y_train)
    print(knn.predecir(x))

def main():
    prueba_NC()
if __name__ == '__main__':
    main()
