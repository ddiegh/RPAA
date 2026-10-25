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
    


def main():
    X_train = np.array([
        [1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0], 
        [8.0, 8.0], [8.0, 9.0], [9.0, 8.0], [9.0, 9.0]  
    ])

    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    x = [3.0, 3.0]

    knn = KNN(k = 3)
    knn.fit(X_train, y_train)
    print(knn.predecir(x))

if __name__ == '__main__':
    main()
