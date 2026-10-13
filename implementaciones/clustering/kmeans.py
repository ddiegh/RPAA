import numpy as np 
import matplotlib.pyplot as plt


class Kmeans:
    def __init__(self, k:int, max_iter:int = 100, tol:int = 1e-6, seed = 42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        np.random.seed(seed)

        #atributos para despues de entrenar
        self.centroides = None
        self.labels_ = None
        self.inercia_ = None
        self.iteraciones_ = None

    def fit(self, X):
        """
        Entrena Kmeans
        """
        #pasar a array
        X_arr = np.array(X)
        n_muestras = len(X_arr)

        #inicializaciones
        indices = np.random.choice(n_muestras, self.k, replace=False)  #k indices aleatorios
        self.centroides = X_arr[indices].astype(float)

        self.labels_ = np.zeros(n_muestras)  #indices del cluster asignado
        self.inercia = 0
        num_iter = 0

        for num_iter in range(self.max_iter):
            c_viejos = np.copy(self.centroides) #para comparar en la norma

            #asignacion
            distancias = np.zeros((n_muestras, self.k))   #matriz de distancias
            for j in range(self.k):
                distancias[:, j] = np.linalg.norm(X_arr - self.centroides[j], axis=1)   #calcula la norma entre cada punto al centroide j
            
            self.labels_ = np.argmin(distancias, axis=1)  

            #self.centroides
            for j in range(self.k):
                mascara = self.labels_ == j  #solo los puntos en el cluster j
                puntos_cluster = X_arr[mascara]

                #asignamos el centroide
                if len(puntos_cluster) > 0:
                    self.centroides[j] = np.mean(puntos_cluster, axis=0)

                else:
                    nuevo_indice = np.random.choice(n_muestras)
                    self.centroides[j] = X_arr[nuevo_indice]

            #convergencia
            M = self.centroides - c_viejos
            error = np.sqrt(np.trace(M.T @ M))

            if error < self.tol:
                break

        #inercia 
        for j in range(self.k):
            mascara = self.labels_ == j
            puntos_cluster = X_arr[mascara]

            if len(puntos_cluster) > 0:
                self.inercia += np.sum((puntos_cluster - self.centroides[j])**2)

        self.iteraciones_ = num_iter + 1

        return self

