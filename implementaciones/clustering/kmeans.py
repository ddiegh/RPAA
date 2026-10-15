import numpy as np 

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

class DBSCAN:
    def __init__(self, eps: float, minpts: int):
        self.eps = eps 
        self.minpts = minpts

    def fit(self, X):
        #calcular la distancia entre cada punto
        n = len(X)
        d = np.zeros((n, n))
        for i, p in enumerate(X):
            for j, q in enumerate(X):
                d[i][j] = np.linalg.norm(p - q)

        #calcular los puntos que estan en el epsilon-vecindario        
        puntos_eps = np.zeros(n)
        for h, v in enumerate(d):
            puntos = 0 
            for dis in v:
                if dis < self.eps:
                    puntos += 1 
            puntos_eps[h] = puntos 

        #definir si un punto es central, frontera o ruido 
        #-1 ruido, 0 frontera, 1 central 
        puntos = np.zeros(n) 
        for i, v in enumerate(puntos_eps):
            if v >= self.minpts: 
                puntos[i] = 1
            elif v < self.minpts: #verificar si es frontera de algun centro
                contador = 0
                for x in range(n):
                    if (d[i][x] < self.eps) and (i != x):
                        contador += 1
                if contador > 0:
                    puntos[i] = 0
                else:
                    puntos[i] = -1

        #crear clusteres 
        copia_puntos = np.copy(puntos)
        # indice del punto con su tipo de punto 
        ind_punto = [[u, v] for u, v in zip(range(n), copia_puntos)]

        Clusteres = [] #indices 

        centros = [v for v in ind_punto if v[1] == 1]
        fronteras = [v for v in ind_punto if v[1] == 0]
        
        while len(centros) != 0:
            a = np.random.choice(len(centros))
            centro_act = centros[a]

            #calcular cuantos centros estan cerca del centro actual 
            centro_cerca = []
            for centro in centros:
                if np.linalg.norm(X[centro[0]] - X[centro_act[0]]) <= self.eps:
                    centro_cerca.append(centro) 

            #metemos todos los puntos densos en el mismo cluster
            cluster = []
            while len(centro_cerca) > 0:
                centro = centro_cerca[0]
                cluster.append(centro[0])
                for x in fronteras:
                    if np.linalg.norm(X[x[0]]-X[centro[0]]) <= self.eps:
                        cluster.append(x[0])
                        fronteras.remove(x)
                centro_cerca.pop(0)
                centros.remove(centro)
            Clusteres.append(cluster)

        return Clusteres






def main():
    X = np.array([
    # --- GRUPO CONECTADO POR CADENA (Clúster 0) ---
    [0.0, 0.0],   # 0: Núcleo izquierdo
    [0.0, 0.5],   # 1: Vecino izquierdo superior
    [0.0, -0.5],  # 2: Vecino izquierdo inferior
    [0.9, 0.0],   # 3: Puente 1 (Solo alcanza al índice 0 y 4)
    [1.8, 0.0],   # 4: Puente 2 (Solo alcanza al índice 3 y 5)
    [2.7, 0.0],   # 5: Núcleo derecho
    [2.7, 0.5],   # 6: Vecino derecho superior
    [2.7, -0.5],  # 7: Vecino derecho inferior
    
    # --- GRUPO DENSO AISLADO (Clúster 1) ---
    [5.0, 5.0],   # 8
    [5.0, 5.5],   # 9
    [5.5, 5.0],   # 10
    [5.5, 5.5],   # 11
    
    # --- RUIDO ---
    [8.0, 8.0],   # 12
    [2.0, 3.0]    # 13
])

    modelo = DBSCAN(eps=1.0, minpts=3)
    print(modelo.fit(X))

if __name__ == '__main__':
    main()