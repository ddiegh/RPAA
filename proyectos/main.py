
import kagglehub
import pandas as pd
import os

def cargar_datos():
    print("Iniciando conexión con Kaggle...")
    # descarga
    path = kagglehub.dataset_download("khushikyad001/impact-of-screen-time-on-mental-health")

    #carga en pandas
    csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
    full_path = os.path.join(path, csv_file)
    df = pd.read_csv(full_path)
    
    return df

# Ejecución
if __name__ == "__main__":
    df = cargar_datos()
    prueba = df.iloc[20]
    #data del primer usuario 
    for i in range(len(prueba)):
        print(f' {df.columns[i]}: {prueba.iloc[i]}') 
    
    #info del data set 
    #print(df.info())
