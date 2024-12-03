import os
import json
import pandas as pd
import kaggle
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from kaggle.api.kaggle_api_extended import KaggleApi
import opendatasets as od
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

router = APIRouter()

# Répertoire pour stocker les datasets
DATA_DIR = "src/data"

# S'assurer que le répertoire existe
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Lire le fichier de configuration pour obtenir les URLs des datasets
def load_datasets_config():
    config_path = "src/config/config.json"
    try:
        with open(config_path, "r") as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Le fichier de configuration est introuvable.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Erreur de lecture du fichier JSON.")

# Télécharger le dataset à partir de Kaggle
def download_kaggle_dataset(dataset_url: str, destination: str):
    try:
        api = KaggleApi()
        api.authenticate()  # Authentifier l'API
        dataset_name = dataset_url.split('/')[-1]  # Extraire le nom du dataset à partir de l'URL
        #api.dataset_download_files(dataset_name, path=destination, unzip=True)
        od.download(dataset_url, destination, force=True)
        return {"message": f"Dataset {dataset_name} téléchargé avec succès dans {destination}."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors du téléchargement : {e}")

@router.get("/download-dataset/{dataset_key}")
def get_dataset(dataset_key: str):
    """
    Télécharger un dataset spécifié dans le fichier de configuration.
    """
    datasets_config = load_datasets_config()

    # Vérifier si la clé du dataset existe dans le fichier de config
    dataset_info = datasets_config.get(dataset_key)
    
    if not dataset_info:
        raise HTTPException(status_code=404, detail="Dataset non trouvé dans la configuration.")
    
    dataset_url = dataset_info.get("url")
    
    if not dataset_url:
        raise HTTPException(status_code=400, detail="URL du dataset manquante dans la configuration.")
    
    # Appeler la fonction pour télécharger le dataset
    return download_kaggle_dataset(dataset_url, DATA_DIR)

@router.get("/load-iris-dataset")
def load_iris_dataset():
    """
    Charger le dataset Iris en tant que DataFrame et le retourner en JSON.
    """
    iris_file_path = os.path.join(DATA_DIR, "iris/Iris.csv")
    
    # Vérifier si le fichier Iris.csv existe
    if not os.path.exists(iris_file_path):
        raise HTTPException(status_code=404, detail="Le fichier Iris.csv est introuvable.")
    
    # Charger le dataset avec pandas
    try:
        iris_df = pd.read_csv(iris_file_path)
        # Convertir le DataFrame en JSON
        return JSONResponse(content=iris_df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du dataset : {e}")
    

@router.get("/process-iris-dataset")
def process_iris_dataset():
    iris_file_path = os.path.join(DATA_DIR, "iris/Iris.csv")

    if not os.path.exists(iris_file_path):
        raise HTTPException(status_code=404, detail="Le fichier iris.csv est introuvable.")

    try:
        # Charger le dataset avec pandas
        iris_df = pd.read_csv(iris_file_path)

        # Vérifier s'il y a des valeurs manquantes
        if iris_df.isnull().sum().any():
            iris_df = iris_df.dropna()  # Suppression des lignes avec valeurs manquantes
            # ou bien, vous pouvez utiliser iris_df.fillna() pour imputer les valeurs manquantes

        # Séparer les features (X) et la cible (y)
        X = iris_df.drop(columns=["Species"])  # Supposons que 'species' est la colonne cible
        y = iris_df["Species"]

        # Normaliser les données (Standardisation)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Retourner les données sous forme de JSON
        processed_data = {
            "features": X_scaled.tolist(),
            "target": y.tolist()
        }

        return JSONResponse(content=processed_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du dataset : {e}")