import os
import json
import pandas as pd
import kaggle
from fastapi import APIRouter, HTTPException
from kaggle.api.kaggle_api_extended import KaggleApi
import opendatasets as od
from sklearn.model_selection import train_test_split
from src.services.data import load_iris_dataset, process_iris_dataset, split_iris_dataset, train_model, predict_with_model

from pydantic import BaseModel
from typing import List

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
def load_iris_dataset_endpoint():
    """
    Charger le dataset Iris en tant que DataFrame et le retourner en JSON.
    """
    return load_iris_dataset()

@router.get("/process-iris-dataset")
def process_iris_dataset_endpoint():
    """
    Traiter le dataset Iris et le retourner après avoir effectué des prétraitements.
    """
    load_iris_dataset()
    return process_iris_dataset()

@router.get("/split-iris-dataset")
def train_test_split_iris():
    """
    Charger le dataset Iris, le traiter et le diviser en données d'entraînement et de test.
    """
    return split_iris_dataset()

@router.get("/train-model")
def train_model_endpoint():
    """
    Charger le dataset Iris, le traiter, diviser les données et entraîner un modèle de classification.
    """
    return train_model()

class PredictionRequest(BaseModel):
    data: List[List[float]]  # List of feature vectors for prediction

@router.post("/predict")
async def predict(request: PredictionRequest):
    """
    Endpoint to make predictions using the trained model.
    """
    return predict_with_model(request.data)