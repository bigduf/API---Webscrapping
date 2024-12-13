from fastapi import HTTPException
from fastapi.responses import JSONResponse
import os
import joblib
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from google.cloud import firestore


DATA_DIR = "src/data"
MODEL_DIR = "src/models"
CONFIG_FILE = "src/config/model_parameters.json"

def load_iris_dataset():
    """
    Charger le dataset Iris en tant que DataFrame et le retourner sous forme de dictionnaire (format JSON).
    """
    iris_file_path = os.path.join(DATA_DIR, "iris/Iris.csv")
    
    # Vérifier si le fichier Iris.csv existe
    if not os.path.exists(iris_file_path):
        raise HTTPException(status_code=404, detail="Le fichier Iris.csv est introuvable.")
    
    # Charger le dataset avec pandas
    try:
        iris_df = pd.read_csv(iris_file_path)
        # Convertir le DataFrame en dictionnaire (format JSON)
        iris_data = iris_df.to_dict(orient="records")  # Retourne une liste de dictionnaires
        return JSONResponse(content=iris_data)  # Retourne directement un JSONResponse
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du dataset : {e}")


def process_iris_dataset():
    """
    Load and preprocess the Iris dataset:
    - Load data
    - Remove missing values
    - Separate the features (X) and the target (y)
    - Normalize data
    """
    iris_file_path = os.path.join(DATA_DIR, "iris/Iris.csv")

    if not os.path.exists(iris_file_path):
        raise HTTPException(status_code=404, detail="Le fichier iris.csv est introuvable.")

    try:
        # Charger le dataset avec pandas
        iris_df = pd.read_csv(iris_file_path)

        # Vérifier s'il y a des valeurs manquantes
        if iris_df.isnull().sum().any():
            iris_df = iris_df.dropna()  # Suppression des lignes avec valeurs manquantes

        # Séparer les features (X) et la cible (y)
        X = iris_df.drop(columns=["Id", "Species"])  # Supposons que 'species' est la colonne cible
        y = iris_df["Species"]

        # Normaliser les données (Standardisation)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Retourner les données sous forme de JSON
        processed_data = {
            "features": X_scaled.tolist(),
            "target": y.tolist()
        }

        return processed_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du dataset : {e}")
    
def split_iris_dataset():
    """
    Diviser le dataset Iris en ensembles d'entraînement et de test, et renvoyer les deux sous forme de JSON.
    """
    iris_file_path = os.path.join(DATA_DIR, "iris/Iris.csv")

    if not os.path.exists(iris_file_path):
        raise HTTPException(status_code=404, detail="Le fichier iris.csv est introuvable.")

    try:
        # Charger le dataset avec pandas
        iris_df = pd.read_csv(iris_file_path)

        # Vérifier s'il y a des valeurs manquantes
        if iris_df.isnull().sum().any():
            iris_df = iris_df.dropna()  # Suppression des lignes avec valeurs manquantes

        # Séparer les features (X) et la cible (y)
        X = iris_df.drop(columns=["Id", "Species"])  # Supposons que 'species' est la colonne cible
        y = iris_df["Species"]

        # Normaliser les données (Standardisation)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Séparer les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Convertir les ensembles en dictionnaires pour les envoyer en réponse JSON
        train_data = {
            "X_train": X_train.tolist(),
            "y_train": y_train.tolist()
        }

        test_data = {
            "X_test": X_test.tolist(),
            "y_test": y_test.tolist()
        }

        # Retourner les données sous forme de JSON (dictionnaire)
        return JSONResponse(content={"train_data": train_data, "test_data": test_data})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du dataset : {e}")


def train_model():
    iris_file_path = os.path.join(DATA_DIR, "iris/Iris.csv")
    
    # Vérifier si le fichier Iris.csv existe
    if not os.path.exists(iris_file_path):
        raise HTTPException(status_code=404, detail="Le fichier Iris.csv est introuvable.")
    
    try:
        # Charger le dataset
        iris_df = pd.read_csv(iris_file_path)

        # Préparer les données
        X = iris_df.drop(columns=["Id", "Species"])  # Features
        y = iris_df["Species"]  # Target

        # Diviser en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Charger les paramètres du modèle depuis le fichier JSON
        with open(CONFIG_FILE, "r") as file:
            config = json.load(file)

        model_name = config["model"]
        parameters = config["parameters"]

        # Créer et entraîner le modèle
        if model_name == "RandomForestClassifier":
            model = RandomForestClassifier(**parameters)
        else:
            raise HTTPException(status_code=400, detail="Modèle non pris en charge.")

        model.fit(X_train, y_train)

        # Évaluer le modèle
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Sauvegarder le modèle entraîné
        model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        joblib.dump(model, model_path)

        return {
            "message": "Modèle entraîné avec succès et sauvegardé.",
            "model_path": model_path,
            "accuracy": accuracy
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'entraînement du modèle : {e}")
    

def predict_with_model(input_data: list):
    print(f"Input data: {input_data}")  # Affiche les données d'entrée
    model_path = "src/models/RandomForestClassifier.joblib"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Le modèle entraîné est introuvable.")

    try:
        model = joblib.load(model_path)
        input_df = pd.DataFrame(input_data)
        print(f"DataFrame created: {input_df.head()}")  # Affiche un aperçu du DataFrame
        expected_features = model.n_features_in_
        print(f"Expected features: {expected_features}")  # Affiche le nombre de features attendu
        if input_df.shape[1] != expected_features:
            raise HTTPException(status_code=400, detail=f"Nombre de features attendu : {expected_features}, reçu : {input_df.shape[1]}")

        predictions = model.predict(input_df)
        (print(f"Predictions: {predictions}"))  # Affiche les prédictions
        return {"predictions": predictions.tolist()}

    except Exception as e:
        print(f"Error: {str(e)}")  # Affiche l'erreur dans la console pour diagnostic
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
    
def get_parameters():
    try:
        db = firestore.Client()
        doc_ref = db.collection("parameters").document("parameters")
        doc = doc_ref.get()

        if doc.exists:
            return doc.to_dict()
        else:
            raise HTTPException(status_code=404, detail="Parameters not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving parameters: {e}")

def update_parameters(data: dict):
    """
    Met à jour ou ajoute des paramètres dans la collection 'parameters' de Firestore.
    """
    try:
        db = firestore.Client()
        doc_ref = db.collection("parameters").document("parameters")
        doc_ref.set(data, merge=True)  # Fusionner avec les données existantes
        return {"message": "Paramètres mis à jour avec succès."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la mise à jour des paramètres : {e}")