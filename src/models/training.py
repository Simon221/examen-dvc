import sklearn
import pandas as pd 
from sklearn import ensemble
import joblib
import numpy as np


# Read file
def read_data(file_path):
    raw_data = pd.read_csv(file_path)
    return raw_data


def train(X_train_scaled, y_train, output_folder):

    y_train = y_train.values.ravel()
    
    # recupération des best_params    
    best_params = joblib.load("models/best_params.pkl")
    # print(best_params)
    
    # Créer le modèle avec les meilleurs paramètres trouvés
    my_model = ensemble.GradientBoostingRegressor(**best_params)

    # Entraîner le modèle sur l'ensemble d'entraînement
    my_model.fit(X_train_scaled, y_train)

    # Sauvegarder le modèle entraîné
    joblib.dump(my_model, output_folder + "/gbr_model.pkl")
        
    print("Model trained and saved successfully.")


def main():
    # read the file
    X_train_scaled_file_path = "./data/processed_data/X_train_scaled.csv"
    y_train_file_path = "./data/processed_data/y_train.csv"
    output_folder = "./models"
    X_train_scaled = read_data(X_train_scaled_file_path)
    y_train = read_data(y_train_file_path)
    
    train(X_train_scaled, y_train, output_folder)
    

if __name__ == '__main__':
    
    main()