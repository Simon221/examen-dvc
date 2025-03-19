from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import f1_score
from sklearn import ensemble
import pandas as pd
import yaml
import joblib


# Read file
def read_data(file_path):
    raw_data = pd.read_csv(file_path)
    return raw_data


# Search the best param
def applyGridSearch(X_train_scaled, y_train, output_folder):

    clf = ensemble.GradientBoostingRegressor()
    y_train = y_train.values.ravel()

    # Définir la grille des hyperparamètres à explorer
        # Charger les paramètres depuis le fichier YAML
    with open("./src/data/params.yaml", "r") as file:
        params = yaml.safe_load(file)

    # Extraire la grille des hyperparamètres
    param_grid = params["gradient_boosting"]
    
    # Créer l'objet GridSearchCV
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

    # Effectuer la recherche sur la grille
    grid_search.fit(X_train_scaled, y_train)

    # Afficher les meilleurs paramètres et score
    print("Meilleurs paramètres:", grid_search.best_params_)
    print("Meilleur score (MSE):", -grid_search.best_score_)  # On inverse car c’est un score négatif
    
    # Sauvegarde du model
    joblib.dump(grid_search.best_params_, output_folder + "/best_params.pkl")



def main():
    # read the file
    X_train_scaled_file_path = "./data/processed_data/X_train_scaled.csv"
    y_train_file_path = "./data/processed_data/y_train.csv"
    output_folder = "./models"
    X_train_scaled = read_data(X_train_scaled_file_path)
    y_train = read_data(y_train_file_path)
    
    applyGridSearch(X_train_scaled, y_train, output_folder)
    


if __name__ == '__main__':
    
    main()