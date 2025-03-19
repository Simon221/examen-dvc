import sklearn
import json
import pandas as pd 
from sklearn.metrics import mean_squared_error, r2_score
import joblib

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature



# Read file
def read_data(file_path):
    raw_data = pd.read_csv(file_path)
    return raw_data


def evaluate(my_model, X_test_scaled, y_test ,output_folder, output_metrics) : 
    
    with mlflow.start_run() as run:

    
        prediction = my_model.predict(X_test_scaled)
        
        # save prediction to prediction.csv and track it with DVC
        test = pd.DataFrame(prediction, columns=["Prediction"]).to_csv(output_folder + "/prediction.csv", index=False)

        # Evaluation
            # MSE (Mean Squared Error) : Erreur quadratique moyenne => plus c'est bas, mieux c'est.
            # RMSE (Root Mean Squared Error) : Racine carrée du MSE => permet d'interpréter directement l'erreur en unités de la variable cible.
            # R² (Coefficient de détermination) : Score entre 0 et 1 qui indique la proportion de la variance expliquée par le modèle (1 étant parfait).
        
        mse = mean_squared_error(y_test, prediction)
        rmse = mean_squared_error(y_test, prediction, squared=False)  # RMSE = sqrt(MSE)
        r2 = r2_score(y_test, prediction)

        # Stocker les résultats dans un dictionnaire
        metrics = {
            "Mean_Squared_Error": mse,
            "Root_Mean_Square_Error": rmse,
            "R_carre_Score": r2
        }
        
        print (metrics)

        # Sauvegarder les métriques dans un fichier JSON
        with open(output_metrics + "/scores.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        
        # Log parameters and metrics using the MLflow APIs
        # mlflow.set_tracking_uri("http://127.0.0.1:8080")
        # signature = infer_signature(X_test_scaled, prediction)
        # best_params = joblib.load("models/best_params.pkl")
        # mlflow.log_params(best_params)
        # mlflow.log_metrics(metrics)

        # Log the sklearn model and register as version 1
        mlflow.sklearn.log_model(
            sk_model=my_model,
            artifact_path="silica_concentrate",
            signature=signature,
            registered_model_name="silica_concentrate_model_V1",
        )



def main():
    # read the file
    X_test_scaled_file_path = "./data/processed_data/X_test_scaled.csv"
    y_test_file_path = "./data/processed_data/y_test.csv"
    output_folder = "./data"
    output_metrics = "./metrics"
    X_test_scaled = read_data(X_test_scaled_file_path)
    y_test = read_data(y_test_file_path)
    
    # get the model
    my_model = joblib.load("models/gbr_model.pkl")

    # evaluate the model
    evaluate(my_model, X_test_scaled, y_test, output_folder, output_metrics)
    

if __name__ == '__main__':
    
    main()