from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

# Read file
def read_data(file_path):
    raw_data = pd.read_csv(file_path)
    return raw_data

# Save dataframes to their respective output file paths
def save_dataframes(X_train_scaled, X_test_scaled, output_folderpath):
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)
               
def normalize_data(X_train, X_test):
    # Normalisation avec StandardScaler (moyenne 0, variance 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

       
def main():
    # read the file
    xtrain_file_path = "./data/processed_data/X_train.csv"
    xtest_file_path = "./data/processed_data/X_test.csv"
    output_folder = "./data/processed_data"
    X_train = read_data(xtrain_file_path)
    X_test = read_data(xtest_file_path)
    
    # split the file raw data    
    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
    
    # Affichage des nouvelles statistiques après normalisation
    # X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    # print("Statistiques avant normalisation :\n", X_train.describe())
    # print("*********************************************************")
    # print("Statistiques après normalisation :\n", X_train_scaled_df.describe())
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # save the splitted file to the output folder
    save_dataframes(X_train_scaled_df, X_test_scaled_df, output_folder)


if __name__ == '__main__':
    
    main()

