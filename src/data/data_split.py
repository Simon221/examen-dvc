import pandas as pd
from sklearn.model_selection import train_test_split
import os



# Read file
def read_data(file_path):
    raw_data = pd.read_csv(file_path)
    return raw_data


# Split data into training and testing sets
def split_data(df):
    target = df['silica_concentrate']
    data = df.drop(['silica_concentrate', 'date'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Save dataframes to their respective output file paths
def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)
            
def main():
    # read the file
    input_file_path = "./data/raw_data/raw.csv"
    output_folder = "./data/processed_data"
    raw_data = read_data(input_file_path)
    
    # split the file raw data
    X_train, X_test, y_train, y_test = split_data(raw_data)
    
    # save the splitted file to the output folder
    save_dataframes(X_train, X_test, y_train, y_test, output_folder)


if __name__ == '__main__':
    
    main()