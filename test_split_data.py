# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

def split_and_save_dataset(csv_file_path):
    # Read the dataset from CSV file
    df_csv = pd.read_csv(csv_file_path)

    # SMILES and Binary_Activity as column names
    X_csv = df_csv.iloc[:, :-1]
    y_csv = df_csv.iloc[:, -1]

    # Drop rows with NaN values
    df_csv_no_missing = df_csv.dropna()

    # Split the dataset into training and test sets
    '''Usage of stratify split:ensure that the proportion of the different categories (0 & 1) 
    in the target variable is the same in both the training set and the test set 
    as it is in the original dataset'''
    X_train_csv, X_test_csv, y_train_csv, y_test_csv = train_test_split(
        df_csv_no_missing.iloc[:, :-1], df_csv_no_missing.iloc[:, -1],
        test_size=0.1, random_state=0, stratify=df_csv_no_missing.iloc[:, -1]) #Reproducibility with random_state

    # Combine features and labels into DataFrames for train and test sets
    train_set_csv = pd.DataFrame({'SMILES': X_train_csv['SMILES'], 'Binary_Activity': y_train_csv})
    test_set_csv = pd.DataFrame({'SMILES': X_test_csv['SMILES'], 'Binary_Activity': y_test_csv})

    # Get the directory of the CSV file
    csv_directory = os.path.dirname(csv_file_path)

    # Save the DataFrames to CSV files in the same directory as the input CSV file
    train_set_csv.to_csv(os.path.join(csv_directory, 'train_set.csv'), index=False)
    test_set_csv.to_csv(os.path.join(csv_directory, 'test_set.csv'), index=False)

if __name__ == "__main__":
    # Check if the CSV file path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script_name.py /path/to/kinase_data.csv")
        sys.exit(1)

    # Extract the CSV file path from the command-line arguments
    csv_file_path = sys.argv[1]

    # Call the function to split and save the dataset
    split_and_save_dataset(csv_file_path)
