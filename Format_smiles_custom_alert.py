import pandas as pd
import sys
import random

def process_csv_file(csv_file_path):
    # Read the dataset from the CSV file
    data = pd.read_csv(csv_file_path)

    def format_smiles(smiles_list):
        formatted_smiles = []
        for smile in smiles_list:
            formatted_smiles.append('"' + smile + '",')
        return formatted_smiles
    
    # Filter the data for SMILES with activity 5 and less and select 10 of them
    selected_smiles = data[data['pAct'] <= 5]['SMILES'].tolist()
    
    # Choose 10 arbitrary SMILES from the selected list
    selected_smiles = random.sample(selected_smiles, k=10)

    # Format and print the formatted SMILES    
    print("Arbitrarily chosen SMILES with pAct <= 5:")
    formatted_selected_smiles = format_smiles(selected_smiles)
    for smile in formatted_selected_smiles:
        print(smile)   
        
if __name__ == "__main__":
    # Extract the CSV file path from the command-line arguments
    csv_file_path = sys.argv[1]

    # Call the function to process the CSV file
    process_csv_file(csv_file_path)
