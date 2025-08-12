import pandas as pd
import sys

def process_csv_file(csv_file_path):
    # Read the dataset from the CSV file
    data = pd.read_csv(csv_file_path)
    
    # Sort the data by pAct in descending order and select the top 20 rows
    top_20_smiles = data.sort_values(by='pAct', ascending=False).head(20)['SMILES'].tolist()

    def format_smiles(smiles_list):
        formatted_smiles = []
        for smile in smiles_list:
            formatted_smiles.append('"' + smile + '",')
        return formatted_smiles

    # Format selected SMILES
    formatted_smiles = format_smiles(top_20_smiles)

    # Print the formatted SMILES
    for smile in formatted_smiles:
        print(smile)
        
if __name__ == "__main__":
    # Extract the CSV file path from the command-line arguments
    csv_file_path = sys.argv[1]

    # Call the function to process the CSV file
    process_csv_file(csv_file_path)
