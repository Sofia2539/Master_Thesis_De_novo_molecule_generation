import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
import sys
import os

# Function to calculate QED score from a SMILES string
def calculate_qed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return QED.qed(mol)
    else:
        return None

def main(input_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Read SMILES colunm
    smiles_column = 'SMILES'

    # Calculate QED scores
    df['QED'] = df[smiles_column].apply(calculate_qed)

    # Extract filename from input CSV path
    filename = os.path.splitext(os.path.basename(input_csv))[0]

    # Define output file path
    output_file = os.path.join(os.getcwd(), f'{filename}_with_qed_scores.csv')

    # Save the results to the output CSV file
    df.to_csv(output_file, index=False)

    print(f'QED scores calculated and saved to {output_file}')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_qed.py <input_csv_path>")
    else:
        input_csv = sys.argv[1]
        main(input_csv)
