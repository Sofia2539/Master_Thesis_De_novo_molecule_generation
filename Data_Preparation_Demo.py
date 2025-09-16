'''REINVENT 3.2: Data Preparation demo
This demo is an updated version of the REINVENT notebook. It illustrates how 
data from ChEMBL are processed, analysed and filtered, using pandas DataFrame.
'''

'''Pre-process the data used for training a generative model:
    1. Removal of invalid or duplicated entries.
    2. Removal of unusual compounds that are clearly not drug-like (too big, reactive groups and etc.). 
    Normally, no point in training a model on such examples since that bias will reflected by the generative model.
    3. Removal of rare tokens. 
    There are rare compounds that can be seen as outliers and they might contain rare tokens. 
    Excluding them frees a slot in the vocabulary and makes it smaller. Smaller vocabulary means 
    faster training and less memory. As a result removing compounds that introduce rare tokens to 
    the vocabulary speeds up the generative model.
'''
# Load dependencies
import pandas as pd
import numpy as np
import molvs as mv
import rdkit.Chem as rkc
import rdkit.Chem.AllChem as rkac
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Define paths
DBS_PATH = "/home/sofia/Downloads/Datasets/PIM1 Kinase/PIM1_dataset_SMILES.csv"
output_dir = "/home/sofia/ReinventCommunity/Data_Preparation_demo"
filtered_csv_file = f'{output_dir}/final.filtered_PIM1_dataset.csv'

# Load data from SMILES file into pandas DataFrame
chembl_df = pd.read_csv(DBS_PATH)
# Remove duplicate entries
chembl_df.drop_duplicates(inplace=True)

def to_mol(smi):
    """
    Create Mol object from a SMILES string.
    """
    try:
        if smi:
            return rkc.MolFromSmiles(smi)
    except Exception as e:
        print(f"Error converting SMILES to Mol: {e}")
    return None

def to_smiles(mol):
    """
    Converts a Mol object into a canonical SMILES string.
    """
    try:
        if mol is not None:
            return rkc.MolToSmiles(mol, isomericSmiles=False)
    except Exception as e:
        print(f"Error converting Mol to SMILES: {e}")
    return None

def _run_reaction(mol, rxn):
    """
    Apply a reaction to molecule.
    """
    while True:
        results = rxn.RunReactants([mol], maxProducts=1)
        if not results:
            return mol
        else:
            mol = results[0][0]

REACTIONS = [
    "[S+:1](=[N:3])[OH:2]>>[S+0:1](=[N:3])=[O:2]",
    "[n+:1][OH:2]>>[n+:1][O-]",
    "[N:1](=[O:2])=[O:3]>>[N+:1]([O-:2])=[O:3]",
    "[S+:1]([O:2])[N:3]>>[S+0:1](=[O:2])[N:3]"
]

REACTIONS = [rkac.ReactionFromSmarts(rxn) for rxn in REACTIONS] 

# Standardize molecule
STANDARDIZER = mv.Standardizer()
ACCEPTED_ATOMS = [6, 7, 8, 9, 16, 17, 35]

def standardize_mol(mol, standardize=True, min_size=0, max_size=1000):
    """
    Standardize a molecule.
    """
    try:
        if standardize:
            for rxn in REACTIONS:
                mol = _run_reaction(mol, rxn)
            
            mol = STANDARDIZER.charge_parent(mol, skip_standardize=True)
            mol = STANDARDIZER.isotope_parent(mol, skip_standardize=True)
            mol = STANDARDIZER.stereo_parent(mol, skip_standardize=True)
            mol = STANDARDIZER.standardize(mol)
            if any([atom.GetAtomicNum() not in ACCEPTED_ATOMS for atom in mol.GetAtoms()]):
                return None
        return mol
    except Exception as e:
        print(f"Error standardizing molecule: {e}")
        return None
    
class SMILESTokenizer:
    """
    Tokenize SMILES strings.
    """
    def __init__(self):
        self.pattern = re.compile(r'\[.*?\]|\(|\)|\||\\|/|=|\.|%[0-9]{2}|[a-z]|[A-Z][a-z]?|[0-9]')

    def tokenize(self, smiles):
        return self.pattern.findall(smiles)

tokenizer = SMILESTokenizer()

def tokenize_smiles(smi):
    """
    Tokenize SMILES string.
    """
    return tokenizer.tokenize(smi)

SMARTS_CHAINS = [rkc.MolFromSmarts("-".join(["[CR0H2]"]*i)) for i in range(1, 11)]

def longest_aliphatic_c_chain(smi):
    """
    Calculate the longest aliphatic carbon chain.
    """
    mol = to_mol(smi)
    curr_chain = 0
    for chain in SMARTS_CHAINS:
        if mol.HasSubstructMatch(chain):
            curr_chain += 1
        else:
            break
    return curr_chain

def num_rings(smi):
    """
    Calculate the number of rings in a molecule.
    """
    mol = to_mol(smi)
    if mol:
        return rkc.GetSSSR(mol)
    return None

def size_largest_ring(smi):
    """
    Calculate the size of the largest ring in a molecule.
    """
    mol = to_mol(smi)
    if mol:
        ring_info = mol.GetRingInfo()
        return max([0] + [len(ring) for ring in ring_info.AtomRings()])
    return None

# Define a function to process each row
def process_rows(row):
    """
    Process each row of the DataFrame.
    """
    try:
        fields = row.split(" ")
        mol = to_mol(fields[0])
        standardized_smiles = None
        if mol:
            standardized_mol = standardize_mol(mol)
            standardized_smiles = to_smiles(standardized_mol)
        return pd.Series({'original_smiles': fields[0], 'smiles': standardized_smiles})
    except Exception as e:
        print(f"Error processing row: {e}")
        return pd.Series({'original_smiles': None, 'smiles': None})
    
# Define a function to process each row
def process_rows(row):
    fields = row.split(" ")
    mol = to_mol(fields[0])
    standardized_smiles = None
    if mol:
        standardized_mol = standardize_mol(mol)
        standardized_smiles = to_smiles(standardized_mol)
    return pd.Series({'original_smiles': fields[0], 'smiles': standardized_smiles})

# Apply the process_rows function to each row of the DataFrame
chembl_df[['original_smiles', 'smiles']] = chembl_df['SMILES'].apply(process_rows)

# Drop rows where 'smiles' is null
chembl_df = chembl_df.dropna(subset=['smiles'])

# Remove duplicates based on 'smiles' column
chembl_df = chembl_df.drop_duplicates(subset=['smiles'])

# Apply standardization to molecules
chembl_df['mol'] = chembl_df['SMILES'].apply(to_mol)
chembl_df['standardized_mol'] = chembl_df['mol'].apply(standardize_mol)
chembl_df.head()

# Apply functions to create new columns in the DataFrame
chembl_df['num_atoms'] = chembl_df['SMILES'].apply(lambda x: to_mol(x).GetNumHeavyAtoms())
chembl_df['c_atom_ratio'] = chembl_df.apply(lambda row: sum(1 for atom in row['mol'].GetAtoms() if atom.GetSymbol() == 'C') / row['num_atoms'], axis=1)
chembl_df['tokens'] = chembl_df['SMILES'].apply(tokenize_smiles)
# Calculate the number of tokens
chembl_df['num_tokens'] = chembl_df['tokens'].apply(len)
chembl_df['num_rings'] = chembl_df['SMILES'].apply(lambda x: num_rings(x))
chembl_df['size_largest_ring'] = chembl_df['SMILES'].apply(lambda x: size_largest_ring(x))
chembl_df['tokens_atom_ratio'] = chembl_df['num_tokens'] / chembl_df['num_atoms']
chembl_df['longest_aliph_c_chain'] = chembl_df['SMILES'].apply(longest_aliphatic_c_chain)

#Data Purging

# Num atoms distribution
import matplotlib.pyplot as plt

# Step 1: Group by 'num_atoms' and aggregate counts
num_atoms_dist = chembl_df.groupby('num_atoms').size().reset_index(name='num')

# Step 2: Calculate percentage
num_atoms_dist['percent'] = num_atoms_dist['num'] * 100 / len(chembl_df)

# Step 3: Sort by 'num_atoms'
num_atoms_dist = num_atoms_dist.sort_values(by='num_atoms', ascending=False)

# Step 4: Plot
num_atoms_dist.plot(x='num_atoms', y='percent', xlim=(0, 100), lw=3)
plt.show()

# Step 5: Filter based on condition
chembl_filtered_df = chembl_df[(chembl_df['num_atoms'] >= 6) & (chembl_df['num_atoms'] <= 70)]
print(chembl_filtered_df.count())

#Number of rings

# Calculate the number of rings for each row
chembl_filtered_df['num_rings'] = chembl_filtered_df['num_rings'].apply(len)

# Group by 'num_rings' and aggregate counts
num_rings_dist = chembl_filtered_df.groupby('num_rings')['num_atoms'].count().reset_index(name='num')

# Calculate percentage
num_rings_dist['percent'] = num_rings_dist['num'] * 100 / len(chembl_filtered_df)

# Sort by 'num_rings'
num_rings_dist = num_rings_dist.sort_values(by='num_rings', ascending=False)

# Plot
num_rings_dist.plot(x='num_rings', y='percent', lw=3, xticks=num_rings_dist['num_rings'])
plt.show()

# Filter based on condition
chembl_filtered_df = chembl_filtered_df[chembl_filtered_df['num_rings'] <= 10]

# Count the number of rows in the filtered DataFrame
print(chembl_filtered_df.count())
chembl_filtered_df['num_rings']

# Size of largest ring
# Group by 'size_largest_ring' and aggregate counts
size_largest_ring_dist = chembl_filtered_df.groupby('size_largest_ring')['num_atoms'].count().reset_index(name='num')

# Calculate percentage
size_largest_ring_dist['percent'] = size_largest_ring_dist['num'] * 100 / len(chembl_filtered_df)

# Sort by 'size_largest_ring'
size_largest_ring_dist = size_largest_ring_dist.sort_values(by='size_largest_ring', ascending=False)

# Plot
size_largest_ring_dist.plot(x='size_largest_ring', y='percent', lw=3)
plt.show()

# Filter based on condition
chembl_filtered_df = chembl_filtered_df[chembl_filtered_df['size_largest_ring'] < 9]
# Count the number of rows in the filtered DataFrame
print(chembl_filtered_df.count())

# Long aliphatic C chains
# Group by 'longest_aliph_c_chain' and aggregate counts
longest_aliph_c_chain = chembl_filtered_df.groupby('longest_aliph_c_chain')['num_atoms'].count().reset_index(name='num')

# Calculate percentage
longest_aliph_c_chain['percent'] = longest_aliph_c_chain['num'] * 100 / len(chembl_filtered_df)

# Sort by 'longest_aliph_c_chain'
longest_aliph_c_chain = longest_aliph_c_chain.sort_values(by='longest_aliph_c_chain', ascending=False)

# Plot
longest_aliph_c_chain.plot(x='longest_aliph_c_chain', y='percent', lw=3)
plt.show()

# Filter based on condition
chembl_filtered_df = chembl_filtered_df[chembl_filtered_df['longest_aliph_c_chain'] < 5]

# Count the number of rows in the filtered DataFrame
print(chembl_filtered_df.count())

# Heteroatom ratios
# Sample 10% of the DataFrame
c_ratio_dist = chembl_filtered_df.sample(frac=0.1)

# Plot histogram
c_ratio_dist.hist(column="c_atom_ratio", bins=32)
plt.show()
# Filter based on condition
chembl_filtered_df = chembl_filtered_df[chembl_filtered_df['c_atom_ratio'] >= 0.5]

# Count the number of rows in the filtered DataFrame
print(chembl_filtered_df.count())

# Number of tokens
# Group by 'num_tokens' and aggregate counts
num_tokens_dist = chembl_filtered_df.groupby("num_tokens").size().reset_index(name="num")

# Calculate percentage
num_tokens_dist['percent'] = num_tokens_dist['num'] * 100 / len(chembl_filtered_df)

# Sort the DataFrame by 'num_tokens'
num_tokens_dist = num_tokens_dist.sort_values(by="num_tokens", ascending=False)

# Plot the data
num_tokens_dist.plot(x="num_tokens", y="percent", lw=3)
plt.show()

# Filter based on condition
chembl_filtered_df = chembl_filtered_df[chembl_filtered_df['num_tokens'] <= 91]

# Count the number of rows in the filtered DataFrame
print(chembl_filtered_df.count())

# Tokens/atom ratio
# Sample the DataFrame and convert to Pandas DataFrame
tokens_atom_ratio_dist = chembl_df.sample(frac=0.1, replace=False, random_state=42)

# Plot histogram
tokens_atom_ratio_dist.hist(column="tokens_atom_ratio", bins=32)
plt.show()

# Filter based on condition
chembl_filtered_df = chembl_filtered_df[chembl_filtered_df['tokens_atom_ratio'] <= 2.0]

# Count the number of rows in the filtered DataFrame
print(chembl_filtered_df.count())

# Token/molecule distribution
# Calculate unique tokens and their counts
token_dist = chembl_filtered_df['tokens'].explode().value_counts().reset_index()
token_dist.columns = ['token', 'num']

# Calculate percentage
token_dist['percent'] = 100.0 * token_dist['num'] / len(chembl_filtered_df)

# Sort by percentage in descending order
token_dist = token_dist.sort_values(by='percent', ascending=False)

# Display the DataFrame
print(token_dist)

# Define tokens to remove based on conditions
tokens_to_remove = token_dist[(token_dist["percent"] < 5E-2) & (token_dist["token"].str.startswith("[")) & ~(token_dist["token"].isin(["[S+]", "[s+]"]))]["token"]

# Generate query for tokens to remove
query_tokens = False
for token in tokens_to_remove:
    query_tokens |= chembl_filtered_df['tokens'].apply(lambda x: token in x)
    
# Initialize an empty Series to hold the query condition
query_tokens = pd.Series([False] * len(chembl_filtered_df))

# Iterate over tokens_to_remove to construct the query condition
for token in tokens_to_remove:
    query_tokens |= chembl_filtered_df["tokens"].apply(lambda x: token in x)
# Ensure that the Boolean Series and DataFrame have the same index
query_tokens.index = chembl_filtered_df.index

# Filter DataFrame to remove tokens
chembl_filtered_df = chembl_filtered_df[~query_tokens].reset_index(drop=True)[["original_smiles", "smiles"]]

# Count the number of rows in the filtered DataFrame
print(chembl_filtered_df.count())

chembl_filtered_df.head()

# Write the filtered dataset to disk
'''All the SMILES that meet the filtering criteria are 
written out to a csv file.'''
filtered_csv_file = f'{output_dir}/final.filtered_PIM1_dataset.csv'
chembl_filtered_df[['smiles']].to_csv(filtered_csv_file, index=False, header=False)
