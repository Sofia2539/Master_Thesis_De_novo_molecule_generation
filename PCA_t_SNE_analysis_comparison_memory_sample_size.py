"""PCA for generated molecules with focus on PIM1 kinase: with different memory and sample size in parameters"""


#Load dependencies
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE


# Function to convert SMILES to fingerprints
def fp_list_from_smiles_list(smiles_list, n_bits=2048):
    fp_list = []
    removed_list = []
    index = 0
    for smiles in tqdm(smiles_list):
        index += 1
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:  # Check if mol is not None (successful conversion)
            fp_list.append(fp_as_array(mol, n_bits))
        else:
            removed_list.append(index)
    return fp_list, removed_list


def fp_as_array(mol, n_bits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((1,), dtype=np.int64)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# Read dataset with generated molecules focused on PIM1 with small memory and sample size
df_with_small= pd.read_csv("scores_smiles_filtered_final_test_prior_20ep_agent_20ep_smaller_memory_sample_size.csv")

# Read dataset with generated molecules focused on PIM1 with big memory and sample size
df_with_big= pd.read_csv("scores_smiles_filtered_final_test_prior_20ep_agent_20ep.csv")


# Add a column in order to indicate the source dataset

df_with_small["Memory & sample size"] = 'Reduced Memory_Size & Sample_Size in scoring Function'

df_with_big["Memory & sample size"] = 'Increased Memory_Size & Sample_Size in scoring Function'

# Concatenate the datasets
combined_df = pd.concat([df_with_small, df_with_big], ignore_index=True)

# Process the SMILES strings
invalid_smiles = []
for idx, row in combined_df.iterrows():
    smiles = row["SMILES"]
    dataset = row["Memory & sample size"]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles.append((smiles, dataset))
    except Exception as e:
        print(f"Error processing SMILES: {smiles} (Memory & sample size: {dataset}) - {e}")

# Invalid SMILES & corresponding datasets
for smiles, dataset in invalid_smiles:
    print(f"Invalid SMILES: {smiles} (Memory & sample size: {dataset})")

# Drop rows with NaN values in the SMILES column
combined_df = combined_df.dropna(subset=['SMILES'])

# Convert float values to string representation
combined_df['SMILES'] = combined_df['SMILES'].astype(str)

# Convert SMILES to fingerprints for combined datasets
fp_list, removed_list = fp_list_from_smiles_list(combined_df['SMILES'])


# PCA on combined dataset
pca = PCA(n_components=2)
crds = pca.fit_transform(fp_list)

# PCA coordinates into a dataframe
crds_df = pd.DataFrame(crds, columns=["PC_1", "PC_2"])
combined_df.reset_index(drop=True, inplace=True)
remaining_df = combined_df.drop(removed_list, inplace=False)
remaining_df.reset_index(drop=True, inplace=True)
crds_df['Memory & sample size'] = remaining_df["Memory & sample size"]

# Set the size of the plot
sns.set(rc={'figure.figsize': (16, 10)}) 
sns.set_style('whitegrid')
ax = sns.scatterplot(data=crds_df, x="PC_1", y="PC_2", hue="Memory & sample size", palette={'Reduced Memory_Size & Sample_Size in scoring Function': "red",'Increased Memory_Size & Sample_Size in scoring Function': "green"})
# Add a bold title to the plot
plt.title('PCA - use case - memory & sample size', fontweight='bold')
plt.show()


var = np.sum(pca.explained_variance_ratio_)
var


#t-SNE
pca = PCA(n_components=50)
pca_result = pca.fit_transform(fp_list)

np.random.seed(42)

# Apply t-SNE to the PCA result
tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(pca_result)

# Create a DataFrame for the t-SNE result
tsne_df = pd.DataFrame(tsne_result, columns=["X", "Y"])
# Add the Dataset column to remaining_df
tsne_df['Memory & sample size'] = remaining_df["Memory & sample size"]


# Set the size of the plot
sns.set(rc={'figure.figsize': (16, 10)})  
sns.set_style('whitegrid')

# Plot the t-SNE result with different colors for initial and generated datasets
ax = sns.scatterplot(data=tsne_df, x="X", y="Y", hue="Memory & sample size", palette={'Reduced Memory_Size & Sample_Size in scoring Function': "red",'Increased Memory_Size & Sample_Size in scoring Function': "green"})
plt.title('t-SNE - use case - memory & sample size', fontweight='bold')
plt.show()

