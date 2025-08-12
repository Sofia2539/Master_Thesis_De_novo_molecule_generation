"""

PCA for generated molecules PIM1 between 3 RL models with different combination of prior and agent. The models include:

    Prior: pre-trained model on 1 epoch in Broad Chembl and input Smiles all Kinases & Agent:pre-trained model on 20 epochs in Broad Chembl and input Smiles PIM1 dataset trained on 1epoch
    Prior: pre-trained model on 5 epoch in Broad Chembl and input Smiles all Kinases & Agent:pre-trained model on 20 epochs in Broad Chembl and input Smiles PIM1 dataset trained on 1epoch
    Prior: pre-trained model on 20 epoch in Broad Chembl and input Smiles all Kinases & Agent:pre-trained model on 20 epochs in Broad Chembl and input Smiles PIM1 dataset trained on 1epoch

In all models the necessary manipulation in the code helped from raising a KeyError while avoiding molecules with undesired properties.

I have decided to continue with this combination between prior and agent to all of the following tests. Prior trained in Broad dataset but with input SMILES all kinases and agent trained on Broad dataset and input SMILES PIM1 kinase dataset.

"""

##Investigation of the importance of different epochs in the prior model

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

# Read the dataset with model 1
df_model1 = pd.read_csv("/home/sofia/Downloads/Datasets/PIM1 Kinase/REINVENT_RL_demo/Final_Tests/scores_smiles_filtered_final_test_20ep.csv")

# Read the dataset with model 2
df_model2 = pd.read_csv("/home/sofia/Downloads/Datasets/PIM1 Kinase/REINVENT_RL_demo/Final_Tests/Prior_5ep/scores_smiles_filtered_final_test_prior_5ep_agent_20ep.csv")

# Read the dataset with model 3
df_model3 = pd.read_csv("/home/sofia/Downloads/Datasets/PIM1 Kinase/REINVENT_RL_demo/Final_Tests/Prior_20ep/scores_smiles_filtered_final_test_prior_20ep_agent_20ep.csv")


# Add a column to indicate the source dataset
df_model1["Number of epochs in prior"] = '1 epoch'
df_model2["Number of epochs in prior"] = '5 epoch'
df_model3["Number of epochs in prior"] = '20 epoch'

# Combine both datasets into a single dataframe
combined_df = pd.concat([df_model1, df_model2,df_model3], ignore_index=True)


# Process the SMILES strings
invalid_smiles = []
for idx, row in combined_df.iterrows():
    smiles = row["SMILES"]
    dataset = row["Number of epochs in prior"]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles.append((smiles, dataset))
    except Exception as e:
        print(f"Error processing SMILES: {smiles} (Number of epochs in prior: {dataset}) - {e}")



# Invalid SMILES & corresponding datasets
for smiles, dataset in invalid_smiles:
    print(f"Invalid SMILES: {smiles} (Number of epochs in prior: {dataset})")

# Drop rows with NaN values in the SMILES column
combined_df = combined_df.dropna(subset=['SMILES'])

# Convert float values to string representation
combined_df['SMILES'] = combined_df['SMILES'].astype(str)

# Convert SMILES to fingerprints for combined datasets
fp_list, removed_list = fp_list_from_smiles_list(combined_df['SMILES'])

# **PCA analysis**

# Perform PCA on combined dataset
pca = PCA(n_components=2)
crds = pca.fit_transform(fp_list)

# PCA coordinates into a dataframe
crds_df = pd.DataFrame(crds, columns=["PC_1", "PC_2"])
combined_df.reset_index(drop=True, inplace=True)
remaining_df = combined_df.drop(removed_list, inplace=False)
remaining_df.reset_index(drop=True, inplace=True)
crds_df["Number of epochs in prior"] = remaining_df["Number of epochs in prior"]

# Set the size of the plot
sns.set(rc={'figure.figsize': (20, 16)}) 
sns.set_style('whitegrid')
# Make a plot of the principal components
ax = sns.scatterplot(data=crds_df, x="PC_1", y="PC_2", hue="Number of epochs in prior", palette={'1 epoch': "green", '5 epoch': "red", '20 epoch': "blue"})
# Add a bold title to the plot
plt.title('PCA - Number of epochs prior', fontweight='bold')

plt.show()

var = np.sum(pca.explained_variance_ratio_)
var



# **t-SNE analysis**

pca = PCA(n_components=50)
pca_result = pca.fit_transform(fp_list)
np.random.seed(42)

# Apply t-SNE to the PCA result
tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(pca_result)

## Create a DataFrame for the t-SNE result
tsne_df = pd.DataFrame(tsne_result, columns=["X", "Y"])
# Add the Dataset column back to tsne_df
tsne_df["Number of epochs in prior"] = remaining_df["Number of epochs in prior"]

# Set the size of the plot
sns.set(rc={'figure.figsize': (20, 16)}) 
sns.set_style('whitegrid')

# Plot the t-SNE result with different colors for initial and generated datasets
ax = sns.scatterplot(data=tsne_df, x="X", y="Y", hue="Number of epochs in prior", palette={'1 epoch': "green", '5 epoch': "red", '20 epoch': "blue"})
plt.title('t-SNE Number of epochs prior', fontweight='bold')

plt.show()

lvar = np.sum(pca.explained_variance_ratio_)
var
