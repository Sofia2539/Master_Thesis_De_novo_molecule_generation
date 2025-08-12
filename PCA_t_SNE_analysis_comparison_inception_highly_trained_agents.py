"""PCA for generated molecules with focus on PIM1 kinase: with and without inception inlcuded in the 
parameters - while exploring the impact of the inception in a highly trained agent as well"""

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


# Read dataset with generated molecules focused on PIM1 with inception - 20 epochs
df_with_inception_20 = pd.read_csv("/home/sofia/Downloads/Datasets/PIM1 Kinase/REINVENT_RL_demo/Final_Tests/Prior_20ep/scores_smiles_filtered_final_test_prior_20ep_agent_20ep.csv")

# Read dataset with generated molecules focused on PIM1 with inception - 50 epochs
df_with_inception_50 = pd.read_csv("/home/sofia/Downloads/Datasets/PIM1 Kinase/REINVENT_RL_demo/Final_Tests/Prior_20ep/scores_smiles_filtered_final_test_prior_20ep_agent_50ep.csv")

# Read dataset with generated molecules focused on PIM1 with inception - 100 epochs
df_with_inception_100 = pd.read_csv("/home/sofia/Downloads/Datasets/PIM1 Kinase/REINVENT_RL_demo/Final_Tests/Prior_20ep/scores_smiles_filtered_final_test_prior_20ep_agent_100ep.csv")

# Read dataset with generated molecules focused on PIM1 without inception
df_without_inception = pd.read_csv("/home/sofia/Downloads/Datasets/PIM1 Kinase/REINVENT_RL_demo/Final_Tests/Prior_20ep/scores_smiles_filtered_final_test_prior_20ep_agent_20ep_no_inception.csv")


inception_smiles =[ "Cc1nc2cccc3c2nc1NC/C=C\\C[C@@H]1CNC(=O)c2cc-3[nH]c21",
                    "CCCNc1nc2c(-c3cc4c([nH]3)CCNC4=O)cccc2nc1C",
                    "C/C1=C\\C[C@H]2CNC(=O)c3cc([nH]c32)-c2cccc3nc(C)c(nc23)NC1",
                    "Cc1nc2cccc3c2nc1N[C@H](C)/C=C/C[C@H]1CNC(=O)c2cc-3[nH]c21",
                    "Cc1nc2cccc3c2nc1NCCCC[C@@H]1CNC(=O)c2cc-3[nH]c21Nc1nccnc1C(=O)Nc1cnccc1N1CCC[C@H](N)C1",
                    "Nc1cccnc1C(=O)Nc1cnccc1N1CCC[C@H](N)C1",
                    "Nc1ccc(Oc2ccccc2)nc1C(=O)Nc1cnccc1N1CCC[C@H](N)C1",
                    "Nc1nccc(-c2ccc(N)c(C(=O)Nc3cnccc3N3CCC[C@H](N)C3)n2)n1",
                    "Nc1ccc(-c2ccccc2C(F)(F)F)nc1C(=O)Nc1cnccc1N1CCC[C@H](N)C1CCOc1cncc(-c2ccc3[nH]cc(-c4ccnc(N)n4)c3c2)n1",
                    "CN(C)CCOc1cncc(-c2ccc3[nH]cc(-c4ccnc(N)n4)c3c2)n1",
                    "CCN(CC)CCOc1cncc(-c2ccc3[nH]cc(-c4ccnc(N)n4)c3c2)n1",
                    "CN(C)CCCOc1cncc(-c2ccc3[nH]cc(-c4ccnc(N)n4)c3c2)n1",
                    "CCN(CC)CCCOc1cncc(-c2ccc3[nH]cc(-c4ccnc(N)n4)c3c2)n1Fc1ccccc1-c1cccc(CNc2cnccc2OC2CCNCC2)n1",
                    "Fc1cccc(F)c1-c1nc(-c2ccc3nccc(N4CCNCC4)c3c2)cs1",
                    "NC1CCN(c2cncc(-c3n[nH]c4ccc(-c5c(F)cccc5F)cc34)n2)CC1",
                    "NC1CCN(c2cncc(-c3n[nH]c4ccc(-c5c(F)cccc5F)cc34)n2)C1",
                    "Fc1cccc(F)c1-c1ccc2[nH]nc(-c3cncc(N4CCCNCC4)n3)c2c1O=C(O)c1ccc2c(c1)nc(Nc1cccc(Cl)c1)c1ccncc12",
                    "COc1cccc(Nc2nc3c(-c4nnc[nH]4)cccc3c3cnccc23)c1Cl",
                    "CN(C)c1cccc(-n2nnc3ccc(NCC4CCNCC4)nc32)c1",
                    "O=c1[nH]c(CN2CC[C@@H](O)C2)nc2c1oc1ccc(Br)cc12",
                    "O=c1[nH]c(C2CCNCC2)nc2c1oc1ccc(Br)cc12",
                    "FC(F)(F)c1ccc(-c2nnc3ccc(NC4CCCCC4)nn23)cc1",
                    "O=c1[nH]c(-c2ccccc2Cl)nc2c1oc1ccc(C(F)(F)F)cc12",
                    "O=c1[nH]c(-c2ccc(CNC3CCNCC3)cc2Cl)nc2c1oc1ccc(Br)cc12",
                    "O=c1[nH]c(CN2CC(O)C2)nc2c1oc1ccc(Br)cc12",
                    "CCOc1ccc2c(c1CN1CCNCC1)O/C(=C\\c1c[nH]c3ccccc13)C2=O",
                    "O=C1Nc2ccccc2/C1=C/c1c[nH]nc1-c1ccccc1[N+](=O)[O-]",
                    "O=C(Nc1c[nH]c2ncc(-c3c(F)cccc3F)cc12)c1cnn2ccc(OCC3CCNCC3)nc12"]

df_inception = pd.DataFrame(inception_smiles, columns=["SMILES"])
df_inception["Parameters"] = "Inception SMILES"

# Add a column in order to indicate the source dataset
df_with_inception_20["Parameters"] = 'Including Inception - Agent trained on 20 epochs'
df_with_inception_50["Parameters"] = 'Including Inception - Agent trained on 50 epochs'
df_with_inception_100["Parameters"] = 'Including Inception - Agent trained on 100 epochs'

df_without_inception["Parameters"] = 'Excluding Inception'

# Concatenate the datasets
combined_df = pd.concat([df_with_inception_20, df_with_inception_50, df_with_inception_100, df_without_inception, df_inception], ignore_index=True)

# Process the SMILES strings
invalid_smiles = []
for idx, row in combined_df.iterrows():
    smiles = row["SMILES"]
    dataset = row["Parameters"]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles.append((smiles, dataset))
    except Exception as e:
        print(f"Error processing SMILES: {smiles} (Parameters: {dataset}) - {e}")

# Invalid SMILES & corresponding datasets
for smiles, dataset in invalid_smiles:
    print(f"Invalid SMILES: {smiles} (Parameters: {dataset})")
    

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
crds_df['Parameters'] = remaining_df["Parameters"]

# Create a size column with larger size for inception SMILES
crds_df['Size'] = crds_df['Parameters'].apply(lambda x: 200 if x == 'Inception SMILES' else 50)

# Set the size of the plot
sns.set(rc={'figure.figsize': (16, 10)})  
sns.set_style('whitegrid')
ax = sns.scatterplot(data=crds_df, x="PC_1", y="PC_2", hue="Parameters", size="Size", sizes=(50, 200), legend="full", palette={'Including Inception - Agent trained on 20 epochs': "green",'Including Inception - Agent trained on 50 epochs': "orange", 'Including Inception - Agent trained on 100 epochs': "grey", 'Excluding Inception': "red", 'Inception SMILES': "blue"})
# Create the legend without the size
handles, labels = ax.get_legend_handles_labels()
unique_labels = sorted(set(labels), key=labels.index)
new_handles = [handles[labels.index(label)] for label in unique_labels if label not in ['50', '200']]
new_labels = [label for label in unique_labels if label not in ['50', '200']]
ax.legend(new_handles, new_labels)

# Add bold title to the plot
plt.title('PCA - use case - inception', fontweight='bold')
plt.show()

var = np.sum(pca.explained_variance_ratio_)
print(f'Explained variance by PCA: {var}')

# t-SNE
pca = PCA(n_components=50)
pca_result = pca.fit_transform(fp_list)

np.random.seed(42)

# Apply t-SNE to the PCA result
tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(pca_result)

# Create a DataFrame for the t-SNE result
tsne_df = pd.DataFrame(tsne_result, columns=["X", "Y"])
# Add the Dataset column to remaining_df
tsne_df['Parameters'] = remaining_df["Parameters"]

# Create a size column with larger size for inception SMILES
tsne_df['Size'] = tsne_df['Parameters'].apply(lambda x: 200 if x == 'Inception SMILES' else 50)

# Set the size of the plot
sns.set(rc={'figure.figsize': (16, 10)})  
sns.set_style('whitegrid')

# Plot the t-SNE result with different colors 
ax = sns.scatterplot(data=tsne_df, x="X", y="Y", hue="Parameters", size="Size", sizes=(50, 200), legend="full", palette={'Including Inception - Agent trained on 20 epochs': "green",'Including Inception - Agent trained on 50 epochs': "orange", 'Including Inception - Agent trained on 100 epochs': "grey", 'Excluding Inception': "red", 'Inception SMILES': "blue"})

# Create the legend without the size
handles, labels = ax.get_legend_handles_labels()
unique_labels = sorted(set(labels), key=labels.index)
new_handles = [handles[labels.index(label)] for label in unique_labels if label not in ['50', '200']]
new_labels = [label for label in unique_labels if label not in ['50', '200']]
ax.legend(new_handles, new_labels)

plt.title('t-SNE - use case - inception', fontweight='bold')
plt.show()
