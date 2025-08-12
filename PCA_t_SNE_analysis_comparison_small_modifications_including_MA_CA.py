"""PCA for generated molecules with focus on PIM1 kinase: small modification in the 
components / Including - excluding Matching Substructure/ Custom Alert/ Increasing Weights of those components"""


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


# Read dataset with generated molecules focused on PIM1 with Matching Substructure & Custom Alert
df1_modified= pd.read_csv("/home/sofia/Downloads/Datasets/PIM1 Kinase/REINVENT_RL_demo/Final_Tests/Prior_20ep/scores_smiles_filtered_final_test_prior_20ep_agent_20ep_small_modifications_weights_components.csv")

# Read dataset with generated molecules focused on PIM1 without Matching Substructure & Custom Alert
df1= pd.read_csv("/home/sofia/Downloads/Datasets/PIM1 Kinase/REINVENT_RL_demo/Final_Tests/Prior_20ep/scores_smiles_filtered_final_test_prior_20ep_agent_20ep_without_MS_CA.csv")

# Read SMILES used for Matching Substructure
MS_smiles = ["O=C(O)[C@]1(Cc2cccc(Nc3cc[nH]n3)n2)CC[C@@H](Oc2cccc(C(F)(F)F)c2F)CC1",
            "NC(=O)c1cccc2c(NCc3cccc(NC(=O)c4cccc(Cl)c4)c3)ncnc12",
            "COCCC(Nc1ncnc2c(C(N)=O)cccc12)c1cccc(NC(=O)c2ccc(OC)cc2F)c1",
            "Cc1cc(Nc2cc[nH]n2)nc(C[C@]2(c3n[nH]c(=O)o3)CC[C@H](Oc3cccc(Cl)c3F)CC2)n1",
            "CC1(C)CC(NC(=O)c2ccc(Nc3ncc(Cl)c(Nc4ccccc4Cl)n3)cc2)CC(C)(C)N1[O]",
            "NCc1cnc(Nc2cccc(CNc3ncnc4c(C(N)=O)cccc34)c2)s1",
            "NC(=O)c1cccc2c(NCc3cccc(NC(=O)c4ccc5c(c4)CCO5)c3)ncnc12",
            "Fc1c(O[C@H]2CC[C@](Cc3cccc(Nc4cc[nH]n4)n3)(c3n[nH]c(=S)o3)CC2)cccc1C(F)(F)F",
            "NC(=O)c1cccc2c(NCc3cccc(NC(=O)c4[nH]nc5c4CCCC5)c3)ncnc12",
            "COc1ccc(C(=O)Nc2cccc(CNc3ncnc4c(C(N)=O)cc(F)cc34)c2)cc1",
            "COc1cc2c(Nc3cnc(NC(=O)c4cccc(Cl)c4)nc3)ncnc2cc1OCCCN1CCCCC1"]

# Read SMILES used for Custom Alert
CA_smiles = [ "O=c1[nH]c2ccc(Nc3nccc(Nc4ccccc4Cl)n3)cc2[nH]1",
              "O=C(Nc1nc2ccc3[nH]ncc3c2s1)C1(Nc2ccc(Cl)cc2)CC1",
              "Cc1cc(Nc2nc(N[C@@H](C)c3ncc(F)cn3)ncc2Cl)[nH]n1",
              "Cc1cc(Nc2cc(-c3ccccc3)c(C#N)c(Sc3ccc(NC(=O)C(C)C)cc3)n2)n[nH]1",
              "NC(=O)c1cccc2c(NCc3cccc(NC(=O)Nc4ccccc4F)c3)ncnc12",
              "NC[C@@H](Nc1ncnc2c(C(N)=O)cccc12)c1ccccc1",
              "CCNC[C@@H](Nc1ncnc2c1=CCCC=2C(N)=O)c1cccc(F)c1",
              "CCN(CCO)CCCOc1ccc2c(Nc3ccc(NC(=O)NC4CCCCC4)cc3)ncnc2c1",
              "CCC(=O)Nc1cc(NC2(C(=O)Nc3nc4ccc5[nH]ncc5c4s3)CC2)ccc1F",
              "O=C(Cc1ccc(Nc2nccc(NC3CCCC3)n2)cc1)NC1CCCCC1",
              "O=C(Nc1ccc(CCNc2ncnc3oc(-c4ccccc4)c(-c4ccccc4)c23)cc1)c1ccccc1",
              "Cc1cc(Nc2nc(Sc3ccc(NC(=O)CN4C[C@H](OC(C)(C)C)C[C@@H]4CO)cc3)nn3cccc23)n[nH]1"]

df_MA = pd.DataFrame(MS_smiles, columns=["SMILES"])
df_MA["Components"] = "Matching Substructure SMILES"

df_CA = pd.DataFrame(CA_smiles, columns=["SMILES"])
df_CA["Components"] = "Custom Alert SMILES"

# Add a column in order to indicate the source dataset
df1_modified["Components"] = 'Including Custom_Alert & Matching_Substructure - increased Weights in Components'

df1["Components"] = 'Exluding Custom_Alert & Matching_Substructure'

# Concatenate the datasets
combined_df = pd.concat([df1_modified, df1, df_MA, df_CA], ignore_index=True)

# Process the SMILES strings
invalid_smiles = []
for idx, row in combined_df.iterrows():
    smiles = row["SMILES"]
    dataset = row["Components"]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles.append((smiles, dataset))
    except Exception as e:
        print(f"Error processing SMILES: {smiles} (Components: {dataset}) - {e}")

# Invalid SMILES & corresponding datasets
for smiles, dataset in invalid_smiles:
    print(f"Invalid SMILES: {smiles} (Components: {dataset})")

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
crds_df['Components'] = remaining_df["Components"]

# Create a size column with larger size for inception SMILES
crds_df['Size'] = crds_df['Components'].apply(lambda x: 200 if x == 'Matching Substructure SMILES' else 50)
crds_df['Size'] = crds_df['Components'].apply(lambda x: 200 if x == 'Custom Alert SMILES' else 50)


# Set the size of the plot
sns.set(rc={'figure.figsize': (16, 10)})  
sns.set_style('whitegrid')
ax = sns.scatterplot(data=crds_df, x="PC_1", y="PC_2", hue="Components", size="Size", sizes=(50, 200), legend="full", palette={'Including Custom_Alert & Matching_Substructure - increased Weights in Components': "red",'Exluding Custom_Alert & Matching_Substructure': "green", "Matching Substructure SMILES": "blue", "Custom Alert SMILES": "black" })

# Create the legend without the size
handles, labels = ax.get_legend_handles_labels()
unique_labels = sorted(set(labels), key=labels.index)
new_handles = [handles[labels.index(label)] for label in unique_labels if label not in ['50', '200']]
new_labels = [label for label in unique_labels if label not in ['50', '200']]
ax.legend(new_handles, new_labels)

# Add a bold title to the plot
plt.title('PCA - use case - Custom_Alert & Matching_Substructure', fontweight='bold')
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
tsne_df['Components'] = remaining_df["Components"]

# Create a size column with larger size for inception SMILES
tsne_df['Size'] = tsne_df['Components'].apply(lambda x: 200 if x == 'Matching Substructure SMILES' else 50)
tsne_df['Size'] = tsne_df['Components'].apply(lambda x: 200 if x == 'Custom Alert SMILES' else 50)

# Set the size of the plot
sns.set(rc={'figure.figsize': (16, 10)})  
sns.set_style('whitegrid')

# Plot the t-SNE result with different colors for initial and generated datasets
ax = sns.scatterplot(data=tsne_df, x="X", y="Y", hue="Components", size="Size", sizes=(50, 200), legend="full", palette={'Including Custom_Alert & Matching_Substructure - increased Weights in Components': "red",'Exluding Custom_Alert & Matching_Substructure': "green", "Matching Substructure SMILES": "blue", "Custom Alert SMILES": "black" })

# Create the legend without the size
handles, labels = ax.get_legend_handles_labels()
unique_labels = sorted(set(labels), key=labels.index)
new_handles = [handles[labels.index(label)] for label in unique_labels if label not in ['50', '200']]
new_labels = [label for label in unique_labels if label not in ['50', '200']]
ax.legend(new_handles, new_labels)

# Add a bold title to the plot
plt.show()
