import os
import pandas as pd
import numpy as np
import random
import pickle
import sklearn.ensemble
from sklearn.metrics import roc_auc_score, mean_squared_error
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, PandasTools
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.Draw import IPythonConsole
from IPython.core.display import display, HTML
from sklearn.model_selection import GridSearchCV
from shutil import copyfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import validation_curve


# Set the plotting parameters
large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

# Define output directory
output_dir = os.path.expanduser("/home/sofia/Downloads/Datasets/PIM1 Kinase/REINVENT_model_building_demo_Final_Test")

# --------- do not change
# get the notebook's root path
try: ipynb_path
except NameError: ipynb_path = os.getcwd()

# if required, generate a folder to store the results
try:
    os.mkdir(output_dir)
except FileExistsError:
    pass
# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


#Define the file paths
train_file_path = os.path.join(output_dir, "train_set.csv")
test_file_path = os.path.join(output_dir, "test_set.csv")

# Copy data sets to output folder
for file_path in [train_file_path, test_file_path]:
    if not os.path.exists(file_path):
        copyfile(os.path.join(ipynb_path, file_path), file_path)

#Function to check SMILES validity
def check_smiles_validity(dataset):
    invalid_smiles = []
    for index, row in dataset.iterrows():
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles.append((index, smiles))
    return invalid_smiles

# Load train and test datasets
train = pd.read_csv(os.path.join(output_dir, "train_set.csv"))
test = pd.read_csv(os.path.join(output_dir, "test_set.csv"))

print("# obs in train: ", train.shape[0])
print("# obs in test: ", test.shape[0])

# Check SMILES validity in train set
print("Checking SMILES validity in train set:")
invalid_smiles_train = check_smiles_validity(train)
if invalid_smiles_train:
    print("Invalid SMILES found in train set:")
    for index, smiles in invalid_smiles_train:
        print(f"Index: {index}, SMILES: {smiles}")
else:
    print("All SMILES in train set are valid.")

# Check SMILES validity in test set
print("Checking SMILES validity in test set:")
invalid_smiles_test = check_smiles_validity(test)
if invalid_smiles_test:
    print("Invalid SMILES found in test set:")
    for index, smiles in invalid_smiles_test:
        print(f"Index: {index}, SMILES: {smiles}")
else:
    print("All SMILES in test set are valid.")

#Descriptors
'''We need to compute descriptors (fingerprints) compatible with `REINVENT`.
   Below is code for generating ECFP6 fingerprints with counts, '''

def smiles_to_mols(query_smiles):
    mols = [Chem.MolFromSmiles(smile) for smile in query_smiles]
    valid = [0 if mol is None else 1 for mol in mols]
    valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
    valid_mols = [mols[idx] for idx in valid_idxs]
    return valid_mols, valid_idxs

class Descriptors:

    def __init__(self, data):
        self._data = data

    def ECFP(self, radius, nBits):
        fingerprints = []
        mols, idx = smiles_to_mols(self._data)
        fp_bits = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits) for mol in mols]
        for fp in fp_bits:
            fp_np = np.zeros((1, nBits), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return fingerprints, idx

    def ECFP_counts(self, radius, useFeatures, useCounts=True):
        mols, valid_idx = smiles_to_mols(self._data)
        fps = [AllChem.GetMorganFingerprint(mol, radius, useCounts=useCounts, useFeatures=useFeatures) for mol in mols]
        size = 2048
        nfp = np.zeros((len(fps), size), np.int32)
        for i, fp in enumerate(fps):
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % size
                nfp[i, nidx] += int(v)
        return nfp, valid_idx

    def Avalon(self, nBits):
        mols, valid_idx = smiles_to_mols(self._data)
        fingerprints = []
        fps = [pyAvalonTools.GetAvalonFP(mol, nBits=nBits) for mol in mols]
        for fp in fps:
            fp_np = np.zeros((1, nBits), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return fingerprints, valid_idx

    def MACCS_keys(self):
        mols, valid_idx = smiles_to_mols(self._data)
        fingerprints = []
        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
        for fp in fps:
            fp_np = np.zeros((1, ), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return fingerprints, valid_idx

def get_ECFP6_counts(inp):
    if not isinstance(inp, list):
        inp = list(inp)
    desc = Descriptors(inp)
    fps, _ = desc.ECFP_counts(radius=3, useFeatures=True, useCounts=True)
    return fps

train_fps = get_ECFP6_counts(train["SMILES"])
test_fps = get_ECFP6_counts(test["SMILES"])

# Check for NaN or Infinity Values
print("Summary statistics of the training data:")
print(train.describe())

# Remove NaN and Infinity values from the target variable
train = train.replace([np.inf, -np.inf], np.nan)
train = train.dropna(subset=["Binary_Activity"])

# Check again for NaN or Infinity values in the target variable
nan_check_target = train["Binary_Activity"].isnull().sum().sum()
inf_check_target = not np.isfinite(train["Binary_Activity"]).all()

print(f"Number of NaN values in target variable: {nan_check_target}")
print(f"Target variable contains Infinity values: {inf_check_target}")

# Check the length of features and target variable
print(len(train_fps), len(train["Binary_Activity"]))

# If there is a mismatch, trim the longer one
min_length = min(len(train_fps), len(train["Binary_Activity"]))
train_fps = train_fps[:min_length]
train["Binary_Activity"] = train["Binary_Activity"][:min_length]

# Verify the length again
print(len(train_fps), len(train["Binary_Activity"]))

# Hide all deprecation/depreciation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Model training

# Hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [None, 2, 5, 10],
    'max_features': ['auto', 'sqrt', 'log2'],
    'class_weight': [None, 'balanced', 'balanced_subsample'],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5, 10, 15]
}

# GridSearchCV for hyperparameter tuning
RFclassifier = sklearn.ensemble.RandomForestClassifier()
grid_search = GridSearchCV(estimator=RFclassifier, param_grid=param_grid, scoring='roc_auc', cv=5)
grid_search.fit(train_fps, train["Binary_Activity"])

# Print best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Get best model
best_RFclassifier = grid_search.best_estimator_

# Predict on train and test sets using the best model
y_pred_train = best_RFclassifier.predict(train_fps)
y_pred_test = best_RFclassifier.predict(test_fps)

# Compute AUC-ROC scores
train_score_best = roc_auc_score(y_true=train["Binary_Activity"], y_score=y_pred_train)
test_score_best = roc_auc_score(y_true=test["Binary_Activity"], y_score=y_pred_test)

print("Training AUC with best model:", train_score_best)
print("Test AUC with best model:", test_score_best)



# Validation curves for each hyperparameter

# Validation Curve: how the performance metric (AUC) changes with different values of a hyperparameter

# Function to plot validation curves
def plot_validation_curve(param_range, train_scores, test_scores, param_name):
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, np.mean(train_scores, axis=1), label="Training AUC")
    plt.plot(param_range, np.mean(test_scores, axis=1), label="Test AUC")
    plt.title(f"Validation Curve for {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("AUC Score")
    plt.legend()
    plt.show()

# Validation curves for hyperparameters
for param_name in ["max_depth", "n_estimators", "min_samples_split", "min_samples_leaf"]:
    param_range = [None, 2, 5, 10] if param_name == "max_depth" else [50, 100, 150]
    train_scores, test_scores = validation_curve(
        sklearn.ensemble.RandomForestClassifier(),
        train_fps, train["Binary_Activity"],
        param_name=param_name, param_range=param_range,
        scoring="roc_auc", cv=5
    )
    plot_validation_curve(param_range, train_scores, test_scores, param_name)

# Save the best hyperparameters to a file
with open(os.path.join(output_dir, "best_hyperparameters.pkl"), "wb") as f:
    pickle.dump(grid_search.best_params_, f)