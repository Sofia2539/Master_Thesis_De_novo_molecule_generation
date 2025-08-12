#Load dependencies
import math
import os.path as op
import pickle
import csv
from collections import defaultdict
import os

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


_fscores = None


def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

   # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = sum(fps.values())  # Calculate the total number of fragments directly
    if nf != 0:
        for bitId, v in fps.items():
            sfp = bitId
            score1 += _fscores.get(sfp, -4) * v
        score1 /= nf
    else:
        score1 = 0

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    min_score = -4.0
    max_score = 2.5
    sascore = 11. - (sascore - min_score + 1) / (max_score - min_score) * 9.

    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore


def processSmiles(smiles_list, output_file):
    results = []
    for smiles in smiles_list:
        if not smiles or len(smiles) == 1: #Skip empty SMILES strings and atomic symbols
            continue
                
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            continue

        s = calculateScore(m)
        results.append((smiles, s))

    # Write results to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['SMILES', 'SAScore'])
        writer.writerows(results)



if __name__ == '__main__':
    import sys
    import time

    t1 = time.time()
    readFragmentScores("fpscores")
    t2 = time.time()

    # Read SMILES from CSV file
    input_file = sys.argv[1]
    base_filename, extension = os.path.splitext(input_file)
    output_file = base_filename + '_SAScore.csv'  # Create output filename by appending "_SAScore.csv" to the base filename

    smiles_list = []
    with open(input_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            smiles_list.append(row['SMILES'])

    t3 = time.time()
    processSmiles(smiles_list, output_file)
    t4 = time.time()

    print('Reading took %.2f seconds. Calculating took %.2f seconds' % ((t2 - t1), (t4 - t3)),
          file=sys.stderr)