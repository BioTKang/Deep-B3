from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
import numpy as np

import os
import math
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
)

def smi_to_mol(smi):
    mol = None
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                 Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                 Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                 Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                 Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                 Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=False)
    return mol

def make_tabular(mol, calculator):
    value_rdkit = list(calculator.CalcDescriptors(mol))
    return value_rdkit

def make_dirs(train_test):
    dirs_images = os.path.join(BASE_DIR, '{0}_images'.format(train_test))
    dirs_tabular = os.path.join(BASE_DIR, train_test)
    if not os.path.exists(dirs_images):
        os.makedirs(dirs_images)
    if not os.path.exists(dirs_tabular):
        os.makedirs(dirs_tabular)


def make_image(mol, id, train_test):
    contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, [x for x, y in contribs])
    img_path = os.path.join(BASE_DIR, '{0}_images'.format(train_test))
    fig.savefig(os.path.join(img_path, '{0}.png'.format(id)), bbox_inches='tight')
    fig.clf()


def balance_data(file):
    logging.info('begin balance data')
    df = pd.read_csv(file)
    value_count = df['label'].value_counts()
    pos_num = value_count[1]
    neg_num = value_count[0]
    logging.info('pos samples num: {0}'.format(pos_num))
    logging.info('neg samples num: {0}'.format(neg_num))
    new_neg = []
    time = math.floor(pos_num / neg_num)
    for index, row in df.iterrows():
        id = row['id']
        smi = row['smi']
        try:label = row['label']
        except: label=0
        if label == 0:
            smis = []
            mol = smi_to_mol(smi)
            for i in range(100):
                try:
                    smis.append(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
                except:
                    logging.error('error, {0}:{1}'.format(id, smi))
            smis = list(set(smis))
            if smi in smis:
                smis.remove(smi)
            if len(smis) == 0:
                logging.warning('{0} generated 0 times'.format(smi))
            for i in range(len(smis)):
                if i < time-1:
                    nid = '{0}_{1}'.format(id, i)
                    new_neg.append([nid, smis[i], 0])
    new_test_df = pd.DataFrame(columns=['id', 'smi', 'label'], data=new_neg)
    new_df = pd.concat([df, new_test_df])
    outfile = '{0}.balance'.format(file)
    new_df.to_csv(outfile, index=False)
    logging.info('balance data ok')
    return outfile

def check_file(file):
    return os.path.isfile(file)

def parse_args():
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--smifile', required=True, help="smi file name with csv format")
    parser.add_argument('--outfile', help="tabular feature file name")
    parser.add_argument('--is_balance', action='store_true', help="balance data, default false")
    parser.add_argument('--is_test', action='store_false', help="features for train or test")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    BASE_DIR = os.path.abspath('.')
    infile = os.path.join(BASE_DIR, args.smifile)
    if not check_file(infile):
        logging.error('input smi file not exists: {0}'.format(infile))
        print('input file is not exists')
        exit(0)

    outfile = args.outfile
    train_test = 'train' if args.is_test else 'test'
    print('generated features for {0} model'.format(train_test))

    make_dirs(train_test)
    if args.is_balance:
        infile = balance_data(infile)
    if not outfile:
        outfile = '{0}.feature'.format(args.smifile)
        outfile = os.path.join(BASE_DIR, './{0}/{1}'.format(train_test, outfile))
    logging.info('feature out file: {0}'.format(outfile))
    des_list = [x[0] for x in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)

    des_list.insert(0, 'ID')
    des_list.insert(1, 'target')
    des_list.append('smi')
    features = pd.DataFrame(columns=des_list)
    features.to_csv(outfile, index=False)
    df = pd.read_csv(infile)
    for index, row in df.iterrows():
        id = row['id']
        smi = row['smi']
        label = row['label']
        mol = smi_to_mol(smi)
        if mol:
            tabular = make_tabular(mol, calculator)
            tabular.insert(0, id)
            tabular.insert(1, label)
            tabular.append(smi)

            new_tab = pd.DataFrame(columns=des_list, data=[tabular])
            new_tab.to_csv(outfile, index=False, header=None, mode='a')
            make_image(mol, id, train_test)
    logging.info('make feature Completion !')