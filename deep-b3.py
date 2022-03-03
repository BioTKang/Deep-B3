from fastai.vision import *
from fastai.tabular import *
from fastai.text import *
from fastai.callbacks import *

import numpy as np
import pandas as pd
import argparse
import os
import logging
import warnings
warnings.filterwarnings('ignore')

import model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
)
cont_names = ["MaxEStateIndex","MinEStateIndex","MaxAbsEStateIndex","MinAbsEStateIndex","qed","MolWt","HeavyAtomMolWt","ExactMolWt","NumValenceElectrons","NumRadicalElectrons","MaxPartialCharge","MinPartialCharge","MaxAbsPartialCharge","MinAbsPartialCharge","FpDensityMorgan1","FpDensityMorgan2","FpDensityMorgan3","BCUT2D_MWHI","BCUT2D_MWLOW","BCUT2D_CHGHI","BCUT2D_CHGLO","BCUT2D_LOGPHI","BCUT2D_LOGPLOW","BCUT2D_MRHI","BCUT2D_MRLOW","BalabanJ","BertzCT","Chi0","Chi0n","Chi0v","Chi1","Chi1n","Chi1v","Chi2n","Chi2v","Chi3n","Chi3v","Chi4n","Chi4v","HallKierAlpha","Ipc","Kappa1","Kappa2","Kappa3","LabuteASA","PEOE_VSA1","PEOE_VSA10","PEOE_VSA11","PEOE_VSA12","PEOE_VSA13","PEOE_VSA14","PEOE_VSA2","PEOE_VSA3","PEOE_VSA4","PEOE_VSA5","PEOE_VSA6","PEOE_VSA7","PEOE_VSA8","PEOE_VSA9","SMR_VSA1","SMR_VSA10","SMR_VSA2","SMR_VSA3","SMR_VSA4","SMR_VSA5","SMR_VSA6","SMR_VSA7","SMR_VSA8","SMR_VSA9","SlogP_VSA1","SlogP_VSA10","SlogP_VSA11","SlogP_VSA12","SlogP_VSA2","SlogP_VSA3","SlogP_VSA4","SlogP_VSA5","SlogP_VSA6","SlogP_VSA7","SlogP_VSA8","SlogP_VSA9","TPSA","EState_VSA1","EState_VSA10","EState_VSA11","EState_VSA2","EState_VSA3","EState_VSA4","EState_VSA5","EState_VSA6","EState_VSA7","EState_VSA8","EState_VSA9","VSA_EState1","VSA_EState10","VSA_EState2","VSA_EState3","VSA_EState4","VSA_EState5","VSA_EState6","VSA_EState7","VSA_EState8","VSA_EState9","FractionCSP3","HeavyAtomCount","NHOHCount","NOCount","NumAliphaticCarbocycles","NumAliphaticHeterocycles","NumAliphaticRings","NumAromaticCarbocycles","NumAromaticHeterocycles","NumAromaticRings","NumHAcceptors","NumHDonors","NumHeteroatoms","NumRotatableBonds","NumSaturatedCarbocycles","NumSaturatedHeterocycles","NumSaturatedRings","RingCount","MolLogP","MolMR","fr_Al_COO","fr_Al_OH","fr_Al_OH_noTert","fr_ArN","fr_Ar_COO","fr_Ar_N","fr_Ar_NH","fr_Ar_OH","fr_COO","fr_COO2","fr_C_O","fr_C_O_noCOO","fr_C_S","fr_HOCCN","fr_Imine","fr_NH0","fr_NH1","fr_NH2","fr_N_O","fr_Ndealkylation1","fr_Ndealkylation2","fr_Nhpyrrole","fr_SH","fr_aldehyde","fr_alkyl_carbamate","fr_alkyl_halide","fr_allylic_oxid","fr_amide","fr_amidine","fr_aniline","fr_aryl_methyl","fr_azide","fr_azo","fr_barbitur","fr_benzene","fr_benzodiazepine","fr_bicyclic","fr_diazo","fr_dihydropyridine","fr_epoxide","fr_ester","fr_ether","fr_furan","fr_guanido","fr_halogen","fr_hdrzine","fr_hdrzone","fr_imidazole","fr_imide","fr_isocyan","fr_isothiocyan","fr_ketone","fr_ketone_Topliss","fr_lactam","fr_lactone","fr_methoxy","fr_morpholine","fr_nitrile","fr_nitro","fr_nitro_arom","fr_nitro_arom_nonortho","fr_nitroso","fr_oxazole","fr_oxime","fr_para_hydroxylation","fr_phenol","fr_phenol_noOrthoHbond","fr_phos_acid","fr_phos_ester","fr_piperdine","fr_piperzine","fr_priamide","fr_prisulfonamd","fr_pyridine","fr_quatN","fr_sulfide","fr_sulfonamd","fr_sulfone","fr_term_acetylene","fr_tetrazole","fr_thiazole","fr_thiocyan","fr_thiophene","fr_unbrch_alkane","fr_urea"]

def set_random_seed(seed):
    import random
    random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    logging.info('deep-b3, set the random seed {0}'.format(seed))

def create_dirs(dir):
    path = os.path.join(BASE_DIR, dir)
    if not os.path.exists(path):
        logging.info('create a new dirs, {0}'.format(path))
        os.makedirs(path)
    return None

def read_csv_file(file):
    if not os.path.isfile(file):
        logging.error('{0} is not a file or not exists'.format(file))
        exit(-1)
    else:
        df = pd.read_csv(file)
        df.fillna(0, inplace=True)
        return df

def pretrain_nlp(train_df, test_df):
    bs = 8
    iter = 1
    bbbAll = pd.concat([train_df, test_df], ignore_index=True).reset_index(drop=True)
    smi = bbbAll[['label', 'smi']]
    data_lm = (TextList.from_df(
        smi, cols='smi'
    ).split_by_rand_pct(0.2).label_for_lm().databunch(bs=bs, path=Path(BASE_DIR)))
    nlp_lm_file = os.path.join(BASE_DIR, 'datapkl/{0}'.format('data_lm_smi.pkl'))
    logging.info('save the nlp lm file {0}'.format(nlp_lm_file))
    data_lm.save(nlp_lm_file)
    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
    logging.info('nlp pre-train model \n{0}'.format(learn.model))
    learn.lr_find()
    learn.recorder.plot(suggestion=True)
    lr_v = learn.recorder.min_grad_lr
    logging.info('find best learn rate {0}'.format(lr_v))
    learn.fit_one_cycle(iter, lr_v)
    learn.freeze_to(-2)
    learn.fit_one_cycle(iter, slice(lr_v / (2.6 ** 4), lr_v), moms=(0.8, 0.7))
    learn.freeze_to(-3)
    learn.fit_one_cycle(iter, slice(lr_v / 2 / (2.6 ** 4), lr_v / 2), moms=(0.8, 0.7))
    learn.unfreeze()
    learn.fit_one_cycle(iter, slice(lr_v / 10 / (2.6 ** 4), lr_v / 10), moms=(0.8, 0.7))
    logging.info('save the encoder at models/text_encoder')
    learn.save_encoder('text_encoder')
    return None

def test_deep_b3(train_df, test_df):
    bs = 64
    test_df['PicturePath'] = test_df.id.map(lambda x: 'test_images/{0}.png'.format(x))
    pklpath = os.path.join(BASE_DIR, 'datapkl')
    modelpath = os.path.join(BASE_DIR, 'models')
    modelfile = os.path.join(modelpath, 'deep-b3.pth')
    lmfile = os.path.join(pklpath, 'data_lm_smi.pkl')
    if not os.path.isfile(modelfile):
        logging.error('deep-b3 model file not exists, please re-train it')
        exit(0)
    if not os.path.isfile(lmfile):
        logging.error('smiles nlp lm file not exists')
        exit(0)
    path = Path(BASE_DIR)
    data_lm = load_data(Path(pklpath), 'data_lm_smi.pkl', bs=bs)
    vocab = data_lm.vocab
    procs = [FillMissing, Categorify, Normalize]
    size = 224

    imgListTest = ImageList.from_df(test_df, path=path, cols='PicturePath')
    tabListTest = TabularList.from_df(test_df, cat_names=[], cont_names=cont_names, procs=procs, path=path)
    textListTest = TextList.from_df(test_df, cols='smi', path=path, vocab=vocab)

    mixedTest = (MixedItemList([imgListTest, tabListTest, textListTest], path, inner_df=tabListTest.inner_df)
                 .split_none()
                 .label_from_df(cols='label', label_cls=FloatList)
                 .transform([[get_transforms()[0], [], []], [get_transforms()[1], [], []]], size=size))

    dataTest = mixedTest.databunch(bs=bs, collate_fn=model.collate_mixed)
    norm, denorm = model.normalize_custom_funcs(*imagenet_stats)
    dataTest.add_tfm(norm)  # normalize images

    learnTest = load_learner(path=modelpath, file='deep-b3.pkl')
    learnTest.data = dataTest

    preds, y = learnTest.get_preds(ds_type=DatasetType.Train)
    logging.info('test result prob:')
    print(preds)


def train_deep_b3(train_df, test_df, bs, vis_out, text_out):
    path = Path(BASE_DIR)
    pklpath = os.path.join(BASE_DIR, 'datapkl')
    size = 224
    procs = [FillMissing, Categorify, Normalize]

    train_df['PicturePath'] = train_df.id.map(lambda x: 'train_images/{0}.png'.format(x))
    test_df['PicturePath'] = test_df.id.map(lambda x: 'test_images/{0}.png'.format(x))

    bytarget = train_df.groupby(['id', 'label']).size().reset_index()
    bytarget = bytarget.sample(frac=.2, random_state=2022).drop([0, 'label'], axis=1)
    bytarget['is_valid'] = True
    bbb_train = pd.merge(train_df, bytarget, how='left', on='id')
    bbb_train.is_valid = bbb_train.is_valid.fillna(False)

    if not os.path.exists(os.path.join(pklpath, 'data_lm_smi.pkl')):
        logging.error('nlp lm data file not exists, please re-trained')
        exit(0)
    data_lm = load_data(Path(pklpath), 'data_lm_smi.pkl', bs=bs)
    vocab = data_lm.vocab

    imgList = ImageList.from_df(bbb_train, path=path, cols='PicturePath')
    tabList = TabularList.from_df(bbb_train, cat_names=[], cont_names=cont_names, procs=procs, path=path)
    textList = TextList.from_df(bbb_train, cols='smi', path=path, vocab=vocab)
    mixed = (MixedItemList([imgList, tabList, textList], path, inner_df=tabList.inner_df)
        .split_from_df(col='is_valid')
        .label_from_df(cols='label')
        .transform([[get_transforms()[0], [], []], [get_transforms()[1], [], []]], size=size)
    )
    data = mixed.databunch(bs=bs, collate_fn=model.collate_mixed)
    norm, denorm = model.normalize_custom_funcs(*imagenet_stats)
    data.add_tfm(norm)  # normalize images

    logging.info('save the mixed data file')
    outfile = os.path.join(pklpath, 'mixed_img_tab_text.pkl')
    outfile = open(outfile, 'wb')
    pickle.dump(mixed, outfile)
    outfile.close()

    test_df['is_valid'] = False

    bbbAll = pd.concat([bbb_train, test_df])
    data_text = (
        TextList.from_df(
            bbbAll, cols='smi', path=path, vocab=vocab
        )
    ).split_none(
    ).label_from_df(
        cols='label'
    ).databunch(bs=bs)

    loss_func = CrossEntropyFlat()
    learn = model.image_tabular_text_learner(data, len(cont_names), data_text, loss_func, vis_out, text_out)
    callbacks = [
        EarlyStoppingCallback(learn, min_delta=1e-5, patience=4),
        SaveModelCallback(learn)
    ]
    learn.callbacks = callbacks

    opt_func = partial(optim.Adam, lr=3e-5, betas=(0.9, 0.99), weight_decay=0.1, amsgrad=True)
    learn.opt_func = opt_func
    logging.info('create multi-model \n {0}'.format(learn.model))

    learn.lr_find()
    learn.recorder.plot(suggestion=True, skip=15)
    lr_v = learn.recorder.min_grad_lr
    logging.info('the best learn rate for multi-model is {0}'.format(lr_v))

    learn.freeze()
    learn.fit_one_cycle(50, lr_v)

    learn.load('bestmodel')
    learn.freeze_to(-1)
    learn.fit_one_cycle(50, slice(lr_v / (2.6 ** 4), lr_v))

    learn.load('bestmodel')
    learn.freeze_to(-2)
    learn.fit_one_cycle(50, slice(lr_v / 2 / (2.6 ** 4), lr_v / 2))

    learn.load('bestmodel')
    learn.freeze_to(-3)
    learn.fit_one_cycle(50, slice(lr_v / 4 / (2.6 ** 4), lr_v / 4))

    learn.load('bestmodel')
    learn.freeze_to(-4)
    learn.fit_one_cycle(50, slice(lr_v / 8 / (2.6 ** 4), lr_v / 8))

    learn.load('bestmodel')
    learn.freeze_to(-5)
    learn.fit_one_cycle(50, slice(lr_v / 8 / (2.6 ** 4), lr_v / 8))

    learn.load('bestmodel')
    learn.freeze_to(-6)
    learn.fit_one_cycle(50, slice(lr_v / 8 / (2.6 ** 4), lr_v / 8))

    learn.load('bestmodel')
    learn.freeze_to(-7)
    learn.fit_one_cycle(50, slice(lr_v / 8 / (2.6 ** 4), lr_v / 8))

    learn.load('bestmodel')
    learn.freeze_to(-8)
    learn.fit_one_cycle(50, slice(lr_v/8/(2.6**4),lr_v/8))

    learn.load('bestmodel')
    learn.unfreeze()
    learn.fit_one_cycle(50, slice(lr_v / 10 / (2.6 ** 4), lr_v / 10))
    #learn.save('deep-B3')

    exrpath = os.path.join(BASE_DIR, '{0}/{1}'.format('models', 'mixed-export.pkl'))
    logging.info('export deep-b3 model as {0}'.format(exrpath))
    learn.export(exrpath)
    #test_deep_b3(train_df, test_df)


def parse_args():
    parser = argparse.ArgumentParser(prog='Deep-B3')
    subparsers = parser.add_subparsers(dest="subcmd", help="Train or test the Deep-B3")

    parser_train = subparsers.add_parser('train', help='train a new Deep-B3')
    parser_test = subparsers.add_parser('test', help='test for Deep-B3')

    parser_train.add_argument('--train_feature', required=True, help='feature file for train model')
    parser_train.add_argument('--test_feature', required=True, help='feature file for test model')
    parser_train.add_argument('--bs', help="batch size")
    parser_train.add_argument('--vis_out', help="feature output from the CNN model for image")
    parser_train.add_argument('--text_out', help="feature output from the NLP model for SMILES")

    parser_test.add_argument('--feature', required=True, help='feature file for train model')

    args = parser.parse_args()
    if not args.subcmd:
        parser.print_help()
        exit(0)
    return args

if __name__ == '__main__':
    args = parse_args()
    set_random_seed(2022)
    BASE_DIR = os.path.abspath('.')
    if args.subcmd == 'train':
        logging.info('begin training a new deep-b3 model')
        logging.info('vision outputs feature is {0}'.format(args.vis_out))
        logging.info('text outputs feature is {0}'.format(args.text_out))
        logging.info('batch size is {0}'.format(args.bs))
        create_dirs('datapkl')
        train = os.path.join(os.path.join(BASE_DIR, 'train'), args.train_feature)
        test = os.path.join(os.path.join(BASE_DIR, 'test'), args.test_feature)
        train_df = read_csv_file(train)
        test_df = read_csv_file(test)
        pretrain_nlp(train_df, test_df)
        train_deep_b3(train_df, test_df, int(args.bs), int(args.vis_out), int(args.text_out))