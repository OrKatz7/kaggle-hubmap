import sys
sys.path.insert(0, '../')
import warnings
warnings.simplefilter('ignore')
from utils import fix_seed
from get_config import get_config
from get_fold_idxs_list import get_fold_idxs_list
from sklearn.model_selection import train_test_split
from run import run
import pickle

import numpy as np
import pandas as pd
import os
from os.path import join as opj
import glob

def foo(row):
    return f"{row:05d}"

if __name__=='__main__':
    # config
    fix_seed(2021)
    config = get_config()
    FOLD_LIST = config['FOLD_LIST']
    VERSION = config['VERSION']
    INPUT_PATH = config['INPUT_PATH']
    OUTPUT_PATH = config['OUTPUT_PATH']
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    device = config['device']
    print(device)
    
#     # import data 
#     train_df = pd.read_csv(opj(INPUT_PATH, 'train.csv'))
#     info_df  = pd.read_csv(opj(INPUT_PATH,'HuBMAP-20-dataset_information.csv'))
#     sub_df = pd.read_csv(opj(INPUT_PATH, 'sample_submission.csv'))
#     print('train_df.shape = ', train_df.shape)
#     print('info_df.shape  = ', info_df.shape)
#     print('sub_df.shape = ', sub_df.shape)
    
    # dataset
#     data_df = []
#     for data_path in config['train_data_path_list']:
#         _data_df = pd.read_csv(opj(data_path,'data.csv'))
#         _data_df['data_path'] = data_path
#         data_df.append(_data_df)
#     data_df = pd.concat(data_df, axis=0).reset_index(drop=True)

#     print('data_df.shape = ', data_df.shape)
#     data_df = data_df[data_df['std_img']>10].reset_index(drop=True)
#     print('data_df.shape = ', data_df.shape)
#     data_df['binned'] = np.round(data_df['ratio_masked_area'] * config['multiplier_bin']).astype(int)
#     data_df['is_masked'] = data_df['binned']>0

#     trn_df = data_df.copy()
#     trn_df['binned'] = trn_df['binned'].apply(lambda x:config['binned_max'] if x>=config['binned_max'] else x)
#     trn_df_1 = trn_df[trn_df['is_masked']==True]
#     print(trn_df['is_masked'].value_counts())
#     print(trn_df_1['binned'].value_counts())
#     print('mean = ', int(trn_df_1['binned'].value_counts().mean()))
    
#     info_df['image_name'] = info_df['image_file'].apply(lambda x:x.split('.')[0])
#     patient_mapper = {}
#     for (x,y) in info_df[['image_name','patient_number']].values:
#         patient_mapper[x] = y
#     data_df['patient_number'] = data_df['filename_img'].apply(lambda x:patient_mapper[x.split('_')[0]])
    
#     val_patient_numbers_list = [
#         [63921], # fold0
#         [68250], # fold1
#         [65631], # fold2
#         [67177], # fold3
#     ]
    
    # train
    
    for seed in config['split_seed_list']:
#         trn_idxs_list = [glob.glob(INPUT_PATH.format("train"))[0:100]]
#         val_idxs_list = [glob.glob(INPUT_PATH.format("val"))[0:100]]
        
        trn_idxs_list = [glob.glob(INPUT_PATH.format("train"))]
        val_idxs_list = [glob.glob(INPUT_PATH.format("val"))]
        
#         df = pd.read_csv("../input/rsna2021-k-fold-split/train_kfold.csv")
#         df.BraTS21ID = df.BraTS21ID.apply(foo)
#         main_list = [row.split("/")[-1].split("_")[0] for row in glob.glob("../input/rsna21pseudolabel/rsna_pseudo_step1/*.npz")]
#         main_fold = [df[df.BraTS21ID==row]['fold'].values[0] for row in main_list]
#         trn_idxs_list = []
#         val_idxs_list = []
#         for f in range(5):
#             trn_list = []
#             val_list = []
#             for row,col in zip(main_list,main_fold):
#                 if col ==f:
#                     val_list.append(row)
#                 else:
#                     trn_list.append(row)
#             trn_idxs_list.append(trn_list)
#             val_idxs_list.append(val_list)
        
        
        
#         print(trn_idxs_list[0][0:10])
#         print(val_idxs_list[0][0:10])
        with open(opj(config['OUTPUT_PATH'],f'trn_idxs_list_seed{seed}'), 'wb') as f:
            pickle.dump(trn_idxs_list, f)
        with open(opj(config['OUTPUT_PATH'],f'val_idxs_list_seed{seed}'), 'wb') as f:
            pickle.dump(val_idxs_list, f)
        run(seed, None, None, trn_idxs_list, val_idxs_list)
        
    # score
    score_list  = []
    for seed in config['split_seed_list']:
        for fold in config['FOLD_LIST']:
            log_df = pd.read_csv(opj(config['OUTPUT_PATH'],f'log_seed{seed}_fold{fold}.csv'))
            score_list.append(log_df['val_score'].max())
    print('CV={:.4f}'.format(sum(score_list)/len(score_list)))
