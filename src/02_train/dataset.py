import numpy as np
import pickle
import cv2
from os.path import join as opj
from torch.utils.data import Dataset
from transforms import get_transforms_train, get_transforms_valid
from utils import rle2mask

class RSNADatasetTrain(Dataset):
    def __init__(self, paths, config, mode='train'): #add df
        self.paths = paths
#         self.df = df
        self.config = config
        if mode=='train':
            self.transforms = get_transforms_train()
        else:
            self.transforms = get_transforms_valid()
        self.h, self.w = self.config['input_resolution']
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,idx):
        p = self.paths[idx]
#         name = str(int(p.split("/")[-1].split("_")[0]))
#         MGMT_value = int(self.df[self.df.BraTS21ID==name]['MGMT_value'].values[0])+1
        data = np.load(p)
        img = data['img'].astype(np.uint8)#[:,:,:3]
        img = cv2.resize(img, (self.w,self.h), interpolation=cv2.INTER_AREA)
        mask =  data['mask'].astype(np.uint8)
        mask = cv2.resize(mask, (self.w,self.h), interpolation=cv2.INTER_AREA)
        #mask /=255.0
        mask = (mask>0.5).astype(np.int8)
        if self.transforms:
            augmented = self.transforms(image=img.astype(np.uint8), 
                                        mask=mask.astype(np.int8))
        img  = augmented['image']
        mask = augmented['mask']
        img = img.transpose(2,0,1)
        mask = mask.transpose(2,0,1) 
        label = data['label'].astype(float) #
        #label = mask.max() * MGMT_value
        
        return {'img':img,'mask':mask,'label':label}
