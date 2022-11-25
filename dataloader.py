#패키지
import SimpleITK as sitk
import pydicom as dcm
import numpy as np
from scipy import ndimage
from sklearn.model_selection import train_test_split
import pandas as pd
import preprocessing as ppc
import torch

def read_dicom_file(filepath):
    """Read and load volume"""
    try:
        sitk_t1 = sitk.ReadImage(filepath)
        sitk_t1.SetDirection([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0])
        image = sitk.GetArrayFromImage(sitk_t1)
        image = image.transpose(1,2,0)
        #image = ndimage.rotate(image,270,reshape=False)
    except:
        print(filepath)

    return image

def preprocessing(image):
    image = ppc.window_image(image,40,400)
    image = ppc.crop_image(image)
    image = ppc.add_pad(image)
    image = ppc.resize(image,desired_width=128,desired_height=128,desired_depth=128)
    #image = ppc.z_score(image)
    image = ppc.minmax(image,min_percentile = 5,max_percentile=95)
    return image

def process_scan(filepath, preprocess= True):
    image = read_dicom_file(filepath)
    if preprocess == True:
        image = preprocessing(image)
    return image

def load_dataset(df_dataset,preprocess = True):
    
    img_dataset = np.array([process_scan(source,path,preprocess) for source,path in np.array(df_dataset[['source','path']])])
    
    return img_dataset

def dataset_split(df_dataset,test_size=0.2,shuffle=True,grp=None,seed=1004):
    df_dataset['grp'] = (df_dataset['source'].str.replace('OASIS-3','1').str.replace('ADNI','2').apply(pd.to_numeric)*1000 
                     + df_dataset['sex'].str.replace('F','1').str.replace('M','2').apply(pd.to_numeric)*100
                     + df_dataset['group_maxinc'].str.replace('CN','1').str.replace('MCI','2').str.replace('AD','3').apply(pd.to_numeric)*10
                     )
                     
    X = df_dataset.drop(labels='group_maxinc',axis=1)
    Y = df_dataset['group_maxinc']
    if grp == None:
        grp = df_dataset['grp']

    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=test_size,shuffle=shuffle,stratify=grp,random_state=seed)

    return X_train,X_test,y_train,y_test


#Dataset Class 
from torch.utils.data import Dataset
class CTDataset(Dataset):
    def __init__(self, X_path,labels,transform=None):
        self.df_train = X_path
        self.df_labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.df_train)

    def __getitem__(self, idx):
        img_path = self.df_train[idx]

        channels = []
        image = process_scan(img_path)
        channels.append(image)
        images = np.array(channels)
        label = self.df_labels[idx]
        
        if self.transform:
            images = self.transform(images)
            
        return torch.tensor(images).float(), torch.tensor(label)