"""
This file shows how to download and preprocess the CheXpert dataset for further analysis with gradient matching. 
The CheXpert training set is composed of chest radiographs, which were annotated on the basis of reports using the rule-based CheXpert labeler. 
Each image is labeled with respect 12 pathologies as well as the observations "No Finding" and "Support Devices". 
For each of these categories, except "No Finding", the assigned weak label is either positive (1.0), negative (0.0), not mentioned (blank) or uncertain (-1.0).
The development set was annotated by radiologists and therefore only contains the binary labels positive (1.0) and negative (0.0). (Irvin et al. (2019))

You can register for obtaining the data under the following link: https://stanfordmlgroup.github.io/competitions/chexpert/. 
Once the registration is finished, you should receive an email which contains links for two different versions of the dataset, the original CheXpert dataset (around 439 GB) and a version with downsampled resolution (around 11 GB). 
The code below uses the downsampled version. 
Please unzip the downloaded folder in a directory of your choice and don't change the filenames or the folder structure, otherwise you might need to change some of the paths used in the following code in order for it to run properly. 
The zip file you obtained should contain a training and a validation set. 
The CheXpert test set is not publicly available, as it is used for the CheXpert competition (see link above). 
The reports that were used to label the images are also unavailable.
"""

## Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torchvision.transforms as transforms
import joblib
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tabulate import tabulate

## Loading the dataset

"""
read the dataset; replace PATH with the path to the folder in which you stored train.csv and valid.csv
if you did not change the folder structure, this path should end with "\CheXpert-v1.0-small\CheXpert-v1.0-small"
"""

path = PATH
os.chdir(path) #change working directory to appropriate location
print(path)
training_set = pd.read_csv("train.csv")

validation_set = pd.read_csv("valid.csv")

"""
first 5 entries of training set
note that this dataset has 4 possible labels:
    positive (1.0)
    negative (0.0)
    uncertain (-1.0)
    not-mentioned (blank), which is read in as NaN
223414 observations in entire training set
"""
training_set.head(5)

"""
first 5 entries of validation set
note that this dataset has 2 possible labels:
    positive (1.0)
    negative (0.0)
234 observations in entire validation set
"""
validation_set.head(5)

"""
in the following we look at some dataset statistics
"""

training_labels = training_set.iloc[:, -13:-1]
labels_per_row = training_labels.count(axis = 1) #number of non-NaN labels per row in the training set

vals = pd.DataFrame(labels_per_row.value_counts())

#make a table
val_list = []
for i in vals.index:
    val_list.append([i, vals[0][i]])
    
print(tabulate(val_list, headers = ["Number of non-NaN labels", "Number of datapoints"]))

#label distribution in training set
val_list = []
for cond in training_labels.columns:
    vals = np.array(training_labels.value_counts(subset=[cond]).sort_index(ascending = True))
    val_list.append([cond, vals[0], vals[1], vals[2]])

print("Label distribution in the training set:", "\n")
print(tabulate(val_list, headers = ["Pathology", "-1.0", "0.0", "1.0"]))

#label distribution in validation set
validation_labels = validation_set.iloc[:, -13:-1]

val_list = []
for cond in validation_labels.columns:
    vals = np.array(validation_labels.value_counts(subset=[cond]).sort_index(ascending=True))
    if cond != "Fracture":
        val_list.append([cond, vals[0], vals[1]])
    else:
        val_list.append([cond, vals[0], 0]) #fracture never positively appears in validation set
        
print("Label distribution in the validation set:", "\n")
print(tabulate(val_list, headers = ["Pathology", "0.0", "1.0"]))

#sample image from training set
image_paths_sample = path[: path.find("CheXpert-v1.0-small")] + "CheXpert-v1.0-small/" + training_set["Path"]

sample_image = Image.open(image_paths_sample[0]).convert('RGB')
plt.imshow(sample_image)
plt.show()
print("Dimensions of image:", sample_image.size)

#define list of transformations that should be applied to the images
transform_list = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #normalization from ImageNet
    ])

#class used for data loading and image preprocessing
#inspirations for class and transformations above:
     #https://github.com/gaetandi/cheXpert/blob/master/cheXpert_final.ipynb
     #https://github.com/Stomper10/CheXpert/blob/master/CheXpert_DenseNet121_FL.ipynb
class prepare_CheXpert(Dataset):
    
    def __init__(self, path, subset = "train", to_ones = None, to_zeros = None, to_ignore = None, transform_sequence = None):
        
        """
        path: path to the folder where train.csv and valid.csv are stored
        subset: either "train" to load the train.csv or "valid" to load valid.csv
        to_ones: list of pathologies for which uncertainty labels should be replaced by 1
        to_zeros: list of pathologies for which uncertainty labels should be replaced by 0
        to_ignore: list of pathologies for which uncertainty labels should be ignored (label will be turned to nan)
        transform_sequence: sequence used to transform the images
        """
        
        #read the dataset
        if subset == "train":
            data = pd.read_csv("train.csv")
            
        elif subset == "valid":
            data = pd.read_csv("valid.csv")
            
        else:
            raise ValueError("Invalid subset, please choose either 'train' or 'valid'")
            
        pathologies = data.iloc[:, -13:-1].columns
        
        #prepare the labels
        data.iloc[:, -13:-1] = data.iloc[:, -13:-1].replace(float("nan"), -1) #blank labels -> uncertain
        
        if to_ones is not None:
            if all(p in pathologies for p in to_ones): #check whether arguments are valid pathologies
                data[to_ones] = data[to_ones].replace(-1, 1) #replace uncertainty labels with ones
            else:
                raise ValueError("List supplied to to_ones contains invalid pathology, please choose from:",
                                 list(pathologies))
            
        if to_zeros is not None:
            if all(p in pathologies for p in to_zeros):
                data[to_zeros] = data[to_zeros].replace(-1, 0) #replace uncertainty labels with zeros
            else:
                raise ValueError("List supplied to to_zeros contains invalid pathology, please choose from:",
                                 list(pathologies))
            
        if to_ignore is not None:
            if all(p in pathologies for p in to_ignore):
                data[to_ignore] = data[to_ignore].replace(-1, float("nan")) #replace uncertainty labels with nan
            else:
                raise ValueError("List supplied to to_ignore contains invalid pathology, please choose from:",
                                 list(pathologies))
        
        #path to access the pictures (for small dataset)
        image_paths = path[: path.find("CheXpert-v1.0-small")] + "CheXpert-v1.0-small/" + data["Path"]
        
        number_of_images = len(data)
            
        self.image_paths = image_paths
        self.number_of_images = number_of_images
        self.transform_sequence = transform_sequence
        self.data = data
    
    def __getitem__(self, index = None):
        
        """
        index: index of example that should be retrieved
        """
        
        image_name = self.image_paths[index]
        
        image_labels = self.data.iloc[index, -13:-1]
        
        patient_image = Image.open(image_name).convert('RGB')
        
        if self.transform_sequence is not None:
            patient_image = self.transform_sequence(patient_image) #apply the transform_sequence if one is specified
        
        else:
            #even if no other transformation is applied, the image should be turned into a tensor
            to_tensor = transforms.ToTensor()
            patient_image = to_tensor(patient_image)
        
        return patient_image, torch.tensor(image_labels)
    
    def __len__(self):
        return self.number_of_images
    
chexpert_train = prepare_CheXpert(path, transform_sequence = transform_list)
chexpert_valid = prepare_CheXpert(path, subset = "valid", transform_sequence = transform_list)

#example output for first training sample
print(chexpert_train.__getitem__(0))
shape_of_image_tensor = chexpert_train.__getitem__(0)[0].shape
shape_of_label_tensor = chexpert_train.__getitem__(0)[1].shape

print("Shape of image tensor:", shape_of_image_tensor)
print("Shape of label tensor:", shape_of_label_tensor)

## Storing the data

"""
The following code can be used to store the preprocessed images and labels on your computer using joblib.
Please note that the resulting files can become very large, especially the one for the training set.
The code is commented out in case you do not wish to store the preprocessed data.
"""

#filename = 'chexpert_data_train.joblib'
#outfile = open(filename,'wb')

#for i in tqdm(range(0, training_set.shape[0])):
    #joblib.dump(chexpert_train.__getitem__(i), outfile)
    
#filename = 'chexpert_data_valid.joblib'
#outfile = open(filename,'wb')

#for i in tqdm(range(0, validation_set.shape[0])):
    #joblib.dump(chexpert_valid.__getitem__(i), outfile)
    
## References

"""
CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison by Irvin et al. (2019): https://arxiv.org/abs/1901.07031
Structured dataset documentation: a datasheet for CheXpert by Garbin et al. (2021): https://arxiv.org/pdf/2105.03020.pdf
"""