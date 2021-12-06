"""
This notebook shows how to download and preprocess the CheXpert dataset for further analysis with weakly supervised experiments. 
The CheXpert training set is composed of chest radiographs, which were annotated on the basis of reports using the rule-based CheXpert labeler. 
Each image is labeled with respect 12 pathologies as well as the observations "No Finding" and "Support Devices". 
For each of these categories, except "No Finding", the assigned weak label is either: (Irvin et al. (2019))

- positive (1.0)
- negative (0.0)
- not mentioned (blank)
- uncertain (-1.0) 

The development set was annotated by radiologists and therefore only contains the binary labels: (Irvin et al. (2019))

- positive (1.0) 
- negative (0.0)

You can register for obtaining the data under the following link: https://stanfordmlgroup.github.io/competitions/chexpert/. 
Once the registration is finished, you should receive an email which contains links for two different versions of the dataset, the original CheXpert dataset (around 439 GB) and a version with downsampled resolution (around 11 GB). 
The code below uses the downsampled version. 
Please unzip the downloaded folder in a directory of your choice and don't change the filenames or the folder structure, otherwise you might need to change some of the paths used in the following code in order for it to run properly. 
The zip file you obtained should contain a training and a validation set. 
The CheXpert test set is not publicly available, as it is used for the CheXpert competition (see link above). 
The reports that were used to label the images are also unavailable.

The original CheXpert paper, "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison" by Irvin et al. (2019), can be found here: https://arxiv.org/abs/1901.07031.
"""

## Imports

import os
from typing import List

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tabulate import tabulate
from tqdm import tqdm

## Loading the dataset

"""
replace "data_path" with the path to the folder in which you stored train.csv and valid.csv
if you did not change the folder structure, this path should end with "\CheXpert-v1.0-small\CheXpert-v1.0-small"
"""

path = "data_path"
os.chdir(path) # change working directory to appropriate location

### Get train data

training_set = pd.read_csv('train.csv')

"""
first 5 entries of training set
note that this dataset has 4 possible labels:
    positive (1.0)
    negative (0.0)
    uncertain (-1.0)
    not-mentioned (blank), which is read in as NaN
"""
training_set.head(5)

print("Number of observations in training set:", training_set.shape[0])

### Get validation data

validation_set = pd.read_csv("valid.csv")

"""
first 5 entries of validation set
note that this dataset has 2 possible labels:
    positive (1.0)
    negative (0.0)
"""

validation_set.head(5)

print("Number of observations in validation set:", validation_set.shape[0])

### Collect statistics

"""
number of non-NaN labels in the training set
"""

training_labels = training_set.iloc[:, -13:-1]
labels_per_row = training_labels.count(axis = 1) # number of non-NaN labels per row in the training set

vals = pd.DataFrame(labels_per_row.value_counts())

# make a table
val_list = [(i, vals[0][i]) for i in vals.index]
    
print(tabulate(val_list, headers = ["Number of non-NaN labels", "Number of datapoints"]))

"""
label distribution in the training set
"""

val_list = []
for cond in training_labels.columns:
    vals = np.array(training_labels.value_counts(subset=[cond]).sort_index(ascending = True))
    val_list.append([cond, vals[0], vals[1], vals[2]])

print("Label distribution in the training set:", "\n")
print(tabulate(val_list, headers = ["Pathology", "-1.0", "0.0", "1.0"]))

"""
label distribution in the validation set
"""

validation_labels = validation_set.iloc[:, -13:-1]

val_list = []
for cond in validation_labels.columns:
    vals = np.array(validation_labels.value_counts(subset=[cond]).sort_index(ascending=True))
    if cond != "Fracture":
        val_list.append([cond, vals[0], vals[1]])
    else:
        val_list.append([cond, vals[0], 0]) # fracture never positively appears in validation set
        
print("Label distribution in the validation set:", "\n")
print(tabulate(val_list, headers = ["Pathology", "0.0", "1.0"]))

## Image preprocessing

# paths to training images
image_paths_train = [os.path.join(path[: path.find("CheXpert-v1.0-small")], "CheXpert-v1.0-small", p) for p in training_set["Path"]]

# paths to validation images
image_paths_valid = [os.path.join(path[: path.find("CheXpert-v1.0-small")], "CheXpert-v1.0-small", p) for p in validation_set["Path"]]

# sample image from training set
sample_image = Image.open(image_paths_train[0]).convert('RGB')
plt.imshow(sample_image)
plt.show()
print("Dimensions of image:", sample_image.size)

# define list of transformations that should be applied to the images
transform_list = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization from ImageNet
    ])

#class used for data loading and image preprocessing
#inspirations for class and transformations above:
     #https://github.com/gaetandi/cheXpert/blob/master/cheXpert_final.ipynb
     #https://github.com/Stomper10/CheXpert/blob/master/CheXpert_DenseNet121_FL.ipynb
     
class CheXpertDatasetProcessor(Dataset):
    
    def __init__(self, 
                 path: str = path,
                 subset: str = "train", 
                 image_paths: List[str] = image_paths_train, 
                 number_of_images: int = training_set.shape[0],  
                 transform_sequence: list = None,
                 to_ones: List[str] = None,
                 to_zeros: List[str] = None, 
                 to_ignore: List[str] = None):
        
        """
        Args:
            path: path to the folder where train.csv and valid.csv are stored
            subset: either "train" to load the train.csv or "valid" to load valid.csv
            image_paths: paths to the images
            number_of_images: number of images in the dataset
            transform_sequence: sequence used to transform the images
            to_ones: list of pathologies for which uncertainty labels should be replaced by 1
            to_zeros: list of pathologies for which uncertainty labels should be replaced by 0
            to_ignore: list of pathologies for which uncertainty labels should be ignored (label will be turned to nan)
        Returns: 
            224 x 224 image tensor and a corresponding tensor containing 12 labels
        """
        
        self.path = path
        self.subset = subset
        self.image_paths = image_paths
        self.number_of_images = number_of_images
        self.transform_sequence = transform_sequence
        self.to_ones = to_ones
        self.to_zeros = to_zeros
        self.to_ignore = to_ignore
        
    def process_chexpert_dataset(self):
        
        # read dataset
        if self.subset == "train":
            data = pd.read_csv("train.csv")
            
        elif self.subset == "valid":
            data = pd.read_csv("valid.csv")
            
        else:
            raise ValueError("Invalid subset, please choose either 'train' or 'valid'")
            
        pathologies = data.iloc[:, -13:-1].columns
        
        # prepare labels
        data.iloc[:, -13:-1] = data.iloc[:, -13:-1].replace(float("nan"), -1) # blank labels -> uncertain
        
        if self.to_ones:
            if all(p in pathologies for p in self.to_ones): # check whether arguments are valid pathologies
                data[self.to_ones] = data[self.to_ones].replace(-1, 1) # replace uncertainty labels with ones
            else:
                raise ValueError("List supplied to to_ones contains invalid pathology, please choose from:",
                                 list(pathologies))
            
        if self.to_zeros:
            if all(p in pathologies for p in self.to_zeros):
                    data[self.to_zeros] = data[self.to_zeros].replace(-1, 0) # replace uncertainty labels with zeros
            else:
                raise ValueError("List supplied to to_zeros contains invalid pathology, please choose from:",
                                 list(pathologies))
            
        if self.to_ignore:
            if all(p in pathologies for p in self.to_ignore):
                    data[self.to_ignore] = data[self.to_ignore].replace(-1, float("nan")) # replace uncertainty labels with nan
            else:
                raise ValueError("List supplied to to_ignore contains invalid pathology, please choose from:",
                                     list(pathologies))
        
        self.data = data
    
    def __getitem__(self, index: int, return_image: bool = True):
        
        """
        index: index of example that should be retrieved
        return_image: True: image tensor and labels are returned, False: only labels are returned
        """
        
        if return_image not in [True, False]:
            raise ValueError("Please set return_image argument either to True or False")
        
        image_labels = self.data.iloc[index, -13:-1]
        
        if return_image == False: # only labels are returned, not the images
            return torch.tensor(image_labels)
        
        else:
            image_name = self.image_paths[index]
        
            patient_image = Image.open(image_name).convert('RGB')
        
            if self.transform_sequence:
                patient_image = self.transform_sequence(patient_image) # apply the transform_sequence if one is specified
        
            else:
                # even if no other transformation is applied, the image should be turned into a tensor
                to_tensor = transforms.ToTensor()
                patient_image = to_tensor(patient_image)
            
            return patient_image, torch.tensor(image_labels)
    
    def __len__(self):
        return self.number_of_images
    
# prepare training data
chexpert_train = CheXpertDatasetProcessor(path = path, subset = "train", image_paths = image_paths_train, number_of_images = training_set.shape[0], transform_sequence = transform_list)
chexpert_train.process_chexpert_dataset()

# prepare validation data
chexpert_valid = CheXpertDatasetProcessor(path = path, subset = "valid", image_paths = image_paths_valid, number_of_images = validation_set.shape[0], transform_sequence = transform_list)
chexpert_valid.process_chexpert_dataset()

# example output for first training sample
chexpert_train.__getitem__(0)

## Store the data

"""
The following code can be used to store the preprocessed images and labels on your computer using joblib.
Please note that the resulting files can become very large, especially the one for the training set.
The code is commented out in case you do not wish to store the preprocessed data.
"""

#filename_train = 'chexpert_data_train_labels.joblib'
#outfile = open(filename_train,'wb')

#for i in tqdm(range(0, training_set.shape[0])):
    #joblib.dump(chexpert_train.__getitem__(i)[1], outfile)
    
#filename_valid = 'chexpert_data_valid_labels.joblib'
#outfile = open(filename_valid,'wb')

#for i in tqdm(range(0, validation_set.shape[0])):
    #joblib.dump(chexpert_valid.__getitem__(i)[1], outfile)
    
## Finish

"""
This concludes the preprocessing of the CheXpert data.
"""
    
### References

"""
CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison by Irvin et al. (2019): https://arxiv.org/abs/1901.07031
Structured dataset documentation: a datasheet for CheXpert by Garbin et al. (2021): https://arxiv.org/pdf/2105.03020.pdf
"""