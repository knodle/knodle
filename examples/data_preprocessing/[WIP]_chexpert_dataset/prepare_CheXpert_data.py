"""
This file shows how to download and preprocess the CheXpert dataset for making weakly supervised experiments.

The original CheXpert paper, "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison" by Irvin et al. (2019), can be found here: https://arxiv.org/pdf/1901.07031.pdf.

The CheXpert training set is composed of chest radiographs, which were annotated on the basis of reports using the rule-based CheXpert labeler. 
Each image is labeled with respect to 12 pathologies as well as the observations "No Finding" and "Support Devices". 
For each of these categories, except "No Finding", the assigned weak label is either: (Irvin et al. (2019))

positive (1.0)
negative (0.0)
not mentioned (blank)
uncertain (-1.0)
The development set was annotated by radiologists and therefore only contains the binary labels: (Irvin et al. (2019))

positive (1.0)
negative (0.0)

You can register for obtaining the data under the following link: https://stanfordmlgroup.github.io/competitions/chexpert/. Once the registration is finished, you should receive an email which contains links for two different versions of the dataset, the original CheXpert dataset (around 439 GB) and a version with downsampled resolution (around 11 GB). 
The code below uses the downsampled version. 
Please unzip the downloaded folder in a directory of your choice and don't change the filenames or the folder structure, otherwise you might need to change some of the paths used in the following code in order for it to run properly. 
The zip file you obtained should contain a training and a validation set. 
The CheXpert test set is not publicly available, as it is used for the CheXpert competition (see link above). 
The reports that were used to label the images are also unavailable.
"""

## Imports

import os
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

## Define storing locations for the preprocessed data

"""
If you wish to save the preprocessed data on your computer, please specify a path to the location where you want to store the data where "storing_location_path" is mentioned in the code underneath.

You will be presented two options of storing the preprocessed data:

- storing each preprocessed image tensor and its corresponding labels in a separate .npz file (~ 35 GB in total)
- storing each preprocessed image as a .jpg file and saving all of the labels in a joblib file (~ 2 GB in total)

Please note that in both approaches, the training and the validation set will be stored separately. 
If you wish to store the data, please create two folders, named "train_images" and "valid_images" respectively, in your specified location.
"""

storing_location = "storing_location_path"

# joblib files in which labels are stored if second option is chosen
joblib_labels_train = os.path.join(storing_location, 'chexpert_data_train_labels.joblib')
joblib_labels_valid = os.path.join(storing_location, 'chexpert_data_valid_labels.joblib')

## Load the dataset

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

"""
first 5 entries of validation set
note that this dataset has 2 possible labels:
    positive (1.0)
    negative (0.0)
"""

validation_set = pd.read_csv("valid.csv")

validation_set.head(5)

print("Number of observations in validation set:", validation_set.shape[0])

### Collect statistics

"""
number of non-NaN labels in the training set
"""

training_labels = training_set.iloc[:, -13:-1]
labels_per_row = training_labels.count(axis=1) # number of non-NaN labels per row in the training set

vals = pd.DataFrame(labels_per_row.value_counts())

# make a table
val_list = [(i, vals[0][i]) for i in vals.index]
    
print(tabulate(val_list, headers=["Number of non-NaN labels", "Number of datapoints"]))

"""
label distribution in the training set
"""

val_list = []
for cond in training_labels.columns:
    vals = np.array(pd.Categorical(training_labels[cond], categories=[-1.0, 0.0, 1.0]).value_counts().sort_index(ascending=True))
    val_list.append([cond, vals[0], vals[1], vals[2]])

print("Label distribution in the training set:", "\n")
print(tabulate(val_list, headers=["Pathology", "-1.0", "0.0", "1.0"]))

"""
label distribution in the validation set
"""

validation_labels = validation_set.iloc[:, -13:-1]

val_list = []
for cond in validation_labels.columns:
    vals = np.array(pd.Categorical(validation_labels[cond], categories=[0.0, 1.0]).value_counts().sort_index(ascending=True))
    val_list.append([cond, vals[0], vals[1]])
        
print("Label distribution in the validation set:", "\n")
print(tabulate(val_list, headers=["Pathology", "0.0", "1.0"]))

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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization from ImageNet
    ])

# class used for data loading and image preprocessing
# inspirations for class and transformations above:
     # https://github.com/gaetandi/cheXpert/blob/master/cheXpert_final.ipynb
     # https://github.com/Stomper10/CheXpert/blob/master/CheXpert_DenseNet121_FL.ipynb
     
class CheXpertDatasetProcessor():
    
    def __init__(self, 
                 path: str,
                 subset: str, 
                 image_paths: List[str], 
                 number_of_images: int,  
                 transform_sequence: List = None,
                 to_ones: List[str] = None,
                 to_zeros: List[str] = None, 
                 to_ignore: List[str] = None,
                 return_image: bool = True):
        
        """
        Args:
            path: path to the folder where train.csv and valid.csv are stored
            subset: "train": load train.csv, "valid": load valid.csv
            image_paths: paths to the images
            number_of_images: number of images in the dataset
            transform_sequence: sequence used to transform the images
            to_ones: list of pathologies for which uncertainty labels should be replaced by 1.0
            to_zeros: list of pathologies for which uncertainty labels should be replaced by 0.0
            to_ignore: list of pathologies for which uncertainty labels should be ignored (label will be turned to nan)
            return_image: True: image tensor and labels are returned, False: only labels are returned
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
        self.return_image = return_image
        
    def process_chexpert_dataset(self):
        
        # read in dataset
        if self.subset == "train":
            data = pd.read_csv("train.csv")
            
        elif self.subset == "valid":
            data = pd.read_csv("valid.csv")
            
        else:
            raise ValueError("Invalid subset, please choose either 'train' or 'valid'")
            
        pathologies = data.iloc[:, -13:-1].columns
        
        # prepare labels
        data.iloc[:, -13:-1] = data.iloc[:, -13:-1].replace(float("nan"), -1) # blank labels -> uncertain
        
        if self.to_ones is not None:
            if all(p in pathologies for p in self.to_ones): # check whether arguments are valid pathologies
                data[self.to_ones] = data[self.to_ones].replace(-1, 1) # replace uncertainty labels with ones
            else:
                raise ValueError("List supplied to to_ones contains invalid pathology, please choose from:",
                                 list(pathologies))
            
        if self.to_zeros is not None:
            if all(p in pathologies for p in self.to_zeros):
                    data[self.to_zeros] = data[self.to_zeros].replace(-1, 0) # replace uncertainty labels with zeros
            else:
                raise ValueError("List supplied to to_zeros contains invalid pathology, please choose from:",
                                 list(pathologies))
            
        if self.to_ignore is not None:
            if all(p in pathologies for p in self.to_ignore):
                    data[self.to_ignore] = data[self.to_ignore].replace(-1, float("nan")) # replace uncertainty labels with nan
            else:
                raise ValueError("List supplied to to_ignore contains invalid pathology, please choose from:",
                                     list(pathologies))
        
        self.data = data
    
    def __getitem__(self, index: int):
        
        """
        index: index of example that should be retrieved
        """
        
        if self.return_image not in [True, False]:
            raise ValueError("Please set return_image argument either to True or False")
        
        image_labels = self.data.iloc[index, -13:-1]
        
        if self.return_image is False: # only labels are returned, not the images
            return torch.tensor(image_labels)
        
        else:
            image_name = self.image_paths[index]
        
            patient_image = Image.open(image_name).convert('RGB')
        
            if self.transform_sequence is not None:
                patient_image = self.transform_sequence(patient_image) # apply the transform_sequence if one is specified
        
            else:
                # even if no other transformation is applied, the image should be turned into a tensor
                to_tensor = transforms.ToTensor()
                patient_image = to_tensor(patient_image)
            
            return patient_image, torch.tensor(image_labels)
    
    def __len__(self):
        return self.number_of_images
    
# prepare training data
chexpert_train = CheXpertDatasetProcessor(path=path, subset="train", image_paths=image_paths_train,
                                          number_of_images=training_set.shape[0], transform_sequence=transform_list)
chexpert_train.process_chexpert_dataset()

# prepare validation data
chexpert_valid = CheXpertDatasetProcessor(path=path, subset="valid", image_paths=image_paths_valid,
                                          number_of_images=validation_set.shape[0], transform_sequence=transform_list)
chexpert_valid.process_chexpert_dataset()

# example output for first training sample
chexpert_train.__getitem__(0)

shape_of_image_tensor = chexpert_train.__getitem__(0)[0].shape
shape_of_label_tensor = chexpert_train.__getitem__(0)[1].shape

print("Shape of image tensor:", shape_of_image_tensor)
print("Shape of label tensor:", shape_of_label_tensor)

# example using to_ones, to_zeros and to_ignore
all_pathologies = training_set.iloc[:, -13:-1].columns

chexpert_train_alt = CheXpertDatasetProcessor(path=path, subset="train",image_paths=image_paths_train,
                                          number_of_images=training_set.shape[0], transform_sequence=transform_list,
                                          to_ones=all_pathologies[0:2],
                                          to_zeros=all_pathologies[2:4],
                                          to_ignore=all_pathologies[4:6],
                                          return_image=False)

chexpert_train_alt.process_chexpert_dataset()
chexpert_train_alt.__getitem__(0)

## Store the preprocessed data

"""
The two different options of storing the data provided in this tutorial are:

- storing each preprocessed image tensor and its corresponding labels separate .npz file (~ 35 GB in total)
- storing each resized image as a .jpg file and saving all of the labels in a joblib file (~ 2 GB in total)

The code for each option is commented out, in case you do not wish to store the data
"""

### Store data as .npz files

"""
If you prefer storing the images and labels in .npz files, please run the following code.
"""

# =============================================================================
# # store the training set
# for i in tqdm(range(0, training_set.shape[0])):
#     x = chexpert_train.__getitem__(i)
#     np.savez_compressed(os.path.join(storing_location, "train_images", "image_" + str(i) + ".npz"), 
#                         image=x[0], label=x[1])
#     
# # store the validation set
# for i in tqdm(range(0, validation_set.shape[0])):
#     x = chexpert_valid.__getitem__(i)
#     np.savez_compressed(os.path.join(storing_location, "valid_images", "image_" + str(i) + ".npz"), 
#                         image=x[0], label=x[1])
# =============================================================================
    
"""
Please note that the `np.load` function returns an array and not a tensor, so the "image" result has to be transposed and turned into a tensor again.
The labels also need to be converted to a tensor again. 
Here is a quick example how to do this:
"""

# =============================================================================
# to_tensor = transforms.ToTensor()
# 
# example = np.load(os.path.join(storing_location, "train_images", "image_" + str(0) + ".npz"))
# print(to_tensor(example["image"].transpose(1,2,0)))
# print(torch.tensor(example["label"]))
# 
# # compare with original output from __getitem__()
# print(chexpert_train.__getitem__(0)) # same result
# =============================================================================

### Store images as .jpg files

"""
If you want to store the resized images as .jpg files and the labels in joblib files, you can run the code below.
Please note that the normalization that we applied earlier results in negative values in the image tensors, which are not compatible with the `save_image` function that we use to store the images.
The images are therefore saved without the normalization.
"""

# =============================================================================
# # define transformations (without normalization)
# 
# transform_list_resize = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     ])
# 
# # store the training images
# chexpert_train_resize = CheXpertDatasetProcessor(path=path, subset="train", image_paths=image_paths_train, number_of_images=training_set.shape[0], transform_sequence=transform_list_resize)
# chexpert_train_resize.process_chexpert_dataset()
# 
# for i in tqdm(range(0, training_set.shape[0])):
#     ex = chexpert_train_resize.__getitem__(i)[0]
#     save_image(ex, os.path.join(storing_location, "train_images", "image_" + str(i) + ".jpg"))
#     
# # store the training labels
# joblib_file = open(joblib_labels_train, 'wb')
# 
# chexpert_train_labels = CheXpertDatasetProcessor(path=path, subset="train", image_paths=image_paths_train, number_of_images=training_set.shape[0], return_image=False)
# chexpert_train_labels.process_chexpert_dataset()
# train_label_loader = DataLoader(chexpert_train_labels, batch_size=training_set.shape[0])
# dataiter = iter(train_label_loader)
# train_labels = dataiter.next()
# joblib.dump(train_labels, joblib_file, compress=2)
# 
# # store the validation images
# chexpert_valid_resize = CheXpertDatasetProcessor(path=path, subset="valid", image_paths=image_paths_valid, number_of_images=validation_set.shape[0], transform_sequence=transform_list_resize)
# chexpert_valid_resize.process_chexpert_dataset()
# 
# for i in tqdm(range(0, validation_set.shape[0])):
#     ex = chexpert_valid_resize.__getitem__(i)[0]
#     save_image(ex, os.path.join(storing_location, "valid_images", "image_" + str(i) + ".jpg"))
#     
# # store the validation labels
# 
# joblib_file = open(joblib_labels_valid, 'wb')
# 
# chexpert_valid_labels = CheXpertDatasetProcessor(path=path, subset="valid", image_paths=image_paths_valid, number_of_images=validation_set.shape[0], return_image=False)
# chexpert_valid_labels.process_chexpert_dataset()
# valid_label_loader = DataLoader(chexpert_valid_labels, batch_size=validation_set.shape[0])
# dataiter = iter(valid_label_loader)
# valid_labels = dataiter.next()
# joblib.dump(valid_labels, joblib_file, compress=2)
# =============================================================================

## Finish

"""
This concludes the preprocessing of the CheXpert data.
"""

### References

"""
CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison by Irvin et al. (2019): https://arxiv.org/abs/1901.07031
Structured dataset documentation: a datasheet for CheXpert by Garbin et al. (2021): https://arxiv.org/pdf/2105.03020.pdf
"""