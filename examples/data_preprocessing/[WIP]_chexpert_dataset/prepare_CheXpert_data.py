"""
This file shows how to download and preprocess the CheXpert dataset for making weakly supervised experiments.

The original CheXpert paper, "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison" by Irvin et al. (2019), can be found here: https://arxiv.org/pdf/1901.07031.pdf.

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

You can register for obtaining the data under the following link: https://stanfordmlgroup.github.io/competitions/chexpert/.
Once the registration is finished, you should receive an email which contains links for two different versions of the dataset, the original CheXpert dataset (around 439 GB) and a version with downsampled resolution (around 11 GB). 
The code below uses the downsampled version. 
Please unzip the downloaded folder in a directory of your choice and don't change the filenames or the folder structure, otherwise you might need to change some of the paths used in the following code in order for it to run properly.
The zip file you obtained should contain a training and a validation set. 
The CheXpert test set is not publicly available, as it is used for the CheXpert competition (see link above). 
The reports that were used to label the images are also unavailable.
"""

## Imports

import copy
import os
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchvision.utils import save_image

## Define storing locations for the preprocessed data

"""
If you wish to save the preprocessed data on your computer, please specify a path to the location where you want to store the data where "storing_location_path" is mentioned in the code underneath.

You will be presented two options of storing the preprocessed data:

- storing each preprocessed image tensor with its corresponding labels in a separate .npz file (~ 35 GB in total)
- storing each preprocessed image as a .jpg file and saving all of the labels in a .lib file (~ 2 GB in total)

Please note that in both approaches, the training and the validation set will be stored separately. 
If you wish to store the data, please create two folders, named "train_images" and "valid_images" respectively, in your specified location.
"""

storing_location = "storing_location_path"

# joblib files in which labels are stored if second option is chosen
joblib_labels_train = os.path.join(storing_location, 'chexpert_data_train_labels.lib')
joblib_labels_valid = os.path.join(storing_location, 'chexpert_data_valid_labels.lib')

os.makedirs(os.path.join(storing_location, "train_images"), exist_ok=True)
os.makedirs(os.path.join(storing_location, "valid_images"), exist_ok=True)

## Load the dataset

"""
Please replace "data_path" with the path to the folder in which you stored train.csv and valid.csv.
If you did not change the folder structure, this path should end with "\CheXpert-v1.0-small\CheXpert-v1.0-small".
"""

path = "data_path"
os.chdir(path) # change working directory to appropriate location

### Get train data

training_set = pd.read_csv('train.csv')

"""
first 5 entries of training set
Note that this dataset has 4 possible labels:
    positive (1.0)
    negative (0.0)
    uncertain (-1.0)
    not-mentioned (blank), which is read in as NaN
"""

training_set.head(5)

print("Number of observations in training set:", training_set.shape[0], "\n")

### Get validation data

"""
first 5 entries of validation set
Note that this dataset has 2 possible labels:
    positive (1.0)
    negative (0.0)
"""

validation_set = pd.read_csv("valid.csv")

validation_set.head(5)

print("Number of observations in validation set:", validation_set.shape[0], "\n")

### Collect statistics

"""
number of non-NaN labels in the training set
"""

training_labels = training_set.iloc[:, -13:-1] # columns that contain the labels for the pathologies
labels_per_row = training_labels.count(axis=1) # number of non-NaN labels per row in the training set

vals = pd.DataFrame(labels_per_row.value_counts())

# make a table
val_list = [(i, vals[0][i]) for i in vals.index]
    
print(tabulate(val_list, headers=["Number of non-NaN labels", "Number of datapoints"]), "\n")

"""
number of positive labels in the training set
"""

labels_per_row = training_labels.eq(1).sum(axis=1) # number of positive labels per row in the training set

vals = pd.DataFrame(labels_per_row.value_counts())

# make a table
val_list = [(i, vals[0][i]) for i in vals.index]
    
print(tabulate(val_list, headers=["Number of positive labels", "Number of datapoints"]), "\n")

"""
label distribution in the training set
"""

val_list = []
for cond in training_labels.columns:
    vals = np.array(pd.Categorical(training_labels[cond], categories=[-1.0, 0.0, 1.0]).value_counts().sort_index(ascending=True))
    val_list.append([cond, vals[0], vals[1], vals[2]])

print("Label distribution in the training set:", "\n")
print(tabulate(val_list, headers=["Pathology", "-1.0", "0.0", "1.0"]), "\n")

"""
label distribution in the validation set
"""

validation_labels = validation_set.iloc[:, -13:-1]

val_list = []
for cond in validation_labels.columns:
    vals = np.array(pd.Categorical(validation_labels[cond], categories=[0.0, 1.0]).value_counts().sort_index(ascending=True))
    val_list.append([cond, vals[0], vals[1]])
        
print("Label distribution in the validation set:", "\n")
print(tabulate(val_list, headers=["Pathology", "0.0", "1.0"]), "\n")

## Image preprocessing

splitted_path = os.path.split(path)[0]

# paths to training images
image_paths_train = [os.path.join(splitted_path,p) for p in training_set["Path"]]

# paths to validation images
image_paths_valid = [os.path.join(splitted_path,p) for p in validation_set["Path"]]

# sample image from training set
sample_image = Image.open(image_paths_train[0]).convert('RGB')
plt.imshow(sample_image)
plt.show()
print("Dimensions of image:", sample_image.size, "\n")

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
                 transform_sequence: object = None,
                 to_ones: List[str] = None,
                 to_zeros: List[str] = None, 
                 to_ignore: List[str] = None,
                 replacement_for_blank: int = -1,
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
            replacement_for_blank: value that should be used to replace the "blank" labels
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
        self.replacement_for_blank = replacement_for_blank
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
        data.iloc[:, -13:-1] = data.iloc[:, -13:-1].replace(float("nan"), self.replacement_for_blank) # blank labels -> specified value
        
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
            return torch.FloatTensor(image_labels)
        
        else:
            image_name = self.image_paths[index]
        
            patient_image = Image.open(image_name).convert('RGB')
        
            if self.transform_sequence is not None:
                patient_image = self.transform_sequence(patient_image) # apply the transform_sequence if one is specified
        
            else:
                # even if no other transformation is applied, the image should be turned into a tensor
                to_tensor = transforms.ToTensor()
                patient_image = to_tensor(patient_image)
            
            return patient_image, torch.FloatTensor(image_labels)
    
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

print("Shape of image tensor:", shape_of_image_tensor, "\n")
print("Shape of label tensor:", shape_of_label_tensor, "\n")

# example using to_ones, to_zeros and to_ignore
all_pathologies = list(training_set.iloc[:, -13:-1].columns)

chexpert_train_alt = CheXpertDatasetProcessor(path=path, subset="train",image_paths=image_paths_train,
                                          number_of_images=training_set.shape[0], transform_sequence=transform_list,
                                          to_ones=all_pathologies[0:2],
                                          to_zeros=all_pathologies[2:4],
                                          to_ignore=all_pathologies[4:6],
                                          return_image=False)

chexpert_train_alt.process_chexpert_dataset()
chexpert_train_alt.__getitem__(0)

### Functional style as an alternative to processor class

def transform_labels(df: pd.DataFrame, 
                     to_ones: List[str] = None, 
                     to_zeros: List[str] = None, 
                     to_ignore: List[str] = None, 
                     replacement_for_blank: int = -1):
    
    """
    Args:
        df: pandas dataframe for which the labels should be altered
        to_ones: list of pathologies for which uncertainty labels should be replaced by 1.0
        to_zeros: list of pathologies for which uncertainty labels should be replaced by 0.0
        to_ignore: list of pathologies for which uncertainty labels should be ignored (label will be turned to nan)
        replacement_for_blank: value that should be used to replace the "blank" labels
    Returns: 
        df with altered labels
    """
    
    data = copy.deepcopy(df) # avoid altering the original dataframe
    pathologies = data.iloc[:, -13:-1].columns
    
    # prepare labels
    data.iloc[:, -13:-1] = data.iloc[:, -13:-1].replace(float("nan"), replacement_for_blank) # blank labels -> specified value
        
    if to_ones is not None:
        if all(p in pathologies for p in to_ones): # check whether arguments are valid pathologies
                data[to_ones] = data[to_ones].replace(-1, 1) # replace uncertainty labels with ones
        else:
            raise ValueError("List supplied to to_ones contains invalid pathology, please choose from:",
                                 list(pathologies))
            
    if to_zeros is not None:
        if all(p in pathologies for p in to_zeros):
            data[to_zeros] = data[to_zeros].replace(-1, 0) # replace uncertainty labels with zeros
        else:
            raise ValueError("List supplied to to_zeros contains invalid pathology, please choose from:",
                             list(pathologies))
            
    if to_ignore is not None:
        if all(p in pathologies for p in to_ignore):
            data[to_ignore] = data[to_ignore].replace(-1, float("nan")) # replace uncertainty labels with nan
        else:
            raise ValueError("List supplied to to_ignore contains invalid pathology, please choose from:",
                             list(pathologies))
        
    return data

df_train = transform_labels(training_set, to_ones=all_pathologies, replacement_for_blank=0) # sample output
print("Sample output of transform_labels:", "\n", df_train.iloc[:,-13:-1].head(5), "\n")

def transform_img(image_path: str,
                  df: pd.DataFrame,
                  transform_sequence: object = None):
    
    """
    Args:
        image_path: path that leads to the image
        df: pandas dataframe that contains the labels
        transform_sequence: sequence used to transform the image
    Returns: 
        224 x 224 image tensor and a corresponding tensor containing 12 labels
    """
    
    data = df[df["Path"] == os.path.join(image_path[image_path.rfind("CheXpert-v1.0-small") :])] # select corresponding row in original df
    
    image_labels = data.iloc[0, -13:-1]
        
    patient_image = Image.open(image_path).convert('RGB')
        
    if transform_sequence is not None:
        patient_image = transform_sequence(patient_image) # apply the transform_sequence if one is specified
        
    else:
        # even if no other transformation is applied, the image should be turned into a tensor
        to_tensor = transforms.ToTensor()
        patient_image = to_tensor(patient_image)
            
    return patient_image, torch.FloatTensor(image_labels)

print("Sample image tensor returned by transform_img:", "\n", transform_img(image_paths_train[0], df_train, transform_list)[0], "\n")
print("Sample label tensor returned by transform_img:", "\n", transform_img(image_paths_train[0], df_train, transform_list)[1], "\n")

def transform_images(image_paths: List[str], 
                     df: pd.DataFrame, 
                     transform_sequence: object = None):
    
    """
    Args:
        image_paths: paths to the images
        df: pandas dataframe that contains the labels
        transform_sequence: sequence used to transform the images
    Returns: 
        224 x 224 image tensor and a corresponding tensor containing 12 labels
    """
    
    for img in image_paths: # parallelizable
        transform_img(img, df, transform_sequence)

# sample input for the function:
# =============================================================================
# transform_images(image_paths_train, df_train, transform_list)
# =============================================================================

## Store the preprocessed data

"""
The two different options of storing the data provided in this tutorial are:

- storing each preprocessed image tensor with its corresponding labels in a separate .npz file (~ 35 GB in total)
- storing each resized image as a .jpg file and saving all of the labels in a joblib file (~ 2 GB in total)

The code for each option is commented out, in case you do not wish to store the data
"""

### Store data as .npz files

"""
If you prefer storing the images and labels in .npz files, please run the following code.
"""

# =============================================================================
# # store the training set
# for i, ex in enumerate(chexpert_train):
#       np.savez_compressed(os.path.join(storing_location, "train_images", str(i) + ".npz"), 
#                         image=ex[0], label=ex[1])
#       
# # store the validation set
# for i, ex in enumerate(chexpert_valid):
#       np.savez_compressed(os.path.join(storing_location, "valid_images", str(i) + ".npz"), 
#                         image=ex[0], label=ex[1])
# =============================================================================
   
"""
Please note that the `np.load` function returns an array and not a tensor, so the "image" result has to be transposed and turned into a tensor again.
The labels also need to be converted to a tensor again. 
Here is a quick example how to do this:
"""

# =============================================================================
# to_tensor = transforms.ToTensor()
# 
# example = np.load(os.path.join(storing_location, "train_images", str(0) + ".npz"))
# print(to_tensor(example["image"].transpose(1,2,0)))
# print(torch.FloatTensor(example["label"]))
# 
# # compare with original output from __getitem__()
# print(chexpert_train.__getitem__(0)[0]) # same result
# print(chexpert_train.__getitem__(0)[1]) # same result
# =============================================================================

### Store images as .jpg files

"""
If you want to store the resized images as .jpg files and the labels in .lib files, you can run the code below.
Please note that the normalization that we applied earlier results in negative values in the image tensors, which are not compatible with the `save_image` function that we use to store the images.
The images are therefore saved without the normalization.
Please be aware that storing the images as .jpg files can introduce some slight inaccuracies in the final image tensor (compared to the original output of __getitem__()).
"""

# =============================================================================
# # define transformations (without normalization)
# transform_list_resize = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     ])
# 
# # store the training images and labels
# chexpert_train_resize = CheXpertDatasetProcessor(path=path, subset="train", image_paths=image_paths_train, number_of_images=training_set.shape[0], transform_sequence=transform_list_resize)
# chexpert_train_resize.process_chexpert_dataset()
# 
# with open(joblib_labels_train, 'wb') as j:
#     for i, ex in enumerate(chexpert_train_resize): 
#         save_image(ex[0], os.path.join(storing_location, "train_images", str(i) + ".jpg"))
#         joblib.dump(ex[1], j)   
# 
# # store the validation images and labels
# chexpert_valid_resize = CheXpertDatasetProcessor(path=path, subset="valid", image_paths=image_paths_valid, number_of_images=validation_set.shape[0], transform_sequence=transform_list_resize)
# chexpert_valid_resize.process_chexpert_dataset()
# 
# with open(joblib_labels_valid, 'wb') as j:
#     for i, ex in enumerate(chexpert_valid_resize):
#         save_image(ex[0], os.path.join(storing_location, "valid_images", str(i) + ".jpg"))
#         joblib.dump(ex[1], j)
# 
# # the labels can be loaded using joblib.load
# with open(joblib_labels_valid, 'rb') as j:
#     print("First label saved in .lib file:", "\n", joblib.load(j), "\n")
#     print("Second label saved in .lib file:", "\n", joblib.load(j), "\n")
# =============================================================================

## Finish

"""
This concludes the preprocessing of the CheXpert data.
"""

### References

"""
CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison by Irvin et al. (2019): https://arxiv.org/pdf/1901.07031.pdf
Structured dataset documentation: a datasheet for CheXpert by Garbin et al. (2021): https://arxiv.org/pdf/2105.03020.pdf
"""