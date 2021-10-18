import pandas as pd
import numpy as np
import sklearn
from PIL import Image
import matplotlib.pyplot as plt
import re

np.random.seed(0) #set seed for document

training_set = pd.read_csv("path_to_data")
validation_set = pd.read_csv("path_to_data")

#different ways to  load the images, will decide on one later
image = Image.open("path_to_image")
#image.show()
plt.imshow(image)
plt.show()

image_asarray = np.asarray(image)
plt.imshow(image_asarray)
plt.show()

import cv2
image_2 = cv2.imread("path_to_image")
image_2_from_array = Image.fromarray(image_2)
plt.imshow(image_2)
plt.show()

plt.imshow(image_2_from_array)
plt.show()

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
image_3 = load_img("path_to_image")
image_3_array = img_to_array(image_3)

plt.imshow(image_3)
plt.show()

#additional available features
print("Additional information:", training_set.columns[1:-14])
print("\n")

#check occurances of diseases in training and validation set
print("Distribution of observations in the training set:")
df = training_set.iloc[:, -14:]
for cond in df.columns:
    print(df.value_counts(subset=[cond]).sort_index(ascending=True))
    print("\n")

print("Distribution of observations in the (original) validation set:")
print(validation_set.iloc[:, -14:].sum(axis=0)) #some conditions are not present in the validation set or there are very few instances -> maybe don't split the set and use it as a test set instead?
print("\n")

#split validation set in validation and test
#extract patient id from path (patients in validation and test should not overlap)

def extract_ids(paths):
    return([re.findall(r'patient\d+', path)[0] for path in paths])

extract_ids(validation_set["Path"]) #test

validation_set.insert(1, "Id", extract_ids(validation_set["Path"]))

#how many unique patients are in the validation set?
print("Number of unique patients in the validation set:", len(np.unique(validation_set["Id"]))) #200, matches paper
print("\n")

#select 100 patients for test set randomly
test_idx = np.random.choice(np.unique(validation_set["Id"]), size=100)

test_set = validation_set[validation_set.Id.isin(test_idx) == True]
validation_set = validation_set[validation_set.Id.isin(test_idx) == False]

#occurences in validation and test
print("Distribution of observations in the NEW validation set:")
print(validation_set.iloc[:, -14:].sum(axis=0))
print("\n")
print("Distribution of observations in the (new) test set:")
print(test_set.iloc[:, -14:].sum(axis=0))


#prepare labels for different training methods
def preprocess_training_labels(train, method, condition = None):
    
    if method not in ["U-Ones", "U-Zeroes", "U-Ignore", "3-Classes"]:
        raise ValueError("Please choose form the methods: 'U-Ones', 'U-Zeroes', 'U-Ignore', '3-Classes'")
    
    if method == "U-Ones":
        
        if condition == None:
            train.iloc[:, -14:] = train.iloc[:, -14:].replace(-1, 1)
        else:
            train[condition] = train[condition].replace(-1, 1)
    
    if method == "U-Zeroes":
        
        if condition == None:
            train.iloc[:, -14:] = train.iloc[:, -14:].replace(-1, 0)
        else:
            train[condition] = train[condition].replace(-1, 0)
            
    if method == "U-Ignore":
        
        if condition == None:
            train.iloc[:, -14:] = train.iloc[:, -14:].replace(-1, float("nan")) #NaN might not be best choice here
        else:
            train[condition] = train[condition].replace(-1, float("nan"))
            
    if method == "3-Classes":
        return train
        
    return train

#example
train_new = preprocess_training_labels(training_set, "U-Zeroes", ["Edema", "Fracture"])
train_new_2 = preprocess_training_labels(training_set, "U-Ignore", ["Edema", "Fracture"])
#print(train_new["Edema"].unique()) #test
#preprocess_training_labels(training_set, "U-zeroes", ["Edema", "Fracture"]) #test exception

#matrices for knodle:
#X: features
#Z: rule matches
#T: mapping from rules to classes

###original paper: annotations come from reports (not available)
