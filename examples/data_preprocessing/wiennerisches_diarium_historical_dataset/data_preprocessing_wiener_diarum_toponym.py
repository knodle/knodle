# import libraries

import os
import re
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
from joblib import dump
from minio import Minio

client = Minio("knodle.cc", secure=False)

# define the path to the folder where the data will be stored
data_path = "../../../data_from_minio/wiener_diarum_toponyms"
os.makedirs(data_path, exist_ok=True)
os.path.join(data_path)

files = ["annotations_3-21_v4.html"]

for file in tqdm(files):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("datasets/wiener_diarum_toponyms/",file),
        file_path=os.path.join(data_path, file[-1]))

# Load html file

with open(os.path.join(data_path, file[-1]), 'r', encoding = "utf-8" ) as f:
    sterbelisten_html = f.read()

# Cleaning the text first from both html tags and xml tags and then in a second step clean the intro and replace with /n

# get rid of the title of the issue and the xml tags by just replacing the whole intro with ""
sterbelisten_strip_html_1 = re.sub(r"<h2>.*</h2><h3>.*xml \|.*\d+</h3><br/>","",sterbelisten_html)

# now working on all the tags around the sentences and weird characters within the words
sterbelisten_strip_html1 = re.sub(r"<hr/>|<p>|</p>|#+|=| =|<html>|</html>|\r|\b \.|\b# \.|(|)","",sterbelisten_strip_html_1)#<mark>|</mark>

sterbelisten_strip_html2 = re.sub(r"    \b","",sterbelisten_strip_html1)

# create an empty list and split the text at line break, creating an element for each sentence
sterbeliste = []
sterbeliste = sterbelisten_strip_html2.split("\n\n\n")

# CREATE LABELED SENTENCES AND A DICTIONARY OF PLACE NAMES

def get_location_id(curr_loc: List, all_loc: Dict) -> tuple[int,Dict]:
    '''
    Descr: This function loops through the list of place names,
    if it doesn't find a given place name it adds the place name to the list
    and returns the new list

    Args: the list or single word that is part of a place name
    Returns: list of place names
    '''
    for word in curr_loc:
        if word not in all_loc:
            all_loc[word]= len(all_loc)
        return all_loc[word]

def preprocess_sentence(sentence):
    '''
    This function replaces the <mark> tag with <start> and the </mark> tag with <end> in order
    to clean the remaining / out of the senctenses. It then splits the sentence into seperate words
    and returns a list with all words in the senctense.

    Args:
        "sentece" is the sentence as a string containing <mark> and </mark> and several "/".
    Returns:
        A list with all words in the sentence split by " " cleaned from "/" and
        containing <start> and <end> tag instead of <mark>
    '''
    sentence = re.sub(r"<mark>","<start> ",sentence)
    sentence = re.sub(r"</mark>"," <end>",sentence)
    sentence = re.sub(r"\(|\)|/","",sentence)
    sentence = sentence.split(" ")
    sentence = list(filter(None,sentence))
    return sentence

def create_labels(sterbeliste: List):
    '''
    Loops through the sentences and creates a list with Labels for each word in a sentence.
    It also creates a dictionary with an ID for each place name
    and collects all cleaned and preprocessed sentences as a list of lists.

    Args:
        sentence= a sentence as a string
        current_loc= a list with all words that are within one place name tag.
        labels=  list with labels for each word, None if it is not a place name and the
        ID of the occuring place name if a word belongs to a place name.
    Return:
        samples = a list with all label lists for all sentences.
        clean_sterbeliste= a list each sentence as a list of words.
        all_loc= a dictionary with all place names and its ID number.

    '''
    # we create a dictionary locations that contains the place name as a key and the ID as a value
    all_loc={}

    # we collect our tagged sentences in samples, these contain None for each word that is no place name and the corresponding ID to each word that is part of a place name
    samples = []

    # we collect a list with each sentence that is a list containing each word as its elements
    clean_sterbeliste=[]


    for sentence in sterbeliste:
        if re.search(r"/mark", sentence) is None:
            continue
        labels = []
        sentence = preprocess_sentence(sentence)
        print(sentence)
        if_loc = False
        for word in sentence:
            if word == '<start>':
                if_loc = True
                curr_loc = []
            elif word == '<end>':
                labels += [get_location_id(curr_loc, all_loc)] * len(curr_loc)
                if_loc = False
            else:
                if if_loc:
                    curr_loc.append(word)
                else:
                    labels.append(None)
        samples.append(labels)
        sentence = [word for word in sentence if word !="<start>" if word!="<end>"]
        clean_sterbeliste.append(sentence)
        print(all_loc)
    return samples, clean_sterbeliste, all_loc
samples, clean_sterbeliste, all_loc = create_labels(sterbeliste)


print(f"Sample_tagging:{samples[0]}")
print(f"Cleaned Sentence: {clean_sterbeliste[0]}")
print(f"=======================================")
print(f"Number of place names: {len(all_loc)}")
print(f"Number of sample sentences: {len(clean_sterbeliste)}")
print(f"=======================================")

# BUILDING TRAINING DATA

# saving the training data, leaving 100 samples as test data

d = {"sample": clean_sterbeliste, "labels":samples}

df = pd.DataFrame(d)

df_train = df[100:]


'''
X Matrix

Dimensions # sentences x # sentences (one column samples df, len(sample))
'''
x_matrix = pd.DataFrame(clean_sterbeliste)

print(f"Shape of X Matrix: {x_matrix.shape}")

'''
T Matrix

Dimensions: # place names x # 2
'''
a= np.zeros(len(all_loc))
b= np.full(len(all_loc),1)

t_matrix = np.stack((b,a.T))
print(f"Shape of T Matrix: {t_matrix.T.shape}")
'''
Z Matrices

In the next function, we are building the Z Matrices by looping through all samples.
Each sample is compared with the list of locations, if the location is in the sample,
the new list "matched_loc" gets an entry 1 otherwise a 0 is added
Then the list is transformed to an numpy array and brought into shape:

Dimensions
number of place names x number of words

'''
locations_list = list(range(0,(len(all_loc.keys()))))

## create a list and turn to np.array 3d

collected_z_matrices=[]

for sample in samples:
    matched_loc= [1 if i == j else 0 for i in sample for j in locations_list]
    i=len(sample)
    Z = np.array(matched_loc)
    Z = np.reshape(Z, (i,len(locations_list)))
    collected_z_matrices.append(Z)

print(f"Number of Z Matrices in our list: {len(collected_z_matrices)}")
print(f"Shape of one Z Matrix: {collected_z_matrices[0].shape}")

#spilt the test and training z_matrices
test_rule_matches_sparse_z_list = collected_z_matrices[:100]

train_rule_matches_sparse_z_list = collected_z_matrices[100:]

# These lists can't be stacked to a tensor yet, as they need padding.

# BUILDING TEST DATA

# Building test data, the first 100 samples have been manually checked

df_test = df[:100]

# SAVING  DATA

# saving the z_matrices

dump(train_rule_matches_sparse_z_list, os.path.join(data_path, "train_rule_matches_z_list.lib"))
dump(test_rule_matches_sparse_z_list, os.path.join(data_path, "test_rule_matches_z_list.lib"))

# saving the t_matix

dump(t_matrix,  os.path.join(data_path, "mapping_rules_labels_t.lib").replace('\\','/'))

# Saving test data frame

df_test.to_csv(os.path.join(data_path, 'df_test.csv'))

# Saving training data frame

df_train.to_csv(os.path.join(data_path, 'df_train.csv'))

