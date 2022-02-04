#!/usr/bin/env python
# coding: utf-8

# Police Killing Dataset: Data Preprocessing using RegEx as rules


"""
This tutorial shows how to find names of people killed by the police in a corpus of newspaper articles.
The corpus was created by Katherine A. Keith et al. (2017) for a similar task using distant supervision.
This dataset contains mentions of people (based on keywords related to “killing” or “police”), who might
have been killed by the police. The dataset (the HTML documents scraped in 2016 themselves as well as the
already sentence-segmented data) are available on the
project’s website (http://slanglab.cs.umass.edu/PoliceKillingsExtraction/) and on MinIO
( https://knodle.dm.univie.ac.at/minio/knodle/datasets/police_killing/).

Data Description

There is a train and a test dataset, both of them containing dictionaries with the following keys:
-	docid: unique identifiers of every mention of a person possibly killed by the police
-	name: the normalized name of the person (a pair of a first name and a last name that was identified
        by the HAPNIS (http://users.umiacs.umd.edu/~hal/HAPNIS/) name parser)
-	downloadtime: time the document was downloaded
-	names_org: the original name of the person mentioned in the document
-	sentnames: other names in the mention (not of the person possibly killed by the police)
-	sent_alter: the mention, but the name of the person possibly killed by the police is replaced by “TARGET” and any
        other names are replaced by “PERSON“
-	plabel: for the training data possibly erroneous labels obtained using weak supervision and gold labels for the
        test data – in this project, only the labels of the test data will be used-	sent_org: the original mention

Compared to the second approach that we try for solving this problem (using a Knowledge Base for labelling,
as demonstrated in the Data Preprocessing with Knowledge Base Tutorial
(https://github.com/knodle/knodle/blob/feature/%23299_police_killing_dataset/examples/data_preprocessing/police_killing/preprocessing_with_kb.ipynb),
we use RegEx as rules in this tutorial. The RegEx should cover all possible ways a sentence can express that a
person "TARGET" was killed by the police (using different words for killing and police as well as active and passive
constructions).

Reference: Keith, Kathrine A. et al. (2017): Identifying civilians killed by police with distantly supervised entity-event extraction. In: Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. doi: [10.18653/v1/D17-1163](https://aclanthology.org/D17-1163/)

"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Union
from itertools import islice

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import dump
from minio import Minio
from tqdm import tqdm


# Get the data

# define the files names
Z_MATRIX_TRAIN = "train_rule_matches_z.lib"
Z_MATRIX_DEV = "dev_rule_matches_z.lib"
Z_MATRIX_TEST = "test_rule_matches_z.lib"

T_MATRIX_TRAIN = "mapping_rules_labels_t.lib"

TRAIN_SAMPLES_OUTPUT = "df_train.lib"
DEV_SAMPLES_OUTPUT = "df_dev.lib"
TEST_SAMPLES_OUTPUT = "df_test.lib"

# file names for .csv files
TRAIN_SAMPLES_CSV = "df_train.csv"
DEV_SAMPLES_CSV = "df_dev.csv"
TEST_SAMPLES_CSV = "df_test.csv"

# define the path to the folder where the data will be stored
data_path = "../../../data_from_minio/police_killing"
os.makedirs(data_path, exist_ok=True)
os.path.join(data_path)


# Get the Train data

client = Minio("knodle.cc", secure=False)
files = [
    "train.json", "test.json" 
]
for file in tqdm(files):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("datasets/police_killing/", file),
        file_path=os.path.join(data_path, file),
    )



def get_train_data(data_path: str) -> pd.DataFrame:
    with open(os.path.join(data_path, "train.json"), 'r') as data:
        train_data = [json.loads(line) for line in data]
    df_train_sent_alter = pd.DataFrame(train_data, columns = ["sent_alter"]).rename(columns={"sent_alter": "sample"})
    return df_train_sent_alter

df_train = get_train_data(data_path)

#Get the Dev and Test Data

"""
The parameter *used_as_dev* reflects the amount of the gold data that should be used for development instead 
of testing. It is set to 30% for now, but can be changed depending on the task definition.
"""

used_as_dev = 30
print(f"{used_as_dev}% of the test data will be used for develoment.")


def get_dev_test_data(data_path: str) -> Union[pd.DataFrame, pd.DataFrame]:
    with open(os.path.join(data_path, "test.json"), 'r') as data:
        dev_test_data = [json.loads(line) for line in data]
    dev_test_sent_alter = pd.DataFrame(dev_test_data, columns = ["sent_alter", "plabel"]).rename(columns={"sent_alter": "sample", "plabel": "label"})
    df_dev = dev_test_sent_alter.sample(n = int(round((dev_test_sent_alter.shape[0]/100)*used_as_dev))).reset_index(drop = True)
    df_test = dev_test_sent_alter.drop(df_dev.index).reset_index(drop = True)
    return df_dev, df_test

df_dev, df_test = get_dev_test_data(data_path)


# Some Statistics

# Count of samples
print(f"Number of samples:")
print(f"Train data: {df_train.shape[0]}")
print(f"Development data: {df_dev.shape[0]}")
print(f"Test data: {df_test.shape[0]}")


# Positive and negative instances in dev and test data
positive_dev = df_dev.groupby("label").count()["sample"][1]
negative_dev = df_dev.groupby("label").count()["sample"][0]
positive_test = df_test.groupby("label").count()["sample"][1]
negative_test = df_test.groupby("label").count()["sample"][0]
print(f"In the develoment data, {positive_dev} ({(100/df_dev.shape[0])*positive_dev}%) instances are positive and {negative_dev} instances ({(100/df_dev.shape[0])*negative_dev}%) are negative.")
print(f"In the test data, {positive_test} ({(100/df_test.shape[0])*positive_test}%) instances are positive and {negative_test} instances ({(100/df_test.shape[0])*negative_test}%) are negative.")


# Output Classes

"""
Our task is to find out whether a sentence describes the killing of a person by the police or does not. 
That means, it is a binary classification task with two output classes. The number of classes is defined 
with the *num_classes* parameter.
"""

num_classes = 2


# Get the rules

"""
These word lists are mainly based on the lists of Keith et al. (2017, p. 11). 
However, here they are split into several different lists to create more precise RegEx. 
A rule must contain a police word, a killing word and in case the killing word is a shooting word, 
also a fatality word (due to the fact that just because someone is shot it does not necessarily 
mean they die). The different constructions make sure the words do not just appear in a random order 
in a senctence, but in a way the sentence can actually mean that the TARGET was killed by the police.
"""

# We start by creating a dictionary with all the rules and their corresponding rule IDs.
POLICE_WORDS = ['police', 'officer', 'officers', 'cop', 'cops', 'detective', 'sheriff', 'policeman', 'policemen',
                'constable', 'patrolman', 'sergeant', 'detectives', 'patrolmen', 'policewoman', 'constables',
                'trooper', 'troopers', 'sergeants', 'lieutenant', 'deputies', 'deputy']

KILLING_WORDS = ['shot', 'shoots', 'shoot', 'shooting', 'shots', 'killed', 'kill', 'kills', 'killing', 'murder', 'murders', 'fires', 'fired', 'hit', 'murdered']

SHOOTING_WORDS = ['shot', 'shoots', 'shoot', 'shooting', 'shots']

FATALITY_WORDS = ['fatal', 'fatally', 'died', 'killed', 'killing', 'dead', 'deadly', 'homicide', 'homicides', 'death']


def create_rules() -> Dict:
    
    rule2rule_id = {}
    rule_id = 0
    
    for police_word in POLICE_WORDS: 
        
        for killing_word in KILLING_WORDS:
            if killing_word not in SHOOTING_WORDS:
                r1 = f"{police_word}.*{killing_word}.*target"
                rule2rule_id[r1] = rule_id
                rule_id += 1
                r2 = f"target.*{killing_word}.*{police_word}"
                rule2rule_id[r2] = rule_id
                rule_id += 1
                r3 = f"{killing_word}.*{police_word}.*target"
                rule2rule_id[r3] = rule_id
                rule_id += 1
            
            else:
                for fatality_word in FATALITY_WORDS:
                    r4 = f"{police_word}.*{killing_word}.*target.*{fatality_word}"
                    rule2rule_id[r4] = rule_id
                    rule_id += 1
                    r5 = f"{police_word}.*{fatality_word}.*{killing_word}.*target"
                    rule2rule_id[r5] = rule_id
                    rule_id += 1
                    r6 = f"{police_word}.*{killing_word}.*{fatality_word}.*target"
                    rule2rule_id[r6] = rule_id
                    rule_id += 1
                    r7 = f"{fatality_word}.*{killing_word}.*{police_word}.*target"
                    rule2rule_id[r7] = rule_id
                    rule_id += 1
                    r8 = f"{fatality_word}.*{killing_word}.*target.*{police_word}"
                    rule2rule_id[r8] = rule_id
                    rule_id += 1
                    r9 = f"target.*{police_word}.*{killing_word}.*{fatality_word}"
                    rule2rule_id[r9] = rule_id
                    rule_id += 1
                    r10 = f"target.*{fatality_word}.*{killing_word}.*{police_word}"
                    rule2rule_id[r10] = rule_id
                    rule_id += 1
                    r11 = f"target.*{killing_word}.*{fatality_word}.*{police_word}"
                    rule2rule_id[r11] = rule_id
                    rule_id += 1
                    r12 = f"target.*{killing_word}.*{police_word}.*{fatality_word}"
                    rule2rule_id[r12] = rule_id
                    rule_id += 1
                    r13 = f"target.*{police_word}.*{fatality_word}.*{killing_word}"
                    rule2rule_id[r13] = rule_id
                    rule_id += 1 
                    
                        
    return rule2rule_id

rule2rule_id = create_rules()

print(f"There are {len(rule2rule_id)} rules.")
print("\nThe first rules of the rule2rule_id dictionary look like this:")
print(dict(islice(rule2rule_id.items(), 6)))

"""
Secondly, we create a dictionary assigning all rules to their label. There are only two classes 
(someone was killed by the police or was not killed by the police). Since there are no rules 
indicating that someone was **not** killed by the police, all rules indicate the positive class 1. 
Therefore, all values of the rule2label dictionary, containing the rule IDs as keys, can be set to 1.
"""

rule2label = {rule_id: 1 for rule_id in rule2rule_id.values()}

"""
Thirdly, a dictionary mapping the a label ID to the corresponding label is required for the further 
preprocessing. As there are only two classes, this can be done manually. 
"""
label_id2label = {0: "negative", 1: "positive"}


# Build the T matrix (rules x classes)

"""
The rows of the T matrix are the rules and the columns the classes. 
The T matrix is one-hot encoded (1 for a rule and its corresponding class). 
It is the same function as it is used in the TAC tutorial 
(https://github.com/knodle/knodle/blob/develop/examples/data_preprocessing/tac_based_dataset/entity_pairs_aka_lfs/Prepare_TAC_based_RE_dataset.ipynb).
"""

def get_mapping_rules_labels_t(rule2label: Dict, num_classes: int) -> np.ndarray:
    """ Function calculates t matrix (rules x labels) using the known correspondence of relations to decision rules """
    mapping_rules_labels_t = np.zeros([len(rule2label), num_classes])
    for rule, labels in rule2label.items():
        mapping_rules_labels_t[rule, labels] = 1
    return mapping_rules_labels_t

mapping_rules_labels_t = get_mapping_rules_labels_t(rule2label, num_classes)


# Build the Z matrix (instances x rules)

# Get the train data.

"""
We start by creating a list of dictionaries (one for each sample, later they will be the rows 
in the dataframe). They contain the sample itself as well as list of the matching rules and 
the corresponding rule IDs. In the first step, the lists are still empty. After that, we want 
to populate these empty lists. We take each rule and apply it to each sample. If it matches, 
the rule and the rule IDs are added to the correct dictionary. In the end, the list of dictionaries 
can be converted into a Pandas Dataframe. 
"""
def get_data(data: pd.DataFrame, rule2rule_id: Dict) -> pd.DataFrame:
    
    data_dicts = [{"sample": sample, "rules": [], "enc_rules": []} for sample in data["sample"].drop_duplicates()]
    
    for rule, rule_id in tqdm(rule2rule_id.items()):
        for data_dict in data_dicts:
            sample = data_dict["sample"]
            if re.search(rule, sample.lower()):
                data_dict["rules"].append(rule)
                data_dict["enc_rules"].append(rule_id)
    
    df = pd.DataFrame.from_dict(data_dicts)            
    df = df.reset_index()
       
    return df

train_data = get_data(df_train, rule2rule_id)


# Get the Dev and Test data

"""
Just as for the train data, we need a Dataframe with a sample, its corresponding rules, 
and the rule IDs. Moreover, we need to add the labels and the label IDs that we obtained 
earlier when reading the test data. We do this by merging the new Dataframe with sample, 
rule, and rule encoding only with the development and test Dataframes that contain the labels.
"""

def get_dev_test_df(rule2rule_id: Dict, data: pd.DataFrame, label_id2label: Dict) -> pd.DataFrame:

    dev_test_data_without_labels = get_data(data, rule2rule_id)
    dev_test_data = dev_test_data_without_labels.merge(data, how='inner').rename(columns={"label": "enc_label"})
    dev_test_data["labels"] = dev_test_data['enc_label'].map(label_id2label)
    
    return dev_test_data

dev_data = get_dev_test_df(rule2rule_id, df_dev, label_id2label)
test_data = get_dev_test_df(rule2rule_id, df_test, label_id2label)


# Convert Dataframes to (Sparse) Matrices

"""
The train, test, and development data that we just stored as Pandas Dataframes should now be converted 
into a Scipy sparse matrix. The rows of the sparse matrix are the samples and the columns are the rules 
(i.e., a cell is 1 if the corresponding rule matches the corresponding sample, 0 otherwise). We initialize 
it as an array in the correct size (samples x rules), fill it with 1s and 0s, and convert it to a sparse 
matrix at the end.
"""

def get_rule_matches_z_matrix(df: pd.DataFrame) -> sp.csr_matrix:

    z_array = np.zeros((len(df["index"].values), len(rule2rule_id)))

    for index in tqdm(df["index"]):
        enc_rules = df.iloc[index]['enc_rules']
        for enc_rule in enc_rules:
            z_array[index][enc_rule] = 1

    rule_matches_z_matrix_sparse = sp.csr_matrix(z_array)

    return rule_matches_z_matrix_sparse
train_rule_matches_z = get_rule_matches_z_matrix(train_data)
dev_rule_matches_z = get_rule_matches_z_matrix(dev_data)
test_rule_matches_z = get_rule_matches_z_matrix(test_data)


# Saving the files

Path(os.path.join(data_path, "processed_regex")).mkdir(parents=True, exist_ok=True)

dump(sp.csr_matrix(mapping_rules_labels_t), os.path.join(data_path, "processed_regex", T_MATRIX_TRAIN))
dump(train_data["sample"], os.path.join(data_path, "processed_regex", TRAIN_SAMPLES_OUTPUT))
train_data["sample"].to_csv(os.path.join(data_path, "processed_regex", TRAIN_SAMPLES_CSV), header=True)
dump(train_rule_matches_z, os.path.join(data_path, "processed_regex", Z_MATRIX_TRAIN))
dump(dev_data[["sample", "labels", "enc_label"]], os.path.join(data_path, "processed_regex", DEV_SAMPLES_OUTPUT))
dev_data[["sample", "labels", "enc_label"]].to_csv(os.path.join(data_path, "processed_regex", DEV_SAMPLES_CSV), header=True)
dump(dev_rule_matches_z, os.path.join(data_path, "processed_regex", Z_MATRIX_DEV))
dump(test_data[["sample", "labels", "enc_label"]], os.path.join(data_path, "processed_regex", TEST_SAMPLES_OUTPUT))
test_data[["sample", "labels", "enc_label"]].to_csv(os.path.join(data_path, "processed_regex", TEST_SAMPLES_CSV), header=True)
dump(test_rule_matches_z, os.path.join(data_path, "processed_regex", Z_MATRIX_TEST))


# Rule Accuracy

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
matched_instances = test_data["enc_rules"].str.len() != 0

for row in tqdm(range(test_data.shape[0])):
    if test_data.loc[row]["enc_label"] == 1: # the true label is 1
        if matched_instances[row]: # the predicted label is 1
            true_positive += 1
        else: # the predicted label is 0
            false_negative += 1
    else: # the true label is 0
        if matched_instances[row]: # the predicted label is 1
            false_positive += 1
        else: # the predicted label is 0
            true_negative += 1
            
positive_samples = test_data[test_data.enc_label == 1].shape[0]
negative_samples = test_data[test_data.enc_label == 0].shape[0]
            
true_positive_percent = (100 / positive_samples) * true_positive
true_negative_percent = (100 / negative_samples) * true_negative

print(f"Out of {test_data.shape[0]} samples in the test_data, {positive_samples} samples are positive and {negative_samples} are negative.\n")
print(f"By using only the rules to obtain weak labels, {true_positive_percent}% of all positive samples are matched by a rule and therefore labeled as positive. {true_negative_percent}% of all negative samples are correctly classified as negative. (Which means that {100 - true_negative_percent}% of all negative instances are covered by a rule\n")
print(f"True positives: {true_positive} \nTrue negatives: {true_negative} \nFalse positives: {false_positive} \nFalse negatives: {false_negative}")

# Data on MinIO:

"""
Raw Data:

- train.json: the raw train data
- test.json: the raw test data

Processed Data:

- df_train.lib: the train samples saved as .lib
- df_train.csv: the train samples as CSV
- df_test.lib: the test samples saved as .lib
- df_test.csv: the test data as CSV, containing the samples and the gold label as well as the corresponding label-ID
- df_dev.lib: the development samples saved as .lib
- df_dev.csv: the development data as CSV, containing the samples and the gold label as well as the corresponding label-ID
- train_rule_matches_z.lib: the Z-matrix for the train data (train samples x rules)
- test_rule_matches_z.lib: the Z-matrix for the test data (test samples x rules)
- mapping_rules_labels_t.lib: the T-matrix (rules x classes)
- dev_rule_matches_z.lib: the Z-matrix for the development data (development samples x rules)
"""