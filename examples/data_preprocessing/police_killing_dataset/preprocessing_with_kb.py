#!/usr/bin/env python
# coding: utf-8

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
(https://github.com/knodle/knodle/blob/feature/%23299_police_killing_dataset/examples/data_preprocessing/police_killing/data_preprocessing_with_regex.ipynb),
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
from typing import List, Dict, Union, Set
from itertools import combinations
from itertools import islice

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import dump
from minio import Minio
from tqdm import tqdm


# Get the data

"""
First of all, the file names for the output at the end of this notebook are defined. 
After that, the raw data can be downloaded from MinIO.
"""

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

client = Minio("knodle.cc", secure=False)
files = [
    "train.json", "test.json", "FATAL ENCOUNTERS DOT ORG SPREADSHEET (See Read me tab).xlsx" 
]
for file in tqdm(files):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("datasets/police_killing/", file),
        file_path=os.path.join(data_path, file),
    )


# Get the train data

def get_train_data(data_path: str) -> pd.DataFrame:
    with open(os.path.join(data_path, "train.json"), 'r') as data:
        train_data = [json.loads(line) for line in data]
    df_train_sent_alter = pd.DataFrame(train_data, columns = ["name", "sent_alter", "names_org"]).rename(columns={"sent_alter": "sample"})
    return df_train_sent_alter

df_train = get_train_data(data_path)
df_train.head()

"""
The parameter *used_as_dev* reflects the amount of the gold data that should be used for development 
instead of testing. It is set to 30% for now, but can be changed depending on the task definition.
"""
used_as_dev = 30
print(f"{used_as_dev}% of the test data will be used for develoment.")


def get_dev_test_data(data_path: str) -> Union[pd.DataFrame, pd.DataFrame]:
    with open(os.path.join(data_path, "test.json"), 'r') as data:
        dev_test_data = [json.loads(line) for line in data]
    dev_test_sent_alter = pd.DataFrame(dev_test_data, columns = ["name", "names_org", "sent_alter", "plabel"]).rename(columns={"sent_alter": "sample", "plabel": "label"})
    df_dev = dev_test_sent_alter.sample(n = int(round((dev_test_sent_alter.shape[0]/100)*used_as_dev))).reset_index(drop = True)
    df_test = dev_test_sent_alter.drop(df_dev.index).reset_index(drop = True)
    return df_dev, df_test

df_dev, df_test = get_dev_test_data(data_path)
df_test.head()


# Get Data from Knowledge Base

"""
The FE-Database contains information about people dying or experiencing violence in encounters with the police, 
including the name of the victim, a short description of the incident and the use of force (whether a person 
in an encounter was, for instance, killed by the police or committed suicide or died in an accident while 
the police was present). For us, only the entries about actual police killings (no suicide, no accidents etc.) 
are relevant.
It can be downloaded as an Excel spreadsheet from Fatal Encounters 
(https://docs.google.com/spreadsheets/d/1dKmaV_JiWcG8XBoRgP8b4e9Eopkpgt7FL7nyspvzAsE/edit#gid=0). 
After that, we want to exclude all the information that is not useful for our task. 
We only need the first sheet that contains the actual information. In this sheet, we only need the name of the 
victim and the column Intended use of force (Developing). We exclude all entries that do not describe Deadly Force.
In the end, we keep only the relevant names.
"""

fe_database = pd.read_excel(os.path.join(data_path, "FATAL ENCOUNTERS DOT ORG SPREADSHEET (See Read me tab).xlsx"), sheet_name = 0)
fe_database = fe_database[['Name', 'Intended use of force (Developing)']]
fe_database = fe_database[fe_database['Intended use of force (Developing)'] == "Deadly force" ]
fe_database = fe_database.drop(['Intended use of force (Developing)'], axis=1).reset_index(drop = True).rename(columns={"Name": "name"})


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


#  Output classes
num_classes = 2


# Get the Rules

"""
For this task, the rules will be the names of people found in both our data and the FE-Database. 
A rule matches if a name found in the FE-Database can also be found in a sample.
"""

# Standardizing the Names

"""

The problem when trying to match the FE-Database with our samples is that there might be different version 
of single names. For their dataset, Keith. et al used a standardized name version and kept all alternative names 
in a list in the column *names_org*. It is possible that the FE-Database contains name versions, which do not 
match with the standardized name of Keith et al., but with one of the versions in the *names_org*-list. 
(For instance, someone is called *Steiney James Richards Jr.* in the FE-Database and *James Richard* in our samples.)
We start solving this problem by mapping the standardized name in all three datasets to the different name versions. 
Then we can create a "reversed" dictionary containing all the different name versions as keys and the corresponding 
standardized name as value. This does not guarantee yet that we can find an exact name of the FE-Database in one of 
the name versions. Sometimes, a person has two forenames in the FE-Database, but only one forename is kept in the 
dataset of Keith et al. Therefore, we will also have to expand the list of names in the FE-Database. For persons 
whose name consists of more than two parts, we will add all different combinations of their name. 

After that, we can create the *intersection_fe_and_samples*-list, which contains the standardized names of people 
mentioned in both datasets and also considers all different versions. (We add a name—a value of the 
names_org2names-Dictionary—to the list if the name can be found in both the keys of the names_org2names-Dictionary 
and in the FE-Database.)
"""

names2names_org = dict(zip(df_train["name"].to_list(), df_train["names_org"].to_list()))
names2names_org.update(dict(zip(df_dev["name"].to_list(), df_dev["names_org"].to_list())))
names2names_org.update(dict(zip(df_test["name"].to_list(), df_test["names_org"].to_list())))

names_org2names = {}

for name, names_org in names2names_org.items():
    for name_org in names_org: 
        names_org2names[name_org] = name

names_in_fe_database = fe_database["name"].to_list()
new_name_combinations = []

for name in names_in_fe_database: 
    name_parts = list(name.split())
    for i in range(len(name_parts)+1):
        for combination in combinations(name_parts, i): 
            if len(combination) > 1:
                new_name = ""
                for word in combination: 
                    new_name += word
                    new_name += " "  
                new_name = new_name[:-1]
                    
                new_name_combinations.append(new_name)
                

names_in_fe_database += new_name_combinations
names_in_fe_database = list(set(names_in_fe_database))

intersection_fe_and_samples = set([names_org2names[name_org] for name_org in names_org2names.keys() if name_org in names_in_fe_database])

print(f"{len(intersection_fe_and_samples)} of the people mentioned in the training, test and test samples can also be found in the FE-Database.")


# Mapping Names to Rules

def get_rule2id(intersection_fe_and_samples: Set) -> Dict:
    rule2rule_id = {}
    rule_id = 0
    for name in intersection_fe_and_samples: 
        rule2rule_id[name] = rule_id
        rule_id += 1

    return rule2rule_id

rule2rule_id = get_rule2id(intersection_fe_and_samples)

print(f"There are {len(rule2rule_id)} rules.")
print("\nThe first rules of the rule2rule_id dictionary look like this:")
print(dict(islice(rule2rule_id.items(), 6)))

rule2label = {rule_id: 1 for rule_id in rule2rule_id.values()}


# Build the T matrix (rules x classes)

"""
The rows of the T matrix are the rules and the columns the classes. The T matrix is one-hot encoded 
(1 for a rule and its corresponding class). It is the same function as it is used in the 
TAC tutorial (https://github.com/knodle/knodle/blob/develop/examples/data_preprocessing/tac_based_dataset/entity_pairs_aka_lfs/Prepare_TAC_based_RE_dataset.ipynb).
"""

def get_mapping_rules_labels_t(rule2label: Dict, num_classes: int) -> np.ndarray:
    """ Function calculates t matrix (rules x labels) using the known correspondence of relations to decision rules """
    mapping_rules_labels_t = np.zeros([len(rule2label), num_classes])
    for rule, labels in rule2label.items():
        mapping_rules_labels_t[rule, labels] = 1
    return mapping_rules_labels_t

mapping_rules_labels_t = get_mapping_rules_labels_t(rule2label, num_classes)


# Build the Z matrix (instances x rules)

"""
Match Train Data with the rules: If an entity (a person) that can also be found in the rule2rule_id-Dictionary 
is mentioned in a sample, the name of this entity will be assigned to the rules column in the dataframe. 
The column enc_rules contains the rule ID corresponding to the name. We do not need the *names_org*-column anymore.
"""

def get_df(data: pd.DataFrame, rule2rule_id: Dict) -> pd.DataFrame: 
    
    rules = [name if name in rule2rule_id.keys() else "" for name in data["name"].to_list()]
    enc_rules = [rule2rule_id[name] if name in rule2rule_id.keys() else "" for name in data["name"].to_list()]
    data["rule"] = rules
    data["enc_rule"] = enc_rules
    data = data.drop(['names_org'], axis=1).reset_index(drop = True)
    data = data.reset_index()
    
    return data


train_data = get_df(df_train, rule2rule_id)
train_data[8:20]

"""
Match Dev and Test Data with the rules: Just as for the train data, we need a Dataframe with a sample, 
its corresponding rules, and the rule IDs for dev and test data. Moreover, we need the labels and the label 
IDs that we obtained earlier when reading the test data. The label ID is already present in the data, the label 
we add manually. 
"""

def get_dev_test_df(rule2rule_id: Dict, data: pd.DataFrame) -> pd.DataFrame:

    dev_test_data = get_df(data, rule2rule_id)
    dev_test_data = dev_test_data.rename(columns={"label": "enc_label"})
    dev_test_data['label'] = np.where(dev_test_data['enc_label'] == 0, "negative", "positive")
    
    return dev_test_data

dev_data = get_dev_test_df(rule2rule_id, df_dev)
test_data = get_dev_test_df(rule2rule_id, df_test)

"""
Convert Dataframes to (Sparse) Matrices: The train, test, and development data that we just stored as Pandas Dataframes
hould now be converted into a Scipy sparse matrix. The rows of the sparse matrix are the samples and the columns 
are the rules (i.e., a cell is 1 if the corresponding rule matches the corresponding sample, 0 otherwise). 
We initialize it as an array in the correct size (samples x rules), fill it with 1s and 0s, and convert it 
to a sparse matrix at the end.
"""

def get_rule_matches_z_matrix(df: pd.DataFrame) -> sp.csr_matrix:

    z_array = np.zeros((len(df["index"].values), len(rule2rule_id)))

    for index in df["index"]:
        rule = df.iloc[index]['enc_rule']
        if rule != "":
            z_array[index][rule] = 1

    rule_matches_z_matrix_sparse = sp.csr_matrix(z_array)

    return rule_matches_z_matrix_sparse

train_rule_matches_z = get_rule_matches_z_matrix(train_data)
dev_rule_matches_z = get_rule_matches_z_matrix(dev_data)
test_rule_matches_z = get_rule_matches_z_matrix(test_data)


# Save the Files

Path(os.path.join(data_path, "processed_kb")).mkdir(parents=True, exist_ok=True)

dump(sp.csr_matrix(mapping_rules_labels_t), os.path.join(data_path, "processed_kb", T_MATRIX_TRAIN))
dump(train_data["sample"], os.path.join(data_path, "processed_kb", TRAIN_SAMPLES_OUTPUT))
train_data["sample"].to_csv(os.path.join(data_path, "processed_kb", TRAIN_SAMPLES_CSV), header=True)
dump(train_rule_matches_z, os.path.join(data_path, "processed_kb", Z_MATRIX_TRAIN))
dump(dev_data[["sample", "label", "enc_label"]], os.path.join(data_path, "processed_kb", DEV_SAMPLES_OUTPUT))
dev_data[["sample", "label", "enc_label"]].to_csv(os.path.join(data_path, "processed_kb", DEV_SAMPLES_CSV), header=True)
dump(dev_rule_matches_z, os.path.join(data_path, "processed_kb", Z_MATRIX_DEV))
dump(test_data[["sample", "label", "enc_label"]], os.path.join(data_path, "processed_kb", TEST_SAMPLES_OUTPUT))
test_data[["sample", "label", "enc_label"]].to_csv(os.path.join(data_path, "processed_kb", TEST_SAMPLES_CSV), header=True)
dump(test_rule_matches_z, os.path.join(data_path, "processed_kb", Z_MATRIX_TEST))


# Rule Accuracy and Some Statistics

positive_test_samples = test_data[test_data.enc_label == 1].shape[0]
negative_test_samples = test_data[test_data.enc_label == 0].shape[0]

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
matched_instances = test_data["enc_rule"].str.len() != 0

for row in range(test_data.shape[0]):
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
                 
true_positive_percent = (100 / positive_test_samples) * true_positive
true_negative_percent = (100 / negative_test_samples) * true_negative

print(f"Out of {test_data.shape[0]} samples in the testdata, {positive_test_samples} samples are positive and {negative_test_samples} are negative.\n") 
print(f"By using only the rules to obtain weak labels, {true_positive_percent}% of all positive samples are matched by a rule and therefore labeled as positive. {true_negative_percent}% of all negative samples are correctly classified as negative.\n") 
print(f"True positives: {true_positive} \nTrue negatives: {true_negative} \nFalse positives: {false_positive} \nFalse negatives: {false_negative}")

relevant = positive_test_samples
retrieved = true_positive + false_positive

prec = true_positive / retrieved
rec = true_positive / relevant

f1 = (2 * prec * rec) / (prec + rec)

print(f"F1-Score: {f1}")

# Data on MinIO:

"""
Raw Data:

- train.json: the raw train data
- test.json: the raw test data
- FATAL ENCOUNTERS DOT ORG SPREADSHEET (See Read me tab).xlsx: the spreadsheet from the Fatal Encounters Database
 
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