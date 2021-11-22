#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing (using rules from the paper)



import pandas as pd
import json
import os
import numpy as np
import re
import scipy.sparse as sp
from tqdm import tqdm
from pathlib import Path
from joblib import dump


# ## Getting the data

# define the files names
Z_MATRIX_OUTPUT_TRAIN = "train_rule_matches_z.lib"
Z_MATRIX_OUTPUT_DEV = "dev_rule_matches_z.lib"
Z_MATRIX_OUTPUT_TEST = "test_rule_matches_z.lib"

T_MATRIX_OUTPUT_TRAIN = "mapping_rules_labels_t.lib"

TRAIN_SAMPLES_OUTPUT = "df_train.lib"
DEV_SAMPLES_OUTPUT = "df_dev.lib"
TEST_SAMPLES_OUTPUT = "df_test.lib"

# file names for .csv files
TRAIN_SAMPLES_OUTPUT_CSV = "df_train.csv"
DEV_SAMPLES_OUTPUT_CSV = "df_dev.csv"
TEST_SAMPLES_OUTPUT_CSV = "df_test.csv"

# define the path to the folder where the data will be stored
data_path = "C:/Users/Emilie/Uni/2021WS/DS_project/data/police_killing"
os.path.join(data_path)


# later first downloaded from Minio

def get_train_data(data_path):
    with open(os.path.join(data_path, "train.json"), 'r') as data:
        train_data = [json.loads(line) for line in data] #a list of dicts
    #df_train_all = pd.DataFrame(train_data)
    df_train_sent_alter = pd.DataFrame(train_data, columns = ["sent_alter"]).rename(columns={"sent_alter": "samples"})
    return df_train_sent_alter

df_train = get_train_data(data_path)



# **getting the dev and test data.**
# 
# The train data contains 132833 samples, the test data 68925 samples. Some of the test data can be used as develoment data.
# 
# The samples for the dev data are randomly taken to avoid imbalances of positive and negative sample in dev and test data.


#how many of the test data should be used as dev data in %

used_as_dev = int(input("How much of the test data should be used for development? "))




def get_dev_test_data(data_path):
    with open(os.path.join(data_path, "test.json"), 'r') as data:
        dev_test_data = [json.loads(line) for line in data]
    dev_test_sent_alter = pd.DataFrame(dev_test_data, columns = ["sent_alter", "plabel"]).rename(columns={"sent_alter": "samples", "plabel": "label"})
    df_dev = dev_test_sent_alter.sample(n = int(round((dev_test_sent_alter.shape[0]/100)*used_as_dev))).reset_index(drop = True)
    df_test = dev_test_sent_alter.drop(df_dev.index).reset_index(drop = True)
    return df_dev, df_test

df_dev, df_test = get_dev_test_data(data_path)

print(f"The dev data contains {df_dev.shape[0]} samples, the test data contains {df_test.shape[0]} samples.")



# ## Getting the rules

# Starting with simple rules based on the keywords of the paper.
# 
# "These lists were semi-automatically constructed by looking up the nearest neighbors of “police” and “kill” (by cosine distance) from Google’s public release of word2vec vectors pretrained on a very large (proprietary) Google News corpus,20 and then manually excluding a small number of misspelled words or redundant capitalizations (e.g. “Police” and “police”)." (Keith et al. p. 11)
# 
# I saved the keywords in a csv.



def get_keywords(data_path):
    keywords = pd.read_csv(os.path.join(data_path, "keywords.csv"))
    return(keywords)

keywords = get_keywords(data_path)


#dictionary mapping the rules to their ID

def rule2id(keywords):
    
    rule2rule_id = dict({})
    rule_id = 0
    for police_word in keywords["police_words"]:
        for kill_word in keywords["kill_words"].dropna():
            rule2rule_id[f'{police_word} {kill_word}'] = rule_id
            rule_id += 1
    
    return(rule2rule_id)

rule2rule_id = rule2id(keywords)


#getting the rule ID and creating a new dict with it, assigning 
#the value 1 to all of them because there is only one class

def rule_id2label_id(rule2rule_id):

    rule2label = dict({})

    for rule, rule_id in rule2rule_id.items():
        rule2label[rule_id] = 1
        
    return(rule2label)
        

rule2label = rule_id2label_id(rule2rule_id)

"""
manually creating a label2label_id dict, since there are only two classes, 
and a label_id2label dict (I will need that later, but I'm not sure if I'm allowed to to it like this)
Is it okay to create it manually?
"""

label2label_id ={"negative":0, "positive":1}
label_id2label = {0: "negative", 1: "positive"}


# ## building the T matrix
# 


num_classes = 2


"""
mapping to t matrix 
(from TAC tutorial)
"""

def get_mapping_rules_labels_t(rule2label, num_classes):
    """ Function calculates t matrix (rules x labels) using the known correspondence of relations to decision rules """
    mapping_rules_labels_t = np.zeros([len(rule2label), num_classes])
    for rule, labels in rule2label.items():
        mapping_rules_labels_t[rule, labels] = 1
    return mapping_rules_labels_t

mapping_rules_labels_t = get_mapping_rules_labels_t(rule2label, num_classes)


# ## buidling the Z matrix (instances x rules)

# **getting the train data**


def rules2regex(rule2rule_id):
    
    searches = dict({})
    
    for rule in rule2rule_id.keys():
            wordpair = rule.split()
            search4police_word = f'{wordpair[0]}\W'
            search4kill_word = f'{wordpair[1]}\W'
            searches[rule] = [search4police_word, search4kill_word]
            
    return searches


def get_data_for_df(data, searches):

    data_for_df = []

    for sample in tqdm(data["samples"].drop_duplicates()):
        data_dict = dict({})
        data_dict["samples"] = sample
        data_dict["rules"] = []
        data_dict["enc_rules"] = []
        
        #if "label" in data.columns:
            #data_dict["label"] = data.loc[data.samples == sample, 'label'].values[0]


#this would have been my idea to keep the labels in the test data. But it makes it very slow. (8 instead of less than 1 mins)
#instead, I add the labels to the test data later in a separate function. Is that okay?

        
        for rule, search in searches.items():
            if re.search(search[0], sample.lower()) and re.search(search[1], sample.lower()):
                data_dict["rules"].append(rule)
                data_dict["enc_rules"].append(rule2rule_id[rule])

        if data_dict["enc_rules"] != []:
            data_for_df.append(data_dict)

    return data_for_df


def get_df(rule2rule_id, data):
    
    searches = rules2regex(rule2rule_id)
    data_for_df = get_data_for_df(data, searches)
    df = pd.DataFrame.from_dict(data_for_df)
    df = df.reset_index()
       
    return(df)



train_data = get_df(rule2rule_id, df_train)
train_data.head()


# **getting the dev and test data**
# 
# (same as for the train data, moreover the labels and label_ids are added)


def get_dev_test_df(rule2rule_id, data, label_id2label):

    test_data_without_labels = get_df(rule2rule_id, data)
    test_data = test_data_without_labels.merge(data, how='inner').rename(columns={"label": "enc_labels"})
    test_data["labels"] = test_data['enc_labels'].map(label_id2label)
    
    return test_data

dev_data = get_dev_test_df(rule2rule_id, df_dev, label_id2label)
test_data = get_dev_test_df(rule2rule_id, df_test, label_id2label)

def get_rule_matches_z_matrix(df):

    """
    creating a sparse matrix with instances as rows and rules as columns, 1 if the rule matches the instance
    """
    z_array = np.zeros((len(df["index"].values), len(rule2rule_id)))

    for index in tqdm(df["index"]):
        enc_rules = df.iloc[index-1]['enc_rules']
        for enc_rule in enc_rules:
            z_array[index][enc_rule] = 1

    rule_matches_z_matrix_sparse = sp.csr_matrix(z_array)

    return rule_matches_z_matrix_sparse


train_rule_matches_z = get_rule_matches_z_matrix(train_data)
dev_rule_matches_z = get_rule_matches_z_matrix(dev_data)
test_rule_matches_z = get_rule_matches_z_matrix(test_data)


# ## saving the files


Path(os.path.join(data_path, "processed")).mkdir(parents=True, exist_ok=True)

dump(sp.csr_matrix(mapping_rules_labels_t), os.path.join(data_path, "processed", T_MATRIX_OUTPUT_TRAIN))

dump(train_data["samples"], os.path.join(data_path, "processed", TRAIN_SAMPLES_OUTPUT))
train_data["samples"].to_csv(os.path.join(data_path, "processed", TRAIN_SAMPLES_OUTPUT_CSV), header=True)
dump(train_rule_matches_z, os.path.join(data_path, "processed", Z_MATRIX_OUTPUT_TRAIN))

dump(dev_data[["samples", "labels", "enc_labels"]], os.path.join(data_path, "processed", DEV_SAMPLES_OUTPUT))
dev_data[["samples", "labels", "enc_labels"]].to_csv(os.path.join(data_path, "processed", DEV_SAMPLES_OUTPUT_CSV), header=True)
dump(dev_rule_matches_z, os.path.join(data_path, "processed", Z_MATRIX_OUTPUT_DEV))

dump(test_data[["samples", "labels", "enc_labels"]], os.path.join(data_path, "processed", TEST_SAMPLES_OUTPUT))
test_data[["samples", "labels", "enc_labels"]].to_csv(os.path.join(data_path, "processed", TEST_SAMPLES_OUTPUT_CSV), header=True)
dump(test_rule_matches_z, os.path.join(data_path, "processed", Z_MATRIX_OUTPUT_TEST))


