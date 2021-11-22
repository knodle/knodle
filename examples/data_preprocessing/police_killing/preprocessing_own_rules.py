#!/usr/bin/env python
# coding: utf-8

#Data Preprocessing using own rules (the Regex)



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



used_as_dev = int(input("How much of the test data should be used for development? (in percent) "))


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



police_words = ['police', 'officer', 'officers', 'cop', 'cops', 'detective', 'sheriff', 'policeman', 'policemen',
                'constable', 'patrolman', 'sergeant', 'detectives', 'patrolmen', 'policewoman', 'constables',
                'trooper', 'troopers', 'sergeants', 'lieutenant', 'deputies', 'deputy']

killing_words_active = ['shot', 'shoots', 'shoot', 'shooting', 'shots', 'killed', 'kill', 'kills', 'killing', 'murder', 'murders']

killing_words_passive = ['hit', 'shot', 'killed', 'murdered']

shooting_words = ['shot', 'shoots', 'shoot', 'shooting', 'shots']

fatality_words = ['fatal', 'fatally', 'died', 'killed', 'killing', 'dead', 'deadly', 'homicide', 'homicides']



def creating_rules():
    
    rule2rule_id = dict({})
    rule_id = 0
    
    for police_word in police_words: 
        
        for killing_word_active in killing_words_active:
            if killing_word_active not in shooting_words:
                a1 = f"{police_word}.*{killing_word_active}.*target"
                rule2rule_id[a1] = rule_id
                rule_id += 1
            else:
                for fatality_word in fatality_words:
                    a2 = f"{police_word}.*{killing_word_active}.*target.*{fatality_word}"
                    rule2rule_id[a2] = rule_id
                    rule_id += 1
                    a3 = f"{police_word}.*{fatality_word}.*{killing_word_active}.*target"
                    rule2rule_id[a3] = rule_id
                    rule_id += 1
                    a4 = f"{police_word}.*{killing_word_active}.*{fatality_word}.*target"
                    rule2rule_id[a4] = rule_id
                    rule_id += 1
                    

        for killing_word_passive in killing_words_passive:
            if killing_word_passive not in shooting_words:
                p1 = f"target.*{killing_word_passive}.*by.*{police_word}"
                rule2rule_id[p1] = rule_id
                rule_id += 1
            else: 
                for fatality_word in fatality_words:
                    p2 = f"target.*{killing_word_passive}.*{fatality_word}.*by.*{police_word}"
                    rule2rule_id[p2] = rule_id
                    rule_id += 1
                    p3 = f"target.*{fatality_word}.*{killing_word_passive}.*by.*{police_word}"
                    rule2rule_id[p3] = rule_id
                    rule_id += 1
                    p4 = f"target.*{killing_word_passive}.*by.*{police_word}.*{fatality_word}"
                    rule2rule_id[p4] = rule_id
                    rule_id += 1
                        
    return(rule2rule_id)


rule2rule_id = creating_rules()
print(f"There are {len(rule2rule_id)} rules.")


#getting the rule ID and creating a new dict with it, assigning 
#the value 1 to all of them because there is only one class

def rule_id2label_id(rule2rule_id):

    rule2label = dict({})

    for rule, rule_id in rule2rule_id.items():
        rule2label[rule_id] = 1
        
    return(rule2label)
        

rule2label = rule_id2label_id(rule2rule_id)

"""
manually creating a label2label_id dict, since there's only one class, 
and a label_id2label dict (I will need that later, but I'm not sure if I'm allowed to to it like this)
Is it okay to create it manually?
"""

label2label_id ={"negative":0, "positive":1}
label_id2label = {0: "negative", 1: "positive"}


# ## building the T matrix

num_classes = 2


#mapping to t matrix (I took this function from the TAC tutorial, still has to be imported from separate script)

def get_mapping_rules_labels_t(rule2label, num_classes):
    """ Function calculates t matrix (rules x labels) using the known correspondence of relations to decision rules """
    mapping_rules_labels_t = np.zeros([len(rule2label), num_classes])
    for rule, labels in rule2label.items():
        mapping_rules_labels_t[rule, labels] = 1
    return mapping_rules_labels_t

mapping_rules_labels_t = get_mapping_rules_labels_t(rule2label, num_classes)


# ## Building the Z matrix


def get_data_dicts(data, rule2rule_id):
#creating a dictionary for each sample, which will be a row in the df

    data_dicts_empty = []

    for sample in data["samples"].drop_duplicates():
        data_dict = dict({})
        data_dict["samples"] = sample
        data_dict["rules"] = []
        data_dict["enc_rules"] = []
        
        data_dicts_empty.append(data_dict)
        
    return data_dicts_empty


def get_data_for_dicts(data_dicts):
#adding the rules and the corresponding IDs to the dictionaries of each sample

    for rule, rule_id in tqdm(rule2rule_id.items()):
        for data_dict in data_dicts:
            sample = data_dict["samples"]
            if re.search(rule, sample.lower()):
                data_dict["rules"].append(rule)
                data_dict["enc_rules"].append(rule_id)
                
    return data_dicts


def get_df(data, rule2rule_id):
# converting the list of dicts in a df
    
    data_dicts_empty = get_data_dicts(data, rule2rule_id)
    data_dicts = get_data_for_dicts(data_dicts_empty)
    df = pd.DataFrame.from_dict(data_dicts)
    df = df.reset_index()
       
    return df


train_data = get_df(df_train, rule2rule_id)


# **getting the dev and test data**



def getting_test_data(rule2rule_id, data, label_id2label):

    test_data_without_labels = get_df(data, rule2rule_id)
    test_data = test_data_without_labels.merge(data, how='inner').rename(columns={"label": "enc_labels"})
    test_data["labels"] = test_data['enc_labels'].map(label_id2label)
    
    return test_data


dev_data = getting_test_data(rule2rule_id, df_dev, label_id2label)
test_data = getting_test_data(rule2rule_id, df_test, label_id2label)
test_data.head()


# **converting to sparse matrix**



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


Path(os.path.join(data_path, "processed_own_rules")).mkdir(parents=True, exist_ok=True)

dump(sp.csr_matrix(mapping_rules_labels_t), os.path.join(data_path, "processed_own_rules", T_MATRIX_OUTPUT_TRAIN))

dump(train_data["samples"], os.path.join(data_path, "processed_own_rules", TRAIN_SAMPLES_OUTPUT))
train_data["samples"].to_csv(os.path.join(data_path, "processed_own_rules", TRAIN_SAMPLES_OUTPUT_CSV), header=True)
dump(train_rule_matches_z, os.path.join(data_path, "processed_own_rules", Z_MATRIX_OUTPUT_TRAIN))

dump(dev_data[["samples", "labels", "enc_labels"]], os.path.join(data_path, "processed_own_rules", DEV_SAMPLES_OUTPUT))
dev_data[["samples", "labels", "enc_labels"]].to_csv(os.path.join(data_path, "processed_own_rules", DEV_SAMPLES_OUTPUT_CSV), header=True)
dump(dev_rule_matches_z, os.path.join(data_path, "processed_own_rules", Z_MATRIX_OUTPUT_DEV))

dump(test_data[["samples", "labels", "enc_labels"]], os.path.join(data_path, "processed_own_rules", TEST_SAMPLES_OUTPUT))
test_data[["samples", "labels", "enc_labels"]].to_csv(os.path.join(data_path, "processed_own_rules", TEST_SAMPLES_OUTPUT_CSV), header=True)
dump(test_rule_matches_z, os.path.join(data_path, "processed_own_rules", Z_MATRIX_OUTPUT_TEST))





