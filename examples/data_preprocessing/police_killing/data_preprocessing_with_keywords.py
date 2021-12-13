#!/usr/bin/env python
# coding: utf-8

"""
Data Preprocessing

The aim of this project is to find names of people killed by the police in a corpus of news paper articles. The 
corpus was created by Katherine A. Keith et al. (2017) for a similar task using distant supervision. 
This dataset contains mentions of people (based on keywords related to “killing” or “police”) who might 
have been killed by the police. The dataset (the HTML documents scraped in 2016 themselves as well as the 
already sentence-segmented data) are available on the project’s website 
(http://slanglab.cs.umass.edu/PoliceKillingsExtraction/) 
and on MinIO ( https://knodle.dm.univie.ac.at/minio/knodle/datasets/police_killing/). 
There is a train and a test dataset, both of them containing dictionaries with the following keys:

-  docid: unique identifiers of every mention of a person possible killed by the police
-  name: the normalized name of the person
-  downloadtime: time the document was downloaded
-  names_org: the original name of the person mentioned in the document
-  sentnames: other names in the mention (not of the person possibly killed by the police)
-  sent_alter: the mention, name of the person possible killed by the policed replaced by “TARGET”, any other names replaced by “POLICE”
-  plabel: for the training data possibly erroneous labels obtained using weak supervision and gold labels for the test data – in this project, only the labels of the test data will be used
-  sent_org: the original mention

Compared to the Data Preprocessing with RegExt Tutorial 
(https://github.com/knodle/knodle/blob/feature/%23299_police_killing_dataset/examples/data_preprocessing/police_killing/data_preprocessing_with_regex.ipynb), 
where RegEx are used in order to cover various different ways a sentence might indiciate that someone was killed 
by the police, the rules that are used in this tutorial are simple word pairs (one of the killing and one of 
the police related keywords each). A rule matches a sample if both of the words in the wordpair are part of 
the sample. Except the rules that are used, everything is done in the same way as in the alternative tutorial. 

Reference: Keith, Kathrine A. et al. (2017): Identifying civilians killed by police with distantly 
supervised entity-event extraction. In: Proceedings of the 2017 Conference on Empirical Methods 
in Natural Language Processing. doi: 10.18653/v1/D17-1163 (URL: https://aclanthology.org/D17-1163/)
"""


#Imports

import pandas as pd
import json
import os
import numpy as np
import re
import sys
import scipy.sparse as sp

from tqdm import tqdm
from pathlib import Path
from joblib import dump
from typing import List, Dict

from minio import Minio

sys.path.append('..')
from data_to_mapping_rules_labels_t import get_mapping_rules_labels_t



"""
Get the data

First of all, the file names for the output at the end of this notebook are defined. After that, 
the raw data can be downloaded from MinIO.
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
    "train.json", "test.json", "keywords.csv"
]
for file in tqdm(files):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("datasets/police_killing/", file),
        file_path=os.path.join(data_path, file),
    )


"""
Get the keywords

Downloading the keywords from MiniO that will later be used to create the rules.
"""
keywords = pd.read_csv(os.path.join(data_path, "keywords.csv"))


"""
Get the Train data**

We read the downloaded data and convert it to a Pandas Dataframe. 
For now, we take only the samples for the train data and the samples as 
well as the labels for the test data. In the end, we will also need the name 
of the person in case it turns out they were killed by the police. However, 
in this step their name should be replaced by the TARGET symbol. Therefore, 
we only take the values for the "sent_alter" key and rename them to "samples".
"""

def get_train_data(data_path: str) ->pd.DataFrame:
    with open(os.path.join(data_path, "train.json"), 'r') as data:
        train_data = [json.loads(line) for line in data] # a list of dicts
    df_train_sent_alter = pd.DataFrame(train_data, columns = ["sent_alter"]).rename(columns={"sent_alter": "samples"})
    return df_train_sent_alter

df_train = get_train_data(data_path)



"""
Get the Dev and Test Data

Since the SLANG Lab provides only train and test data,
 but no development data, the part of the test data will be used as a development set. The samples for the development 
 data will be selected randomly to avoid imbalances of positive and negative samples in dev and test data.

The parameter *used_as_dev* reflects the amount of the gold data that should be used for development instead of testing. It is set to 30% for now, but can be changed depending on the task definition.
"""

used_as_dev = 30
print(f"{used_as_dev}% of the test data will be used for develoment.")


def get_dev_test_data(data_path: str) -> pd.DataFrame:
    with open(os.path.join(data_path, "test.json"), 'r') as data:
        dev_test_data = [json.loads(line) for line in data]
    dev_test_sent_alter = pd.DataFrame(dev_test_data, columns = ["sent_alter", "plabel"]).rename(columns={"sent_alter": "samples", "plabel": "label"})
    df_dev = dev_test_sent_alter.sample(n = int(round((dev_test_sent_alter.shape[0]/100)*used_as_dev))).reset_index(drop = True)
    df_test = dev_test_sent_alter.drop(df_dev.index).reset_index(drop = True)
    return df_dev, df_test

df_dev, df_test = get_dev_test_data(data_path)



#Some Statistics: 

# Count of samples
print(f"Number of samples:")
print(f"Train data: {df_train.shape[0]}")
print(f"Development data: {df_dev.shape[0]}")
print(f"Test data: {df_test.shape[0]}")

# Positive and negative instances in dev and test data
positive_dev = df_dev.groupby("label").count()["samples"][1]
negative_dev = df_dev.groupby("label").count()["samples"][0]
positive_test = df_test.groupby("label").count()["samples"][1]
negative_test = df_test.groupby("label").count()["samples"][0]
print(f"In the develoment data, {positive_dev} ({(100/df_dev.shape[0])*positive_dev}%) instances are positive and {negative_dev} instances ({(100/df_dev.shape[0])*negative_dev}%) are negative.")
print(f"In the test data, {positive_test} ({(100/df_test.shape[0])*positive_test}%) instances are positive and {negative_test} instances ({(100/df_test.shape[0])*negative_test}%) are negative.")


"""
Output classes

Our task is to find out whether a sentence describes the killing of a person by the police or 
does not. That means, it is a binary classification task with two output classes. The number 
of classes is defined with the *num_classes* parameter.

"""

num_classes = 2


"""

Get the rules

In the paper of Keith et al. (2017), two lists of police- and killing-related words are used to extract 
the relevant mentions: 
# 
# "*These lists were semi-automatically constructed by looking up the nearest neighbors of 
“police” and “kill” (by cosine distance) from Google’s public release of word2vec vectors 
pretrained on a very large (proprietary) Google News corpus and then manually excluding a 
small number of misspelled words or redundant capitalizations (e.g. “Police” and “police”).*" (Keith et al. p. 11)
# 
# The keywords are saved in a CSV file that we have already downloaded and read in as a Pandas Dataframe. 
Now we will use it to create the rules. Each rule is a pair of a police and a killing word, both of 
these words must appear in the sample.
"""

def get_rule2id(keywords: pd.DataFrame) -> Dict:
    rule2rule_id = {}
    rule_id = 0
    for police_word in keywords["police_words"]:
        for kill_word in keywords["kill_words"].dropna():
            rule2rule_id[f'{police_word} {kill_word}'] = rule_id
            rule_id += 1
    return rule2rule_id

rule2rule_id = get_rule2id(keywords)


"""
# Secondly, we create a dictionary assigning all rules to their label. There are only two classes 
(someone was killed by the police or was not killed by the police). Since there are no rules 
indicating that someone was **not** killed by the police, all rules indicate the positive class 1. 
Therefore, all values of the rule2label dictionary, containing the rule IDs as keys, can be set to 1.

"""
rule2label = {rule_id: 1 for rule_id in rule2rule_id.values()}


"""
Thirdly, a dictionary mapping the labels to their ID as well as a dictionary mapping the 
ID to the corresponding label are required for the further preprocessing. As there are only 
two classes, this can be done manually. 
"""

label2label_id ={"negative":0, "positive":1}
label_id2label = {0: "negative", 1: "positive"}



"""
Build the T matrix (rules x classes)

The rows of the T matrix are the rules and the columns the classes. The T matrix is one-hot 
encoded. (1 for a rule and its corresponding class.) It will be imported from the data_preprocessing 
folder of Knodle examples, since the same function can be used in several preprocessing tutorials. 
"""

mapping_rules_labels_t = get_mapping_rules_labels_t(rule2label, num_classes)



"""
Buidling the Z matrix (instances x rules)

 Get the Train Data

For now, the word pairs contained in the *rule2rule_id* dictionary are simple strings. 
To use them as actual rules, we should convert them to regexes firstly to be able to 
look for the exact words in the word pair. For example, the words in the rules must 
not end in another word character (a-z). If the killing word in a rule is, for instance, 
\"kill\", only sentences containing the word \"kill\" should be matched and no sentences 
containing words like \"kills\" or \"killed\" because there are separate rules for these words. 
(This is ensured by appending "\W" - the word not followed by any letter - to both the police 
 and the killing word contained in a rule.) Technically, it is realized by creating a dictionary 
containing the strings of word pairs as keys and a list of the corresponding RegEx as values.

Secondly, we apply the regexes to the samples. In the beginning, everything is stored in a list of dictionaries (one for each individual sample in the train data). The dictionaries contain the sample, a list of the rules matched in it, and a list of the corresponding rule IDs. After that, we convert the list of dictionaries to a Pandas Dataframe.

"""

def convert_rules2regex(rule2rule_id: Dict) -> Dict:
       
        return {rule: [f'{rule.split()[0]}\W', f'{rule.split()[1]}\W'] for rule in rule2rule_id}

def get_data_for_df(data: pd.DataFrame, searches: Dict) -> List:

    data_for_df = []

    for sample in tqdm(data["samples"].drop_duplicates()):
        data_dict = {}
        data_dict["samples"] = sample
        data_dict["rules"] = []
        data_dict["enc_rules"] = []
   
        for rule, search in searches.items():
            if re.search(search[0], sample.lower()) and re.search(search[1], sample.lower()):
                data_dict["rules"].append(rule)
                data_dict["enc_rules"].append(rule2rule_id[rule])

        data_for_df.append(data_dict)

    return data_for_df


def get_df(rule2rule_id: Dict, data: pd.DataFrame) -> pd.DataFrame:
    
    searches = convert_rules2regex(rule2rule_id)
    data_for_df = get_data_for_df(data, searches)
    df = pd.DataFrame.from_dict(data_for_df)
    df = df.reset_index()
       
    return(df)

train_data = get_df(rule2rule_id, df_train)


"""
# ### Get the Dev and Test data
# 
# Just as for the train data, we need a Dataframe with a sample, its corresponding rules, and the rule IDs. Moreover, we need to add the labels and the label IDs that we obtained earlier when reading the test data. We do this by merging the new Dataframe with sample, rule, and rule encoding only with the development and test Dataframes that contain the labels.
"""


def get_dev_test_df(rule2rule_id: Dict, data: pd.DataFrame, label_id2label: Dict) -> pd.DataFrame:

    test_data_without_labels = get_df(rule2rule_id, data)
    test_data = test_data_without_labels.merge(data, how='inner').rename(columns={"label": "enc_labels"})
    test_data["labels"] = test_data['enc_labels'].map(label_id2label)
    
    return test_data


dev_data = get_dev_test_df(rule2rule_id, df_dev, label_id2label)
test_data = get_dev_test_df(rule2rule_id, df_test, label_id2label)



"""
Convert Dataframes to (Sparse) Matrices

The train, test, and development data that we just stored as Pandas Dataframes should 
now be converted into a Scipy sparse matrix. The rows of the sparse matrix are the samples 
and the columns are the rules (i.e., a cell is 1 if the corresponding rule matches the 
corresponding sample, 0 otherwise). We initialize it as an array in the correct size 
(samples x rules), fill it with 1s and 0s, and convert it to a sparse matrix at the end.
"""


def get_rule_matches_z_matrix(df: pd.DataFrame) -> sp.csr_matrix:

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


# Saving the files

Path(os.path.join(data_path, "processed_keywords")).mkdir(parents=True, exist_ok=True)

dump(sp.csr_matrix(mapping_rules_labels_t), os.path.join(data_path, "processed_keywords", T_MATRIX_TRAIN))

dump(train_data["samples"], os.path.join(data_path, "processed_keywords", TRAIN_SAMPLES_OUTPUT))
train_data["samples"].to_csv(os.path.join(data_path, "processed_keywords", TRAIN_SAMPLES_CSV), header=True)
dump(train_rule_matches_z, os.path.join(data_path, "processed_keywords", Z_MATRIX_TRAIN))

dump(dev_data[["samples", "labels", "enc_labels"]], os.path.join(data_path, "processed_keywords", DEV_SAMPLES_OUTPUT))
dev_data[["samples", "labels", "enc_labels"]].to_csv(os.path.join(data_path, "processed_keywords", DEV_SAMPLES_CSV), header=True)
dump(dev_rule_matches_z, os.path.join(data_path, "processed_keywords", Z_MATRIX_DEV))

dump(test_data[["samples", "labels", "enc_labels"]], os.path.join(data_path, "processed_keywords", TEST_SAMPLES_OUTPUT))
test_data[["samples", "labels", "enc_labels"]].to_csv(os.path.join(data_path, "processed_keywords", TEST_SAMPLES_CSV), header=True)
dump(test_rule_matches_z, os.path.join(data_path, "processed_keywords", Z_MATRIX_TEST))


"""
Rule Accuracy and some statistics

For the rule accuracy, we will compare the weak labels of the test data to the gold labels 
to check how reliable the rules are. 
"""


positive_test_samples = test_data[test_data.enc_labels == 1].shape[0]
negative_test_samples = test_data[test_data.enc_labels == 0].shape[0]
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
matched_instances = test_data["enc_rules"].str.len() != 0

for row in tqdm(range(test_data.shape[0])):
    if test_data.loc[row]["enc_labels"] == 1: #the true label is 1
        if matched_instances[row]: #the predicted label is 1
            true_positive += 1
        else: #the predicted label is 0
            false_negative += 1
    else: #the true label is 0
        if matched_instances[row]: #the predicted label is 1
            false_positive += 1
        else: #the predicted label is 0
            true_negative += 1
                 
true_positive_percent = (100 / positive_test_samples) * true_positive
true_negative_percent = (100 / negative_test_samples) * true_negative

print(f"Out of {test_data.shape[0]} samples in the testdata, {positive_test_samples} samples are positive and {negative_test_samples} are negative.\n") 
print(f"By using only the rules to obtain weak labels, {true_positive_percent}% of all positive samples are matched by a rule and therefore labeled as positive. {true_negative_percent}% of all negative samples are correctly classified as negative.\n") 
print(f"True positives: {true_positive} \nTrue negatives: {true_negative} \nFalse positives: {false_positive} \nFalse negatives: {false_negative}")
