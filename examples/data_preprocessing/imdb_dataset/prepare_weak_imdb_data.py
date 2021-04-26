#!/usr/bin/env python
# coding: utf-8

"""
IMDB Dataset - Create Weak Supervision Sources and Get the Weak Data Annotations

This notebook shows how to use keywords as a weak supervision source on the example of a well-known IMDB Movie Review dataset, which targets a binary sentiment analysis task.

The original dataset has gold labels, but we will use these labels only for evaluation purposes, since we want to test models under the weak supervision setting with Knodle. The idea behind it is that you don't have a dataset which is purely labeled with strong supervision (manual) and instead use heuristics (e.g. rules) to obtain a weak labeling. In the following tutorial, we will look for certain keywords expressing positive and negative sentiments that can be helpful to distinguish between classes. Specifically, we use the [Opinion lexicon](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html) of the University of Illinois at Chicago.

First, we load the dataset from a Knodle dataset collection. Then, we will create [Snorkel](https://www.snorkel.org/) labeling functions from two sets of keywords and apply them to the IMDB reviews. Please keep in mind, that keyword matching can be done without Snorkel; however, we enjoy the transparent annotation functionality of this library in our tutorial. 
Each labeling function (i.e. keyword) will be further associated with a respective target label. This concludes the annotation step.

To estimate how good our weak labeling works on its own, we will use the resulting keyword matches together with a basic majority vote model. Finally, the preprocessed data will be saved in a knodle-friendly format, so that other denoising models can be trained with the IMDB dataset.
The IMDB dataset available in the Knodle collection was downloaded from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) in January 2021. 
"""

import os
from joblib import dump
from tqdm import tqdm

import pandas as pd 
import numpy as np 
from scipy.sparse import csr_matrix

from bs4 import BeautifulSoup
from snorkel.labeling import LabelingFunction, PandasLFApplier, LFAnalysis

from knodle.transformation.rule_label_format import transform_snorkel_matrix_to_z_t
from knodle.transformation.majority import z_t_matrices_to_majority_vote_labels

# client to access the dataset collection
from minio import Minio
client = Minio("knodle.dm.univie.ac.at", secure=False)

# init pandas parameters
tqdm.pandas()
pd.set_option('display.max_colwidth', -1)

# Constants for Snorkel labeling functions
POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1
COLUMN_WITH_TEXT = "reviews_preprocessed"


## Download the dataset
data_path = "../../../data_from_minio/imdb"

# Together with the IMDB data, let us also collect the keywords that we will need later.
files = [("IMDB Dataset.csv",), ("keywords", "negative-words.txt"), ("keywords", "positive-words.txt")]

for file in tqdm(files):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("datasets/imdb", *file),
        file_path=os.path.join(data_path, file[-1]),
    )

imdb_dataset_raw = pd.read_csv(os.path.join(data_path, "IMDB Dataset.csv"))
print("Gold label counts")
print(imdb_dataset_raw.groupby("sentiment").count())

# ## Preprocess dataset

# The dataset contains many HTML tags. We'll remove them
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

imdb_dataset_raw[COLUMN_WITH_TEXT] = imdb_dataset_raw[COLUMN_WITH_TEXT].apply(
    lambda x: strip_html(x))

# ## Keywords
# 
# For weak supervision sources we use sentiment keyword lists for positive and negative words.
# https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
# 
# We have downloaded them from the Knodle collection earlier, with the IMDB dataset. 
# 
# After parsing the keywords from separate files, they are stored in a pd.DataFrame with the corresponding sentiment as "label".

positive_keywords = pd.read_csv(os.path.join(data_path, "positive-words.txt"), sep=" ", header=None, error_bad_lines=False, skiprows=30)
positive_keywords.columns = ["keyword"]
positive_keywords["label"] = "positive"

negative_keywords = pd.read_csv(os.path.join(data_path, "negative-words.txt"),
                                sep=" ", header=None, error_bad_lines=False,  encoding='latin-1', skiprows=30)
negative_keywords.columns = ["keyword"]
negative_keywords["label"] = "negative"

all_keywords = pd.concat([positive_keywords, negative_keywords])
all_keywords.label.value_counts()

# remove overlap of keywords between two sentiments
all_keywords.drop_duplicates('keyword', inplace=True)


# ## Labeling Functions
# 
# Now we start to build labeling functions with Snorkel with these keywords and check the coverage.
# 
# This is an iterative process of course so we surely have to add more keywords and regulary expressions ;-) 

def keyword_lookup(x, keyword, label):
    return label if keyword in x[COLUMN_WITH_TEXT].lower() else ABSTAIN

def make_keyword_lf(keyword: str, label: str) -> LabelingFunction:
    """
    Creates labeling function based on keyword.
    Args:
        keyword: what keyword should be look for
        label: what label does this keyword imply

    Returns: LabelingFunction object

    """
    return LabelingFunction(
        name=f"keyword_{keyword}",
        f=keyword_lookup,
        resources=dict(keyword=keyword, label=label),
    )

def create_labeling_functions(keywords: pd.DataFrame) -> np.ndarray:
    """
    Create Labeling Functions based on the columns keyword and regex. Appends column lf to df.

    Args:
        keywords: DataFrame with processed keywords

    Returns:
        All labeling functions. 1d Array with shape: (number_of_lfs x 1)
    """
    keywords = keywords.assign(lf=keywords.progress_apply(
        lambda x:make_keyword_lf(x.keyword, x.label_id), axis=1
    ))
    lfs = keywords.lf.values
    return lfs

all_keywords["label_id"] = all_keywords.label.map({'positive':POSITIVE, 'negative':NEGATIVE})
labeling_functions = create_labeling_functions(all_keywords)


# ### Apply Labeling Functions
# We get a matrix with all labeling functions applied. This matrix has the shape $(instances \times labeling functions)$

applier = PandasLFApplier(lfs=labeling_functions)
applied_lfs = applier.apply(df=imdb_dataset_raw)

print("Shape of applied labeling functions: ", applied_lfs.shape)
print("Number of reviews", len(imdb_dataset_raw))
print("Number of labeling functions", len(labeling_functions))


# ### Analysis 
# 
# Now we can analyse some basic stats about our labeling functions. The main figures are:
# 
# - Coverage: How many labeling functions match at all
# - Overlaps: How many labeling functions overlap with each other (e.g. awesome and amazing)
# - Conflicts: How many labeling functions overlap and have different labels (e.g. awesome and bad)
# - Correct: Correct LFs
# - Incorrect: Incorrect Lfs

lf_analysis = LFAnalysis(L=applied_lfs, lfs=labeling_functions).lf_summary()
print("LF analysis:")
print(lf_analysis)
print("LF analysis mean values:")
print(pd.DataFrame(lf_analysis.mean()))
print("LF analysis median values:")
print(pd.DataFrame(lf_analysis.median()))

# Lets have a look at some examples that were labeled by a positive keyword. You can see, that the true label for some of them is negative.
# consider 1110th keyword
kw = all_keywords.iloc[1110]
print(f"1110th keyword is '{kw.keyword}' and labels with '{kw.label}'")
print()

# sample 2 random examples where the 50th LF assigned a positive label
examples = imdb_dataset_raw.iloc[applied_lfs[:, 1110] == POSITIVE].loc[:2, [COLUMN_WITH_TEXT,'sentiment']]
print("This keyword has matched the following instances with labels:")
print()

for idx, ex in examples.iterrows():
    print("{} - {}".format(ex.sentiment, ex[COLUMN_WITH_TEXT]))
    print()

    
# ## Transform rule matches
# 
# To work with knodle the dataset needs to be transformed into a binary array $Z$
# 
# (shape: `#instances x # rules_`).
# 
# Where a cell $Z_{ij}$ means that for instance i
# 
#     0 -> Rule j didn't match 
#     1 -> Rule j matched
# 
# Furthermore, we need a matrix `mapping_rule_labels` which has a mapping of all rules to labels, stored in a binary manner as well 
# 
# (shape `#rules x #labels`).

rule_matches, mapping_rules_labels = transform_snorkel_matrix_to_z_t(applied_lfs)


# ### Majority Vote
# 
# Now we make a majority vote based on all rule matches. First we get the `rule_counts` by multiplying `rule_matches` with the `mapping_rules_labels`, then we divide it sumwise by the sum to get a probability value. In the end we counteract the divide with zero issue by setting all nan values to zero. All this happens in the `z_t_matrices_to_majority_vote_labels` function.

# the ties are resolved randomly internally, so the predictions might slightly vary
pred_labels = z_t_matrices_to_majority_vote_labels(rule_matches, mapping_rules_labels)

# There are more positive labels predicted by the majority vote.
labels, counts = np.unique(pred_labels, return_counts=True)

# accuracy of the weak labels
acc = (pred_labels == imdb_dataset_raw.sentiment.map({'positive':POSITIVE, 'negative':NEGATIVE})).mean()
print(f"Accuracy of majority voting: {acc}")

      
# ## Save Files
rule_matches_sparse = csr_matrix(rule_matches)
dump(rule_matches_sparse, os.path.join(data_path, "rule_matches.lib"))
dump(mapping_rules_labels,  os.path.join(data_path, "mapping_rules_labels.lib"))

# We also save the preprocessed texts with labels to use them later to evalutate a classifier.
imdb_dataset_raw['label_id'] = imdb_dataset_raw.sentiment.map({'positive':POSITIVE, 'negative':NEGATIVE})
imdb_dataset_raw.to_csv(os.path.join(data_path, 'imdb_data_preprocessed.csv'), index=None)

all_keywords.to_csv(os.path.join(data_path, 'keywords.csv'), index=None)

# Now, we have created a weak supervision dataset. Of course it is not perfect but it is something with which we can compare performances of different denoising methods with. :-) 
