## import libraries

import os
import re
import codecs
import pandas as pd
import numpy as np
from typing import List
from minio import Minio
from tqdm import tqdm

# Define the path to the folder where the data will be stored
data_path = "../../../data_from_minio/wiener_diarum_toponyms"
os.makedirs(data_path, exist_ok=True)
os.path.join(data_path)

# Get the data

client = Minio("knodle.cc", secure=False)
files = [
    "annotations_3-21_v4.html", "timemachine_evaluation_v1_edited_corrected.jsonl" 
]
for file in tqdm(files):
    client.fget_object(
        bucket_name="knodle",
        object_name=os.path.join("datasets/wiener_diarum_toponyms/", file),
        file_path=os.path.join(data_path, file),
    )

#Load html file

file = codecs.open('annotations_3-21_v4.html', "r", "utf-8")
sterbelisten_html = file.read()

#Cleaning the text first from both html tags and xml tags and then in a second step clean the intro and replace with /n

# get rid of the title of the issue and the xml tags by just replacing the whole intro with ""
sterbelisten_strip_html_1 = re.sub(r"<h2>.*</h2><h3>.*xml \|.*\d+</h3><br/>","",sterbelisten_html)

#now working on all the tags around the sentences and weird characters within the words
sterbelisten_strip_html1 = re.sub(r"<hr/>|<p>|</p>|#+|=| =|<html>|</html>|\r|\b \.|\b# \.|(|)","",sterbelisten_strip_html_1)#<mark>|</mark>

sterbelisten_strip_html2 = re.sub(r"    \b","",sterbelisten_strip_html1)

#create an empty list and split the text at line break, creating an element for each sentence

sterbeliste = []
sterbeliste = sterbelisten_strip_html2.split("\n\n\n")
#print(sterbeliste[0])

# check <mark> tags

print(f"Number of <mark> tags: {len(re.findall('<mark>',sterbelisten_html))}")
print(f"Number of </mark> tags: {len(re.findall('</mark>',sterbelisten_html))}")
print((f"Number of unclosed tags: {len(re.findall('<mark>',sterbelisten_html))-len(re.findall('</mark>',sterbelisten_html))}"))

def get_location_id(location: List, locations: List) -> int:
    '''
    this function loops through the list of place names, 
    if it doesn't find a given place name it adds the place name to the list
    and returns the new list 
    
    input: the list or single word that is part of a place name
    output: list of place names 
    '''
    for word in location:
        if word not in locations:
            locations[word]= len(locations)
            return locations[word]

# we create a dictionary locations that contains the place name as a key and the ID as a value

locations={}

# we collect our tagged sentences in samples, these contain None for each word that is no place name and the corresponding ID to each word that is part of a place name 
samples = []

# we collect a list with each sentence that is a list containing each word as its elements
clean_sterbeliste=[]

# we loop through each sentence
for sentence in sterbeliste:
    labels=[]
    sentence = re.sub(r"<mark>","<start> ",sentence)
    sentence = re.sub(r"</mark>"," <end>",sentence)
    sentence = re.sub(r"\(|\)|/","",sentence)
    sentence = sentence.split(" ")
    sentence = list(filter(None,sentence))
    if_loc= False
    for word in sentence:
        if word=='<start>':
            if_loc= True
            location = []
        elif word=='<end>':
            labels +=[get_location_id(location, locations)] * len(location)
            if_loc= False
        else:
            if if_loc:
                location.append(word)
            else:
                labels.append(None)
    samples.append(labels)
    sentence= [word for word in sentence if word !="<start>" if word!="<end>"]
    clean_sterbeliste.append(sentence)

print(f"Length of the sentence: {len(sentence)}")
print(f"Sample_tagging:{samples[0]}")
print(f"Number of place names: {len(locations)}")
print(f"=======================================")
assert len(sentence) == len(labels)
#print(f"Sentence and labeles: {[(word, label) for (word, label) in zip(sentence, labels)]}")
print(f"Cleaned Sentence: {clean_sterbeliste[0]}")

'''
X Matrix

Dimensions # sentences x # sentences (one column samples df, len(sample))

'''
X = pd.DataFrame(clean_sterbeliste)

'''
T Matrix
Dimensions: # place names x # 2

'''
import numpy as np
a= np.zeros(len(locations))
b= np.full(len(locations),1)

T = np.stack((b,a.T))
T.T.shape

'''
Z Matrices

In the next function, we are building the Z Matrixes by looping through all samples.
Each sample is compared with the list of locations, if the location is in the sample,
the new list "matched_loc" gets an entry 1 otherwise a 0 is added
Then the list is transformed to an numpy array and brought into shape:

Dimensions
number of place names x number of words

'''
locations_list = list(range(0,(len(locations))))


for sample in samples:
    matched_loc= [1 if i == j else 0 for i in sample for j in locations_list]
    i=len(sample)
    Z = np.array(matched_loc)
    Z = np.reshape(Z, (i,len(locations_list)))
    
# Needed: a way to save them with their index number!


