# -*- coding: utf-8

"""
Preprocessing of MIMIC-CXR dataset

This file illustrates how weak supervision can be applied on medical images 
and the corresponding reports. Since there are two sources of data (images and 
reports) we establish a double layer weak supervision. 

In this example the MIMIC-CXR dataset is used. There are to versions of this 
dataset: 

[MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) Database (Johnson, 
Pollard et al. (2019) is a large publicly available dataset of chest X-rays 
including radiology reports. It contains 377110 images and 227835 radiographic 
. A radiographic study consists of one report and one or multiple images. 

[MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) Database 
(Johnson, Lungren et al. (2019) bases on MIMIC-CXR. It additionally includes 
weak labels which are derived from the radiology reports using CheXpert labler 
(Irvin, Rajpurkar et al. 2019) and the images are in JPG format instead of 
DICOM format. 

Neither versions of the MIMIC-CXR dataset have gold labels. Since both the 
CheXpert data and the MIMIC-CXR data contain chest X-Rays, the CheXpert labler 
was used in the MIMIC-CXR-JPG Database to obtain weak labels. We will use a 
small subset of the MIMIC images and their weak labels in the data 
preprocessing to finetune our image encoder CNN. Apart from that we do not 
touch any labels until evaluation.
To evaluate our results in the end, we apply the trained model (Knodle output) 
to the validation data of the CheXpert dataset, since they have gold labels. 

In the data preprocessing we build the three input matrices knodle requires:
 * The rules are generated from the CheXpert Labler phrases. The phrases 
   contain mentions (synonyms or related words) for each class, which we use to 
   build our T matrix, so the "rule to class" matrix.
 * The Z matrix, so the "rule matches" matrix is generated from the reports 
   and the rules. 
 * The images are encoded with a CNN. We try two different approaches: 
     1) CNN with pretrained weight without finetuning and 
     2) CNN with pretrained weights and finetuning. Therefore, we need the weak 
       labels.  

"""

import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import random
import copy
import csv
import itertools

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from typing import Dict
from joblib import dump
from PIL import Image


# set directory
# os.chdir("")

# set n between 1 and 377110
n = 1000
# PhysioNet
USERNAME = "your_username_her"
PASSWORD = "your_pw_here"
download = False

# Files that will be created:
Z = "rule_matches_z.lib"
T = "mapping_rules_labels_t.lib"
X = "train_X.lib"
X_finetuned = "train_X_finetuned.lib"
X_test = "X_test.lib"
y_test = "gold_labels_test.lib"


if download:
    # downloads from mimic-cxr 
    url = ["wget -N -c -np --user=", USERNAME, " --password=", PASSWORD, 
           " https://physionet.org/files/mimic-cxr/2.0.0/"]
    
    command = "".join(url+["cxr-record-list.csv.gz"]) # paths to images
    os.system(command)
    command = "".join(url+["cxr-study-list.csv.gz"]) # paths to reports
    os.system(command)
    command = "".join(url+["mimic-cxr-reports.zip"]) # folder of all reports
    os.system(command)

    # downloads from mimic-cxr-jpg
    url = ["wget -N -c -np --user=", USERNAME, " --password=", PASSWORD, 
           " https://physionet.org/files/mimic-cxr-jpg/2.0.0/"]
    command = "".join(url+["mimic-cxr-2.0.0-chexpert.csv.gz"]) # chexpert output 
                                                               # for mimic dataset
    os.system(command)

    # NOW UNZIP ALL DOWNLOADED FILES AND THE REPORT FOLDER WITH 7zip

##############################################################################
# MIMIC-CXR-JPG images
############################################################################## 
record_list_all = pd.read_csv("cxr-record-list.csv")
study_list = pd.read_csv("cxr-study-list.csv").to_numpy()

# restrict records
# only want to include studies where there are two images
# only want to include one per person
two_records_per_study = record_list_all.groupby("study_id").count() == 2
two_records_per_study = two_records_per_study.rename(columns={"subject_id":"two_rec"})
record_list = pd.merge(record_list_all, two_records_per_study["two_rec"], 
                              how = "left", on= ["study_id"])
record_list_reduced = record_list[record_list["two_rec"]]
record_list_reduced = record_list_reduced.groupby("subject_id").head(2)
record_list_pd = record_list_reduced.drop(columns = ["two_rec"])
record_list = record_list_pd.to_numpy()

# draw a random subset 
random.seed(10)
study_indices = random.sample(range(int(len(record_list)/2)), n)
record_indices = [element * 2 for element in study_indices]+[element * 2+1 for element in study_indices]
record_indices.sort() 
record_indices = record_indices[:10000] ########################remove in final version
if download:            
    for i in tqdm(record_indices):
        path = record_list[i,3]
        url = ["wget -N -c -np --user=", USERNAME, " --password=", PASSWORD, 
               " https://physionet.org/files/mimic-cxr-jpg/2.0.0/",
               path, " -P ", path.replace("/"+record_list[i,2]+".dcm", "")]
        command = "".join(url).replace(".dcm", ".jpg")
        os.system(command)
        
        
    # load reports and save all in one csv
    with open("mimic_cxr_text.csv", "w", newline="", encoding="utf-8") as f:
        for i in tqdm(range(len(study_list))):
            with open("".join(["mimic-cxr-reports/", study_list[i,2]])) as f_path:
                text = "".join(f_path.readlines())
            text = text.replace("\n", "")
            text = text.replace(",", "")
            start = text.find("FINDINGS:")
            end = text.find("IMPRESSION:")
            findings = text[start:end]
            impressions = text[end:len(text)]
            row = [study_list[i,0],study_list[i,1], findings, impressions]
            csvwriter = csv.writer(f)
            csvwriter.writerow(row)

# open report csv
reports = pd.read_csv("mimic_cxr_text.csv", 
                      names = ["subject_id","study_id", "findings", "impressions"], 
                      na_values=".")

print("average length findings section:", 
      np.mean(reports["findings"].str.len()))

print("average length impression section:", 
      np.mean(reports["impressions"].str.len()))

print("number of NAs in findings and impressions:\n", 
      pd.isna(reports[["findings", "impressions"]]).sum())

# if impression is missing insert finding
reports.impressions.fillna(reports.findings, inplace=True)
#if neither are there, we do not analyse this study -> drop
del reports["findings"]
reports_processed = reports.dropna()

# merge reports to record_list
record_report_list = pd.merge(record_list_pd, reports_processed, 
                              how = "left", on= ["study_id","subject_id"])

# only first n rows, drop nas
input_list_pd = record_report_list.iloc[record_indices,:].dropna()
input_list = input_list_pd.to_numpy()
# save new n
n = len(input_list)


##############################################################################
# make rules from reports and Chexpert-labler
##############################################################################

labels_chexpert = pd.read_csv("mimic-cxr-2.0.0-chexpert.csv")
labels = {id: cat for (cat, id) in enumerate(labels_chexpert.columns[2:16])}
# lower case & replace whitespace with _
classes = [string.lower().replace(" ", "_") for string in labels]
num_classes = len(classes)
labels2ids = {classes[i]:i for i in range(num_classes)}
if download:
    # create folder
    os.makedirs("".join([os.getcwd(),"/chexpert_rules"]))
    # store files in folder
    for i in range(len(classes)):
        os.system("".join(["curl https://raw.githubusercontent.com/stanfordmlgroup/chexpert-labeler/master/phrases/mention/", 
                           classes[i], ".txt ", "-o chexpert_rules/", classes[i], ".txt"]))

# make T matrix
lines = {}
for i in range(len(classes)):
    with open("".join(["chexpert_rules/", classes[i], ".txt"])) as f:
        lines[classes[i]] = [each_string.replace("\n", "") for each_string in f.readlines()]
          
mentions = pd.DataFrame({"label": label, "rule": rule} for (label, rule) in lines.items())
mentions.head()

rules = pd.DataFrame([i for i in itertools.chain.from_iterable(mentions["rule"])], columns = ["rule"])
rules["rule_id"] = range(len(rules))
rules["label"] = np.concatenate([
    np.repeat(mentions["label"][i], len(mentions["rule"][i])) for i in range(num_classes)])
rules["label_id"] = [labels2ids[rules["label"][i]] for i in range(len(rules))]
rules.head()

rule2rule_id = dict(zip(rules["rule"], rules["rule_id"]))
rule2label = dict(zip(rules["rule_id"], rules["label_id"]))

def get_mapping_rules_labels_t(rule2label: Dict, num_classes: int) -> np.ndarray:
    """ Function calculates t matrix (rules x labels) using the known correspondence of relations to decision rules """
    mapping_rules_labels_t = np.zeros([len(rule2label), num_classes])
    for rule, labels in rule2label.items():
        mapping_rules_labels_t[rule, labels] = 1
    return mapping_rules_labels_t

mapping_rules_labels_t = get_mapping_rules_labels_t(rule2label, len(labels2ids))
mapping_rules_labels_t[0:5,:]
mapping_rules_labels_t.shape

dump(mapping_rules_labels_t, T)

len(np.unique(rules["rule"])) == len(rules["rule"])
rules_size = rules.groupby("rule").size() 
rules_size[np.where(rules_size > 1)[0]]
# rule defib appears for two different classes

# make Z matrix     
def get_rule_matches_z (data: np.ndarray, num_rules: int) -> np.ndarray:
    """
    Function calculates the z matrix (samples x rules)
    data: np.array (reports)
    output: sparse z matrix
    """
    rule_matches_z = np.zeros([len(data), num_rules])
    for ind in range(len(data)):
        for rule, rule_id in rule2rule_id.items():
            if rule in (data[ind]):
                rule_matches_z[ind, rule_id] = 1
    return rule_matches_z

rule_matches_z = get_rule_matches_z(input_list[:,4], (len(rule2rule_id)+1))

dump(rule_matches_z, Z)
######################################################################
# image - encoding: 
# without finetuning
######################################################################
class mimicDataset(Dataset):
    
    def __init__(self, path, load_labels = False):
        "initialization"
        self.path = path
        self.load_labels = load_labels
        
    def __len__(self):
        "total number of samples"
        return len(self.path)
    
    def __getitem__(self, index):
        "one sample of data"
        # Select sample
        image = Image.open(self.path[index,3].replace(".dcm", ".jpg")).convert("RGB")
        X = self.transform(image)
        if self.load_labels: # for the second approach with finetuning
            label = self.path[index,5]      
            return X, torch.tensor(label)
        else:
            return X # for the first approach without labels
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

model = models.resnet50(pretrained=True)
modules = list(model.children())[:-1]
model=torch.nn.Sequential(*modules)
for p in model.parameters():
    p.requires_grad = False
    
model.eval()
# apply modified resnet50 to data
dataloaders = DataLoader(mimicDataset(input_list), batch_size=n,num_workers=0)
    
data = next(iter(dataloaders))
with torch.no_grad():
    features_var = model(data)
    features = features_var.data 
    train_X = features.reshape(n,2048).numpy()
    
# concatenate both image embeddings of a study to one 

# save feature matrix
dump(train_X, X)

##############################################################################
# Finetuning a pretrained CNN and extracting the second last layer as features
##############################################################################

# For finetuning the CNN, we use the weak labels from Chexpert
labels = {id: cat for (cat, id) in enumerate(labels_chexpert.columns[2:16])}
# initialise labels with 0
labels_chexpert["label"] = 0
labels_list = labels_chexpert.columns.to_numpy()
# iterate through labels: 
# three cases: only one, non, or multiple diagnoses
for i in tqdm(range(len(labels_chexpert))):
    # which labels are 1? 
    label_is1 = labels_chexpert.iloc[i,:] == 1.0
    if (sum(label_is1)==1):
       labels_chexpert.iloc[i,16] = labels_list[label_is1]
    elif sum(label_is1) > 1:
        labels_chexpert.iloc[i,16] = random.choice(labels_list[label_is1])
    else: 
        labels_chexpert.iloc[i,16] = "No Finding"
        
    
        
# merge labels with records and reports
input_list_labels_pd = pd.merge(input_list_pd, 
                                    labels_chexpert.iloc[:,[0,1,16]], 
                                    how = "left", 
                                    on = ["study_id","subject_id"])

print("classes proportions:", 
      input_list_labels_pd.groupby("label").size()/len(input_list_labels_pd))
# keep in mind that the dataset is unbalenced

# Changing names to indices
for i in tqdm(range(len(input_list_labels_pd))):
 input_list_labels_pd.iloc[i,5] = labels.get(input_list_labels_pd.iloc[i,5])

# convert to numpy
input_list_labels = input_list_labels_pd.to_numpy()
dump(input_list_labels, "input_list_labels.lib")

input_list_labels = load("input_list_labels.lib") ##################### remove in final version
# finetuning
# m ... number of samples used for finetuning
m = min(750,n)

# 80% training and 20% validation
n_train = round(m*0.8)
indices_train = random.sample(range(m),n_train)

input_train = input_list_labels[:m,:][indices_train,:]
input_validate = np.delete((input_list_labels[:m,:]),indices_train, axis = 0)

# Since the dataset is unbalanced, we use a weighted sampler 
class_counts = np.zeros(num_classes)
for i in range(num_classes): 
    class_counts[i] = sum(input_train[:,5]==i)
weight = 1/class_counts
sample_weights = np.array([weight[t] for t in input_train[:,5]])
sample_weights = torch.from_numpy(sample_weights)
sample_weights = sample_weights.double()
sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, 
                                                 num_samples=len(sample_weights))

dataset = {"train" : mimicDataset(input_train, load_labels = True),
           "val": mimicDataset(input_validate, load_labels = True)}

dataloaders = {"train": DataLoader(dataset["train"] , batch_size=4, num_workers=0, sampler = sampler),
               "val": DataLoader(dataset["val"] , batch_size=4, num_workers=0 )}


dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print("Best val Acc: {:4f}".format(best_acc), )

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model = models.resnet50(pretrained=True)

# set output size to 14 (number of classes)
model.fc = nn.Linear(1000, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=3)

# delete last layer of the network
modules = list(model.children())[:-1]
model=torch.nn.Sequential(*modules)
for p in model.parameters():
    p.requires_grad = False
    
model.eval()
# apply modified resnet50 to data
dataloaders = DataLoader(mimicDataset(input_list_labels, load_labels = True), batch_size=n,num_workers=0)
    
data, weak_labels = next(iter(dataloaders))
with torch.no_grad():
    features_var = model(data)
    features = features_var.data 
    train_X_finetuned = features.reshape(n,2048).numpy()

# save features matrix
dump(train_X_finetuned, X_finetuned)

##############################################################################
# Test data preprocessing 
# - riqueres a model defined for image encoding
##############################################################################
# download Chexpert data and unzip it to the same directory
validation_set_pd = pd.read_csv("CheXpert-v1.0-small/CheXpert-v1.0-small/valid.csv")
validation_set = validation_set_pd.to_numpy()

labels_test_list = validation_set_pd.columns[5:19].to_numpy()
class chexpertDataset(Dataset):
    
    def __init__(self, path):
        "initialization"
        self.path = path
        
    def __len__(self):
        "total number of samples"
        return len(self.path)
    
    def __getitem__(self, index):
        "one sample of data"
        # Select sample
        image = Image.open("".join("CheXpert-v1.0-small/" + self.path[index,0])).convert("RGB")
        X = self.transform(image)
        
        label_is1 = self.path[index,5:19] == 1.0
        if (sum(label_is1)==1):
            y = labels.get(labels_test_list[np.where(label_is1)[0][0]])
        elif sum(label_is1) > 1:
            y = labels.get(labels_test_list[random.choice(np.where(label_is1)[0])])
        else: 
            y = 8 #no finding
        
        return X, torch.tensor(y)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

n_test = len(validation_set)
dataloaders = DataLoader(chexpertDataset(validation_set), batch_size=n_test,num_workers=0)
    
data_test, gold_labels_test = next(iter(dataloaders))

model.eval()
with torch.no_grad():
    features_var = model(data_test) # same model as used with training data
    features = features_var.data 
    test_X = features.reshape(n_test,2048).numpy()

# save test data
dump(test_X, X_test)
dump(gold_labels_test, y_test)

