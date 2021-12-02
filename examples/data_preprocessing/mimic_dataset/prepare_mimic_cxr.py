# -*- coding: utf-8

"""
Preprocessing of MIMIC-CXR dataset

This notebook illustrates how week supervision can be applied on X-rays and 
radiology reports.
MIMIC-CXR Database is a large publicly available dataset of chest X-rays 
including radiology reports. It contains 377110 images and 227835 radiographic 
studies. MIMIC-CXR-JPG Database also includes weak labels which are derived 
from the radiology reports using CheXpert labler.

"""

import os
import numpy as np
import pandas as pd
import random
import copy
import torch.optim as optim
import csv
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
import itertools
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from typing import Dict
from joblib import dump, load


#download the csvs 
#"cxr-record-list.csv", (from mimic-cxr)
#"cxr-study-list.csv", 
#"mimic-cxr-2.0.0-split.csv" (from mimic-cxr-jpg)
#"mimic-cxr-2.0.0-chexpert.csv"
#unzip them with 7zip and set your working directoy to this folder
#os.chdir("")


##############################################################################
# MIMIC-CXR-JPG images
############################################################################## 
#set n between 1 and 377110
n = 1000
record_list = pd.read_csv("cxr-record-list.csv").to_numpy()
study_list = pd.read_csv("cxr-study-list.csv").to_numpy()

username = "your_username_here"
password = "your_pw_here"

#image download - run only once
for i in tqdm(range(n)):
    url = ["wget -r -N -c -np --user=", username, " --password=", password, " https://physionet.org/files/mimic-cxr-jpg/2.0.0/",record_list[i,3]]
    command = "".join(url)
    command = "".join([command.replace(".dcm", ""),".jpg"])
    os.system(command)

        
    
##############################################################################
# MIMIC-CXR reports: download and extraction
##############################################################################

username = "your_username_here"
password = "your_pw_here"

url = ["wget -r -N -c -np --user=", username, " --password=", password, " https://physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip"]
command = "".join(url)
os.system(command)

'''
#Now unzip the folder:
#-Either running the code below might work for you, or if not
#-unzip the folder manually with 7zip

#>> start unzipping
from zipfile import ZipFile
with ZipFile("C:/Users/marli/physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip", 'r') as zipObj:
   zipObj.extractall()
#<< end
'''

with open('mimic_cxr_text.csv', 'w', newline='', encoding='utf-8') as f:
    for i in tqdm(range(len(study_list))):
        with open(''.join(["mimic-cxr-reports/", study_list[i,2]])) as f_path:
            text = ''.join(f_path.readlines())
        text = text.replace("\n", "")
        text = text.replace(",", "")
        start = text.find("FINDINGS:")
        end = text.find("IMPRESSION:")
        findings = text[start:end]
        impressions = text[end:len(text)]
        row = [study_list[i,0],study_list[i,1], findings, impressions]
        csvwriter = csv.writer(f)
        csvwriter.writerow(row)

#open
reports = pd.read_csv("mimic_cxr_text.csv", names = ["subject_id","study_id", "findings", "impressions"])

#missing values are denoted by "." in impressions -> nan
ind = np.where(reports['impressions'] == '.')[0]
reports['impressions'][ind][:] = np.nan
print("number of NAs in findings and impressions:", pd.isna(reports[['findings', 'impressions']]).sum())

#if impression is missing insert finding
reports.impressions.fillna(reports.findings, inplace=True)
#if non of both are there, we cannot analyse this study
del reports['findings']
reports_processed = reports.dropna()

#merge reports to record_list
record_list = pd.read_csv("cxr-record-list.csv")
record_report_list = pd.merge(record_list, reports_processed, how = 'left', on= ['study_id','subject_id'])

##############################################################################
#labels and split
##############################################################################
split_list = pd.read_csv("mimic-cxr-2.0.0-split.csv")
labels_chexpert = pd.read_csv("mimic-cxr-2.0.0-chexpert.csv")
#initialise labels with 0
labels_chexpert['label'] = 0
labels_list = labels_chexpert.columns.to_numpy()
#iterate through labels: 
#three cases: only one, non, or multiple diagnoses
for i in tqdm(range(len(labels_chexpert))):
    #which labels are 1? 
    label_is1 = labels_chexpert.iloc[i,:] == 1.0
    if (sum(label_is1)==1):
       labels_chexpert.iloc[i,16] = labels_list[label_is1]
    elif sum(label_is1) > 1:
        labels_chexpert.iloc[i,16] = random.choice(labels_list[label_is1])
    else: 
        labels_chexpert.iloc[i,16] = 'No Finding'

labels = {'Atelectasis':0,
               'Cardiomegaly':1,
               'Consolidation':2,
               'Edema':3,
               'Enlarged Cardiomediastinum':4,
               'Fracture':5,
               'Lung Lesion':6,
               'Lung Opacity':7,
               'Pleural Effusion':8,
               'Pneumonia':9,
               'Pneumothorax':10,
               'Pleural Other':11,
               'Support Devices':12,
               'No Finding':13}        
        
for i in tqdm(range(len(labels_chexpert))):
 labels_chexpert.iloc[i,16] = labels.get(labels_chexpert.iloc[i,16])
        
#merge records, labels and split
record_report_split_list = pd.merge(record_report_list, split_list, how = 'left', on = ['dicom_id', 'study_id','subject_id'])
record_report_split_label_list = pd.merge(record_report_split_list, labels_chexpert.iloc[:,[0,1,16]], how = 'left', on = ['study_id','subject_id'])

print("classes proportions:", record_report_split_label_list.groupby('label').size()/len(record_report_split_label_list))
print("train-val-test split proportions:" ,record_report_split_label_list.groupby('split').size()/len(record_report_split_label_list))
# keep in mind that the dataset is unbalenced


input_list_full = record_report_split_label_list
#save the whole file
dump(input_list_full, "input_list.lib")
#open only first n rows
input_list_pd = load("input_list.lib").iloc[:n,:]
#drop nas
input_list = input_list_pd.dropna().to_numpy()
#save new n
n = len(input_list)
##############################################################################
#make rules from reports and Chexpert-labler
##############################################################################

#download synonym list from chexpert
classes = list(labels)
#lower case
classes = [each_string.lower() for each_string in classes]
#replace whitespace with _
classes = [each_string.replace(" ", "_") for each_string in classes]
labels2ids = {classes[i]:i for i in range(14)}
#create folder
os.makedirs("".join([os.getcwd(),"/chexpert_rules"]))
#store files in folder
for i in range(len(classes)):
    os.system("".join(["curl https://raw.githubusercontent.com/stanfordmlgroup/chexpert-labeler/master/phrases/mention/", 
                       classes[i], ".txt ", "-o chexpert_rules/", classes[i], ".txt"]))

#make T matrix
lines = {}
for i in range(len(classes)):
    with open("".join(["chexpert_rules/", classes[i], ".txt"])) as f:
        lines[classes[i]] = [each_string.replace("\n", "") for each_string in f.readlines()]
          
mentions = pd.DataFrame({'label': label, 'rule': rule} for (label, rule) in lines.items())
mentions.head()

rules = pd.DataFrame([i for i in itertools.chain.from_iterable(mentions['rule'])], columns = ["rule"])
rules['rule_id'] = range(len(rules))
rules['label'] = np.concatenate([
    np.repeat(mentions['label'][i], len(mentions['rule'][i])) for i in range(14)])
rules['label_id'] = [labels2ids[rules['label'][i]] for i in range(len(rules))]
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


len(np.unique(rules['rule'])) == len(rules['rule'])
rules_size = rules.groupby('rule').size() 
rules_size[np.where(rules_size > 1)[0]]
#rule defib appears for two different classes

#make Z matrix     
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

dump(rule_matches_z, "rule_matches_z.lib")
######################################################################
#image - encoding: 
#without finetuning
######################################################################
class mimicDataset(Dataset):
    
    def __init__(self, path):
        'Initialization'
        self.path = path
        #self.y = y
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.path)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = Image.open("".join(["physionet.org/files/mimic-cxr-jpg/2.0.0/",self.path[index,3].replace(".dcm", ".jpg")])).convert('RGB')
        X = self.transform(image)
        label = self.path[index,6]
        
        return X, torch.tensor(label)
    
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
#apply modified resnet50 to data
dataloaders = DataLoader(mimicDataset(input_list[:n,:]), batch_size=n,num_workers=0)
    
data, labels = next(iter(dataloaders))
with torch.no_grad():
    features_var = model(data)
    features = features_var.data 
    all_X = features.reshape(n,2048).numpy()

#save feature matrix
dump(all_X, "all_X.lib")

##############################################################################
#Finetuning a pretrained CNN and extracting the second last layer as features
##############################################################################

input_train = input_list[input_list[:,5] == 'train',:]
input_validate = input_list[input_list[:,5] == 'validate',:]
input_test = input_list[input_list[:,5] == 'test',:]

#Since the dataset is unbalanced, we use a weighted sampler 
class_counts = np.zeros(14)
for i in range(14): class_counts[i] = sum(input_train[:,6]==i)
weight = 1/class_counts
sample_weights = np.array([weight[t] for t in input_train[:,6]])
sample_weights = torch.from_numpy(sample_weights)
sample_weights = sample_weights.double()
sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))

dataset = {'train' : mimicDataset(input_train),
           'val': mimicDataset(input_validate),
           'test':  mimicDataset(input_test)}

dataloaders = {'train': DataLoader(dataset['train'] , batch_size=4, num_workers=0, sampler = sampler),
               'val': DataLoader(dataset['val'] , batch_size=4, num_workers=0 )}


dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
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
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'val':
                    print('predictions',preds)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print('Best val Acc: {:4f}'.format(best_acc), )

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
# set output size to 14 (number of classes)
model.fc = nn.Linear(num_ftrs, 14)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=2)

modules = list(model.children())[:-1]
model=torch.nn.Sequential(*modules)
for p in model.parameters():
    p.requires_grad = False
    
model.eval()
#apply modified resnet50 to data
dataloaders = DataLoader(mimicDataset(input_list[:n,:]), batch_size=n,num_workers=0)
    
data, labels = next(iter(dataloaders))
with torch.no_grad():
    features_var = model(data)
    features = features_var.data 
    all_X_finetuned = features.reshape(n,2048).numpy()

#save features matrix
dump(all_X_finetuned, "all_X_finetuned.lib")


