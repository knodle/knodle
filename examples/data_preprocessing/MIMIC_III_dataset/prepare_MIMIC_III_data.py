# -*- coding: utf-8 -*-
"""
MIMIC III Dataset
Data Preprocessing
"""

import os

import numpy as np
import pandas as pd

from itertools import chain
from tqdm import tqdm
from scipy import sparse
from joblib import dump

##############################################################################
# Define Path

#path = "define_path"
#os.chdir(path)

# Load Data

Notes = pd.read_csv("NOTEEVENTS.csv")
Diagnoses = pd.read_csv("DIAGNOSES_ICD.csv")
D_Diagnoses = pd.read_csv("D_ICD_DIAGNOSES.csv")

# ICD Codes Corresponding to CheXpert Classes

CheXpert_ICD = {'Enlarged Cardiomediastinum':['1642','1643','1648','1649','1971','2125','5193'],
                'Cardiomegaly':['4293'],
                'Lung Opacity':['79319'],
                'Lung Lesion':['51889','5172','5178','74860','74869','86120','86121','86122','86130','86131','86132','9471','79319'],
                'Edema':['5061','5184','7823'],
                'Consolidation':["5078",'51889','79319','486','481'],
                'Pneumonia':['01166','00322','01160','01161','01162','01163','01164','01165',
                             '0413','0551','0382','11505','11515','0730','48249','48281','48282',
                             '48283','4800','4801','4802','4803','4808','4809','481','4820','4821',
                             '4822','48230','48231','48232','48239','48240','48241','48242','48284','48289',
                             '4829','4830','4831','4838','4841','4843','4845',
                             '4846','4847','4848','485','486','4870','48801',
                             '48811','48881','51630','51630','51635','51636',
                             '51637','5171','7700','V066','99731','99732',
                             'V0382','V1261'],
                'Atelectasis':['7704','7705'],
                'Pneumothorax':['01170','01171','01172','01173',
                                '01174','01175','01176','5120',
                                '5121','51281','51282','51283',
                                '51289','8600','8601'],
                'Pleural Effusion':['51181','5119','5111'],
                'Pleural Other':["5110"],
                'Fracture':['80700','80710','8190','8191','V5427',
                            '80701','80702','80703','80704','80705',
                            '80706','80707','80708','80709','81103',
                            '81109','81110','80711','80712','80713',
                            '80714','80715','80716','80717','80718',
                            '80719','8072','8073','8074','81000',
                            '81001','81002','81003','81010','81011',
                            '81012','81013','81100','81101','81102',
                            '81103','73311','81111','81112','81113',
                            '81119','80500','80501','80502','80503',
                            '80504','80505','80506','80507','80508',
                            '80510','80511','80512','80513','80514',
                            '80515','80516','80517','80518','8052',
                            '8053','8054','8055','8058','8059','80600',
                            '80601','80602','80603','80604','80605',
                            '80606','80607','80608','80609','80610',
                            '80611','80612','80613','80614','80615',
                            '80616','80617','80618','80619','80620',
                            '80621','80622','80623','80624','806025',
                            '80626','80627','80628','80629','80630',
                            '80631','80632','80633','80634','80635',
                            '80636','80637','80638','80639','8064',
                            '8065','8068','8069'],
                'Support Devices':['V4321','V4500','V4509','99600','99609','9961','9962','99661','V5339','99672','99674']}


# List with CheXpert Classes

CheXpert_classes = list(CheXpert_ICD.keys())
CheXpert_classes = [x.lower().replace(" ", "_") for x in CheXpert_classes]

# Filter Diagnoses for CheXpert Classes only

Diagnoses_CheXpert = Diagnoses[Diagnoses["ICD9_CODE"].isin(list(chain.from_iterable(list(CheXpert_ICD.values()))))]

# Add Corresponding Notes

Diagnoses_CheXpert_Text = Diagnoses_CheXpert.merge(Notes[["SUBJECT_ID", "HADM_ID", "TEXT"]], on=["SUBJECT_ID","HADM_ID"], how="left")

Diagnoses_CheXpert_Text = Diagnoses_CheXpert_Text.drop_duplicates(["ICD9_CODE", "TEXT"])


##############################################################################
# CheXpert Rules

mentions = {}
for i in range(len(CheXpert_classes)):
    with open("".join(["mention/", CheXpert_classes[i], ".txt"])) as f:
        mentions[CheXpert_classes[i]] = [item.replace("\n", "") for item in f.readlines()]

##############################################################################
# Create T Matrix

T_df = pd.concat([pd.Series(v, name=k).astype(str) for k, v in mentions.items()], 
               axis=1)
T_df = pd.get_dummies(T_df.stack()).sum(level=1).T

CheXpert_rules = list(T_df.index)

T_matrix = np.matrix(T_df)
dump(T_matrix, "T_matrix.lib")

##############################################################################
# Create Z Matrix

Z_df = Diagnoses_CheXpert_Text.copy()

for i in tqdm(CheXpert_rules):
    Z_df[i] = Z_df.apply(lambda row: 1 if (isinstance(row['TEXT'], str) and i in row['TEXT'].lower()) else 0, axis = 1)


Z_df = Z_df[CheXpert_rules]

Z_matrix = np.matrix(Z_df)
dump(Z_matrix, "Z_matrix.lib")

Z_matrix_sparse = sparse.csr_matrix(Z_matrix)
sparse.save_npz("Z_matrix_sparse.npz", Z_matrix_sparse)