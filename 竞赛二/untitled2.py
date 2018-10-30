# -*- coding: utf-8 -* -
"""
Created on Wed Oct 10 18:50:07 2018

@author: lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
import pandas as pd
import csv

with open('.\\train.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    data = open('.\\train-data.txt')
    for each_line in data:
        a = each_line.strip("\n").split(',')
        writer.writerow(a)

x_train=pd.DataFrame(pd.read_csv('.\\train.csv',header=None))
x_test=pd.DataFrame(pd.read_csv('.\\testx.csv'))

x=x_train.replace([' Private',' Self-emp-not-inc',' Self-emp-inc',' Federal-gov',' Local-gov',' State-gov',' Without-pay',' Never-worked',' ?',
                   ' Bachelors',' Some-college',' 11th',' HS-grad',' Prof-school',' Assoc-acdm',' Assoc-voc',' 9th',' 7th-8th',' 12th',' Masters',' 1st-4th',' 10th',' Doctorate',' 5th-6th',' Preschool',
                   ' Married-civ-spouse',' Divorced',' Nevermarried',' Separated',' Widowed',' Married-spouse-absent',' Married-AF-spouse',
                   ' Tech-support',' Craft-repair',' Other-service',' Sales',' Exec-managerial',' Prof-specialty',' Handlers-cleaners',' Machine-op-inspct',' Adm-clerical',' Farming-fishing',' Transport-moving',' Priv-house-serv',' Protective-serv',' Armed-Forces',
                   ' Wife',' Own-child',' Husband',' Not-in-family',' Other-relative',' Unmarried',
                   ' White',' Asian-Pac-Islander',' Amer-Indian-Eskimo',' Other',' Black',
                   ' Female',' Male',
                   ' United-States',' Cambodia',' England',' Puerto-Rico',' Canada',' Germany',' Outlying-US',' India',' Japan',' Greece',' South',' China',' Cuba',' Iran',' Honduras',' Philippines',' Italy',' Poland',' Jamaica',' Vietnam',' Mexico',' Portugal',' Ireland',' France',' Dominican-Republic', ' Laos',' Ecuador',' Taiwan',' Haiti',' Columbia',' Hungary',' Guatemala',
                   ' Nicaragua',' Scotland',' Thailand',' Yugoslavia',' El-Salvador',' Trinadad&Tobago',' Peru',' Hong',' Holand-Netherlands',
                   ' <=50k',' >50k'],
                    [0,1,2,3,4,5,6,7,'unknown',
                    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                    0,1,2,3,4,5,6,
                    0,1,2,3,4,5,6,7,8,9,10,11,12,13,
                    0,1,2,3,4,5,
                    0,1,2,3,4,
                    0,1,
                    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                    0,1])
print(x),
#print(x_train)