   # -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 18:50:07 2018

@author: lenovo
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import csv

with open('.\\train.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    data = open('.\\train-data.txt') 
    for each_line in data:
        a = each_line.strip("\n").split(',')
        writer.writerow(a)

x_train=pd.DataFrame(pd.read_csv('.\\train.csv',header=None))
x_test=pd.DataFrame(pd.read_csv('.\\testx.csv'))
result=pd.DataFrame(pd.read_csv('.\\sample.csv'))
result=np.array(result)
result=result[:,0]
result=result.astype(int)
#print(result)
x=x_train.replace([' Private',' Self-emp-not-inc',' Self-emp-inc',' Federal-gov',' Local-gov',' State-gov',' Without-pay',' Never-worked',' ?',
                   ' Bachelors',' Some-college',' 11th',' HS-grad',' Prof-school',' Assoc-acdm',' Assoc-voc',' 9th',' 7th-8th',' 12th',' Masters',' 1st-4th',' 10th',' Doctorate',' 5th-6th',' Preschool',
                   ' Married-civ-spouse',' Divorced',' Never-married',' Separated',' Widowed',' Married-spouse-absent',' Married-AF-spouse',
                   ' Tech-support',' Craft-repair',' Other-service',' Sales',' Exec-managerial',' Prof-specialty',' Handlers-cleaners',' Machine-op-inspct',' Adm-clerical',' Farming-fishing',' Transport-moving',' Priv-house-serv',' Protective-serv',' Armed-Forces',
                   ' Wife',' Own-child',' Husband',' Not-in-family',' Other-relative',' Unmarried',
                   ' White',' Asian-Pac-Islander',' Amer-Indian-Eskimo',' Other',' Black',
                   ' Female',' Male',
                   ' United-States',' Cambodia',' England',' Puerto-Rico',' Canada',' Germany',' Outlying-US(Guam-USVI-etc)',' India',' Japan',' Greece',' South',' China',' Cuba',' Iran',' Honduras',' Philippines',' Italy',' Poland',' Jamaica',' Vietnam',' Mexico',' Portugal',' Ireland',' France',' Dominican-Republic', ' Laos',' Ecuador',' Taiwan',' Haiti',' Columbia',' Hungary',' Guatemala',
                   ' Nicaragua',' Scotland',' Thailand',' Yugoslavia',' El-Salvador',' Trinadad&Tobago',' Peru',' Hong',' Holand-Netherlands',
                   ' <=50K',' >50K'],
                   [0,1,2,3,4,5,6,7,'NaN',
                    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                    0,1,2,3,4,5,6,
                    0,1,2,3,4,5,6,7,8,9,10,11,12,13,
                    0,1,2,3,4,5,
                    0,1,2,3,4,
                    0,1,
                    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                    0,1])
x_test=x_test.replace([' Private',' Self-emp-not-inc',' Self-emp-inc',' Federal-gov',' Local-gov',' State-gov',' Without-pay',' Never-worked',' ?',
                   ' Bachelors',' Some-college',' 11th',' HS-grad',' Prof-school',' Assoc-acdm',' Assoc-voc',' 9th',' 7th-8th',' 12th',' Masters',' 1st-4th',' 10th',' Doctorate',' 5th-6th',' Preschool',
                   ' Married-civ-spouse',' Divorced',' Never-married',' Separated',' Widowed',' Married-spouse-absent',' Married-AF-spouse',
                   ' Tech-support',' Craft-repair',' Other-service',' Sales',' Exec-managerial',' Prof-specialty',' Handlers-cleaners',' Machine-op-inspct',' Adm-clerical',' Farming-fishing',' Transport-moving',' Priv-house-serv',' Protective-serv',' Armed-Forces',
                   ' Wife',' Own-child',' Husband',' Not-in-family',' Other-relative',' Unmarried',
                   ' White',' Asian-Pac-Islander',' Amer-Indian-Eskimo',' Other',' Black',
                   ' Female',' Male',
                   ' United-States',' Cambodia',' England',' Puerto-Rico',' Canada',' Germany',' Outlying-US(Guam-USVI-etc)',' India',' Japan',' Greece',' South',' China',' Cuba',' Iran',' Honduras',' Philippines',' Italy',' Poland',' Jamaica',' Vietnam',' Mexico',' Portugal',' Ireland',' France',' Dominican-Republic', ' Laos',' Ecuador',' Taiwan',' Haiti',' Columbia',' Hungary',' Guatemala',
                   ' Nicaragua',' Scotland',' Thailand',' Yugoslavia',' El-Salvador',' Trinadad&Tobago',' Peru',' Hong',' Holand-Netherlands',
                   ' <=50K',' >50K'],
                   [0,1,2,3,4,5,6,7,'NaN',
                    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                    0,1,2,3,4,5,6,
                    0,1,2,3,4,5,6,7,8,9,10,11,12,13,
                    0,1,2,3,4,5,
                    0,1,2,3,4,
                    0,1,
                    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                    0,1])
x=x.astype(float) #训练集转化成浮点型
x_test=x_test.astype(float) #测试集转化成浮点型
x=x.fillna(x.mean()) #训练集缺失值用样本的均值代替
x_train=np.array(x) #将训练集转化成数组
x_test=x_test.fillna(x_test.mean()) #测试集缺失值处理
x_test=np.array(x_test)
#print(x_test)
x_train= np.delete(x_train,32561, axis = 0)
y=x_train[:,14]
x_train= np.delete(x_train,14, axis = 1)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
Y=np.ravel(y)
#X_train, X_test, y_train, y_test = train_test_split(x_train, Y, test_size=0.3, random_state=1)
model = LogisticRegression()
model.fit(x_train, Y)
#print(model)       #输出模型
# make predictions
#expected = y_test                       #测试样本的期望输出
predicted = model.predict(x_test)
predicted=predicted.astype(int)      #测试样本预测
#print(predicted)
sample=np.c_[result.T,predicted.T]
data = pd.DataFrame(sample)
data.to_csv(".sample1.csv",index=False,header=False)
#输出结果
#print(metrics.classification_report(expected, predicted))       #输出结果，精确度、召回率、f-1分数
#print(metrics.confusion_matrix(expected, predicted))     
#print(x_train)
