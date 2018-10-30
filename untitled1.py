# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 11:33:29 2018

@author: lenovo
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
#from sklearn.cross_validation import train_test_split
#from sklearn.tree import DecisionTreeClassifier

x=pd.DataFrame(pd.read_csv('.\\trainx.csv'))
x=x.fillna(method='ffill')
print(x)
#X = np.array(x[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
#                'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
#                'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
#                'cons.conf.idx', 'euribor3m', 'nr.employed']])
#sc = StandardScaler()
#X_std = sc.fit_transform(X)
#pca = PCA(n_components=13)
#X_std_pca = pca.fit_transform(X_std)
#print(X_std_pca.shape)
#print(pca.explained_variance_ratio_)
#
#x1=pd.DataFrame(pd.read_csv('.\\testx.csv'))
#x1=x1.fillna(method='ffill')
#X1 = np.array(x1[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
#                'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
#                'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
#                'cons.conf.idx', 'euribor3m', 'nr.employed']])
#X_std1 = sc.fit_transform(X1)
#pca = PCA(n_components=13)
#X_std_pca1 = pca.fit_transform(X_std1)
#print(X_std_pca1.shape)
#
#df1 = pd.read_csv('.\\trainy.csv')
#Y = np.ravel(df1)

#X_train, X_test, y_train, y_test = train_test_split(X_std_pca, Y, test_size=0.30, random_state=42)

#clf = RandomForestClassifier()
#clf = clf.fit(X_std_pca,Y)
#print(clf.predict(X_std_pca1))
#result=clf.predict(X_std_pca1)
#
##a=clf.fit(X_train, y_train)
##b=clf.score(X_test, y_test)
##print(b)
#
#data = pd.DataFrame(result)
#data.to_csv(".\\data1.csv",index=False,header=False)



