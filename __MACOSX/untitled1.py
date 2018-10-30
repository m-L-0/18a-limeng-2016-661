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

x=pd.DataFrame(pd.read_csv('.\\trainx.csv'))
x=x.fillna(method='ffill')
X = np.array(x[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
                'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
                'cons.conf.idx', 'euribor3m', 'nr.employed']])
sc = StandardScaler()
X_std = sc.fit_transform(X)
pca = PCA(n_components=10)
X_std_pca = pca.fit_transform(X_std)
print(X_std_pca.shape)

x1=pd.DataFrame(pd.read_csv('.\\testx.csv'))
x1=x1.fillna(method='ffill')
X1 = np.array(x1[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
                'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
                'cons.conf.idx', 'euribor3m', 'nr.employed']])
X_std1 = sc.fit_transform(X1)
pca = PCA(n_components=10)
X_std_pca1 = pca.fit_transform(X_std1)
print(X_std_pca1.shape)

df1 = pd.read_csv('.\\trainy.csv')
Y = np.ravel(df1)

clf = RandomForestClassifier()
clf = clf.fit(X_std_pca,Y)
print(clf.predict(X_std_pca1))
result=clf.predict(X_std_pca1)
np.savetxt(".\\result1.csv",result)

