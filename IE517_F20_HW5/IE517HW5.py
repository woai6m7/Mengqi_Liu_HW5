#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 17:50:37 2020

@author: liumengqi
"""

import pandas as pd
import seaborn as sns
from scipy import stats
import pylab
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
import numpy as np
from mlxtend.plotting import heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import time
from sklearn.svm import SVC
from sklearn.svm import SVR

df=pd.read_csv('/Users/liumengqi/Desktop/hw5_treasury_yield_curve_data.csv')
print(df.head())

df.dropna(inplace=True)
df.describe()

nrow = df.shape[0]
ncol = df.shape[1]
print("Number of Rows of Data = ", nrow) 
print("Number of Columns of Data = ", ncol)


cols = ['SVENF01','SVENF02','SVENF09','SVENF10','SVENF19','SVENF20','SVENF29','SVENF30','Adj_Close']
sns.pairplot(df[cols], height=2.5)
plt.show()

cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)
plt.show()

stats.probplot(df['Adj_Close'],dist="norm",plot=pylab)
plt.show()

# Use 85% of the data for the training set
X, y = df[df.columns[1:-1]].values, df['Adj_Close'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,
                     random_state=42)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#Part 2: Perform a PCA on the Treasury Yield dataset
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print('Explained_variance_ratio for all components: ', pca.explained_variance_ratio_)
print('Explained_variance for all components: ', pca.explained_variance_)

cov_mat = np.cov(X_test_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals,reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(0,30),var_exp,alpha=0.5,align='center',label='individual explained variance')
plt.step(range(0,30),cum_var_exp,where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()

pca3= PCA(n_components=3)
X_train_pca3 = pca3.fit_transform(X_train_std)
X_test_pca3 = pca3.transform(X_test_std)
print('Explained_variance_ratio of the 3 component version: ', pca3.explained_variance_ratio_)
print('Explained_variance of the 3 component version: ', pca3.explained_variance_)

cov_mat = np.cov(X_test_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals,reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(0,30),var_exp,alpha=0.5,align='center',label='individual explained variance')
plt.step(range(0,30),cum_var_exp,where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()

#Part 3: Linear regression v. SVM regressor - baseline

#Linear regression without PCA
start_reg= time.process_time()
reg = LinearRegression()
reg.fit(X_train, y_train)
end_reg= time.process_time()

#R2 score 
R2_train = reg.score(X_train, y_train)
R2_test = reg.score(X_test, y_test)
print("R^2 training: {}".format(R2_train))
print("R^2 test: {}".format(R2_test))

#rmse
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("RMSE training: {}".format(rmse_train))
print("RMSE test: {}".format(rmse_test))
print("Training time: {}s".format(end_reg-start_reg))

#Linear regression with PCA
start_reg_pca= time.process_time()
reg_pca = LinearRegression()
reg_pca.fit(X_train_pca3, y_train)

end_reg_pca= time.process_time()

#R2 score
R2_train_pca = reg_pca.score(X_train_pca3, y_train)
R2_test_pca = reg_pca.score(X_test_pca3, y_test)
print("R^2 training: {}".format(R2_train_pca))
print("R^2 test: {}".format(R2_test_pca))

#rmse
y_train_pred_pca = reg_pca.predict(X_train_pca3)
y_test_pred_pca = reg_pca.predict(X_test_pca3)
rmse_train_pca = np.sqrt(mean_squared_error(y_train, y_train_pred_pca))
rmse_test_pca = np.sqrt(mean_squared_error(y_test, y_test_pred_pca))
print("RMSE training: {}".format(rmse_train_pca))
print("RMSE test: {}".format(rmse_test_pca))
print("Training time: {}s".format(end_reg_pca-start_reg_pca))

#SVM regressor without PCA

start_svm= time.process_time()
svm = SVR(kernel="rbf")
svm.fit(X_train, y_train)

end_svm= time.process_time() 

#R2 score
R2_train_svm = svm.score(X_train, y_train)
R2_test_svm = svm.score(X_test, y_test)
print("R^2 training: {}".format(R2_train_svm))
print("R^2 test: {}".format(R2_test_svm))

#rmse
y_train_pred_svm = svm.predict(X_train)
y_test_pred_svm = svm.predict(X_test)
rmse_train_svm = np.sqrt(mean_squared_error(y_train, y_train_pred_svm))
rmse_test_svm = np.sqrt(mean_squared_error(y_test, y_test_pred_svm))
print("RMSE training: {}".format(rmse_train_svm))
print("RMSE test: {}".format(rmse_test_svm))
print("Training time: {}s".format(end_svm-start_svm))


#SVM regressor with PCA
start_svm_pca= time.process_time() 
svm_pca = SVR(kernel="rbf", gamma='auto')
svm_pca.fit(X_train_pca3, y_train)

end_svm_pca= time.process_time() 

#R2 score
R2_train_svm_pca = svm_pca.score(X_train_pca3, y_train)
R2_test_svm_pca = svm_pca.score(X_test_pca3, y_test)
print("R^2 training: {}".format(R2_train_svm_pca))
print("R^2 test: {}".format(R2_test_svm_pca))

#rmse
y_train_svm_pred_pca = svm_pca.predict(X_train_pca3)
y_test_svm_pred_pca = svm_pca.predict(X_test_pca3)
rmse_train_svm_pca = np.sqrt(mean_squared_error(y_train, y_train_svm_pred_pca))
rmse_test_svm_pca = np.sqrt(mean_squared_error(y_test, y_test_svm_pred_pca))
print("RMSE training: {}".format(rmse_train_svm_pca))
print("RMSE test: {}".format(rmse_test_svm_pca))
print("Training time: {}s".format(end_svm_pca-start_svm_pca))

print("My name is {Mengqi Liu}")
print("My NetID is: {mengqi3}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")