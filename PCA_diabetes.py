# Description：
# Author：朱勇
# Time：2021/3/7 16:02

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv('task2_data.csv')
x = data.drop(['label'],axis=1)
y = data.loc[:,'label']
print(x.shape,y.shape)
#没有PCA的逻辑回归模型
model1 = LogisticRegression(max_iter=1000)
model1.fit(x,y)
y_predict = model1.predict(x)
print(accuracy_score(y,y_predict))
#数据标准化
x_norm = StandardScaler().fit_transform(x)
#计算均值与标准差
x1_mean = x.loc[:,'glucose'].mean()
x1_norm_mean = x_norm[:,1].mean()
x1_sigma = x.loc[:,'glucose'].std()
x1_norm_sigma = x_norm[:,1].std()
print(x1_mean,x1_sigma,x1_norm_mean,x1_norm_sigma)
#可视化
fig1 = plt.figure(figsize=(12,5))
fig1_1 = plt.subplot(121)
plt.hist(x.loc[:,'glucose'],bins=100)
fig1_2 = plt.subplot(122)
plt.hist(x_norm[:,1],bins=100)
plt.show()
#PCA主成分分析
pca = PCA(n_components=8)
x_pca = pca.fit_transform(x_norm)
#计算分析后各成分的方差以及方差比例
var = pca.explained_variance_
var_ratio = pca.explained_variance_ratio_
print(var)
print(var_ratio)
print(sum(var_ratio))
#可视化方差比例
fig2 = plt.figure(figsize=(10,5))
plt.bar([1,2,3,4,5,6,7,8],var_ratio)
plt.show()
#数据降维到2维
pca_2 = PCA(n_components=2)
x_pca_2 = pca_2.fit_transform(x_norm)
var_2_ratio = pca_2.explained_variance_ratio_
print('2维的方差比例:',end='')
print(type(x_pca_2))
#降维数据可视化
fig3 = plt.figure()
plt.scatter(x_pca_2[:,0][y==0],x_pca_2[:,1][y==0],marker='x',label='negative')
plt.scatter(x_pca_2[:,0][y==1],x_pca_2[:,1][y==1],marker='*',label='positive')
plt.legend()
plt.show()
#降维后训练
model2 = LogisticRegression()
model2.fit(x_pca_2,y)
y_predict_pca = model2.predict(x_pca_2)
print(accuracy_score(y_predict_pca,y))



