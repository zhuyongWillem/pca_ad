# Description：
# Author：朱勇
# Time：2021/3/7 16:01

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.covariance import EllipticEnvelope

data = pd.read_csv('task1_data.csv')
fig1 = plt.figure()
plt.scatter(data.loc[:,'frequency'],data.loc[:,'payment'],marker='x')
plt.title('raw_data')
plt.xlabel('frequency')
plt.ylabel('payment')
plt.show()
x = data
x1 = x.loc[:,'frequency']
x2 = x.loc[:,'payment']
#数据分布可视化操作
fig2 = plt.figure(figsize=(20,10))
fig2_1 = plt.subplot(121)
plt.hist(x1,bins=100)
plt.title('frequency data')
plt.xlabel('frequency')
plt.ylabel('counts')
fig2_2 = plt.subplot(122)
plt.hist(x2,bins=100)
plt.title('payment data')
plt.xlabel('payment')
plt.ylabel('counts')
plt.show()
#计算平均值，以及标准差
x1_mean = x1.mean()
x1_sigma = x1.std()
x2_mean = x2.mean()
x2_sigma = x2.std()
#计算基于高斯分布的概率函数
x1_range = np.linspace(0,10,300)
x1_normal = norm.pdf(x1_range,x1_mean,x1_sigma)
x2_range = np.linspace(0,400,300)
x2_normal = norm.pdf(x2_range,x2_mean,x2_sigma)
#可视化概率密度函数
fig3 = plt.figure(figsize=(20,10))
fig3_1 = plt.subplot(121)
plt.plot(x1_range,x1_normal)
plt.title('x1(frequency) Gaussian Distribution')
plt.xlabel('x1(frequency)')
plt.ylabel('p(x1)')
fig3_2 = plt.subplot(122)
plt.plot(x2_range,x2_normal)
plt.title('x2(payment) Gaussian Distribution')
plt.xlabel('x2(payment)')
plt.ylabel('p(x2)')
plt.show()
#3D
#设置范围
x_min, x_max = 0, 10
y_min, y_max = 0, 400
h1 = 0.1
h2 = 0.1
#生成矩阵数据
xx, yy = np.meshgrid(np.arange(x_min, x_max, h1), np.arange(y_min, y_max, h2))
print(xx.shape,yy.shape)

#展开矩阵数据
x_range = np.c_[xx.ravel(), yy.ravel()]
x1 = np.c_[xx.ravel()]
x2 = np.c_[yy.ravel()]
x_range_df = pd.DataFrame(x_range)
#x_range_df.to_csv('data.csv')
#高斯分布参数
u1 = x1_mean
u2 = x2_mean
sigma1 = x1_sigma
sigma2 = x2_sigma

#计算高斯分布概率
p1 = 1/sigma1/math.sqrt(2*math.pi)*np.exp(-np.power((x1-u1),2)/2/math.pow(sigma1,2))
p2 = 1/sigma2/math.sqrt(2*math.pi)*np.exp(-np.power((x2-u2),2)/2/math.pow(sigma2,2))
p = np.multiply(p1,p2)
#对概率密度维度转化
p_2d = p.reshape(xx.shape[0],xx.shape[1])
#3D作图
fig5 = plt.figure()
axes3d = Axes3D(fig5)
axes3d.plot_surface(xx,yy,p_2d,cmap=cm.rainbow)
#建立异常检测模型
model = EllipticEnvelope(contamination=0.03)
model.fit(x)
y_predict = model.predict(x)
print(pd.value_counts(y_predict))
fig6 = plt.figure()
plt.scatter(data.loc[:,'frequency'],data.loc[:,'payment'],marker='x',label='raw_data')
plt.scatter(data.loc[:,'frequency'][y_predict==-1],data.loc[:,'payment'][y_predict==-1],marker='o',facecolor='none',edgecolors='red',label='anomaly_data')
plt.title('raw_data')
plt.xlabel('frequency')
plt.ylabel('payment')
plt.legend(loc='upper left')
plt.show()
