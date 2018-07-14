#import matplotlib.pyplot as plt
import numpy as np

print("#---------------------------------------#")
print("           Feature Engineering           ")
print("#---------------------------------------#")
print("\n")

data = [
{'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
{'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
{'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
{'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]

#"# sklearn 中提供了将字典向量化的操作"

from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer(sparse=False,dtype=int)

# fit_transform()的含义是既进行了拟合也进行了转换操作

print(vec.fit_transform(data))
print(" ")

# 输出特征的名称

print(vec.get_feature_names())
print(" ")

# 将 sparse 稀疏设置成 True

vec=DictVectorizer(sparse=True,dtype=int)
print(vec.fit_transform(data))
print(" ")

sample = ['problem of evil',
'evil queen',
'horizon problem']

# 下面是抽提文本内容中的特征，主要是使用到了 word count 这个特征进行处理

from sklearn.feature_extraction.text import CountVectorizer

# 获得模型
vec=CountVectorizer()

# 利用模型对数据进行拟合和转换
X=vec.fit_transform(sample)
print(X)
print(" ")

# 将 sparse matrix 转换成 pandas中的dataframe

import pandas as pd
print(pd.DataFrame(X.toarray(),columns=vec.get_feature_names()))
print(" ")

from sklearn.feature_extraction.text import TfidfVectorizer
vec=TfidfVectorizer()
X=vec.fit_transform(sample)
print(pd.DataFrame(X.toarray(),columns=vec.get_feature_names()))
print(" ")

# 衍生特征
import matplotlib.pyplot as plt

x=np.array([1,2,3,4,5])
y=np.array([4,2,1,3,7])
plt.scatter(x,y)
plt.show()

from sklearn.linear_model import LinearRegression
X=x[:,np.newaxis]
model=LinearRegression().fit(X,y)
yfit=model.predict(X)
plt.scatter(x,y)
plt.plot(x,yfit)
plt.show()

# 增加额外的特征列

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=3,include_bias=False)
X2=poly.fit_transform(X)
print(X2)

model=LinearRegression().fit(X2,y)
yfit=model.predict(X2)
plt.scatter(x,y)
plt.plot(x,yfit)
