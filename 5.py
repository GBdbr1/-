import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
#1.读取数据
df1 = pd.read_csv('faceDR', header = None)
df2 = pd.read_csv('faceDS', header = None)
f1 = pd.DataFrame(np.zeros(len(df1)), columns = ['label'])
f2 = pd.DataFrame(np.zeros(len(df2)), columns = ['label'])
for i in range(0, len(df1)):
    f1['label'][i] = df1[0][int(f'{i}')].split()[2]
for i in range(0, len(df2)):
    f2['label'][i] = df2[0][int(f'{i}')].split()[2]
f1['label'] = f1['label'].map(lambda x : x.replace(')', ''))
f2['label'] = f2['label'].map(lambda x : x.replace(')', ''))
f1 = f1.replace(to_replace = 'descriptor', value = np.NaN)
f2 = f2.replace(to_replace = 'descriptor', value = np.NaN)
f1 = f1.dropna().reset_index(drop = True)
f2 = f2.dropna().reset_index(drop = True)
f = pd.concat([f1, f2], axis = 0).reset_index(drop = True)
X = []
folders = os.listdir('rawdata')
for i in folders:
    x = np.fromfile('rawdata/' + i)
    X.append(x.reshape(1, -1).flatten().tolist())
df = pd.DataFrame(X)

#2.截取非缺失值，填充少数缺失值为 0
df = df[df.columns[:2048]]
df = df.fillna(0)
df = pd.concat([df, f], axis = 1)

#3.预测值转换为数值
Dict = {'male' : 0, 'female' : 1}
df['label'] = df['label'].map(Dict)

#4.归一化
model_M = MinMaxScaler()
df_feature = df[df.columns[:2048]]
for i in df_feature.columns:
    df_feature[i] = model_M.fit_transform(pd.DataFrame(df_feature[i]))

#5.划分数据集、PCA 降维
X_train, X_test, y_train, y_test = train_test_split(df_feature, df['label'], test_size = 0.3)
model_PCA = PCA(n_components = 0.95)
model_PCA.fit(X_train)
X_train = model_PCA.transform(X_train)
X_test = model_PCA.transform(X_test)

#6.KNN
#6.1 网格调参
model_KNN = KNeighborsClassifier()
parameters = {
    'p':[2],
    'n_neighbors':[i for i in range(3, 11)],
}
model_GSCV = GridSearchCV(model_KNN, parameters, scoring = 'accuracy', n_jobs = -1, cv = 5)
model_GSCV.fit(X_train, y_train)
score_ACC = accuracy_score(y_test, model_GSCV.predict(X_test))
print('验证集准确率：', score_ACC)
print('最优参数：', model_GSCV.best_params_)
#6.2 模型预测
model_KNN = KNeighborsClassifier(
    p = 2,
    n_neighbors = 8
)
model_KNN.fit(X_train, y_train)
predict = model_KNN.predict(X_test)
print("最近邻识别准确率", accuracy_score(predict, y_test))

#7.SVM
#7.1 网格调参
model_SVC = SVC()
parameters = {
    'kernel':['linear'],
    'C':[i for i in range(1, 5)],
}
model_GSCV = GridSearchCV(model_SVC, parameters, scoring = 'accuracy', n_jobs = -1, cv = 5)
model_GSCV.fit(X_train, y_train)
score_ACC = accuracy_score(y_test, model_GSCV.predict(X_test))
print('验证集准确率：', score_ACC)
print('最优参数：', model_GSCV.best_params_)
#7.2 模型预测
model_SVC = SVC(
    kernel = 'linear',
    C = 1
)
model_SVC.fit(X_train, y_train)
predict = model_SVC.predict(X_test)
print("支持向量机识别准确率", accuracy_score(predict, y_test))
