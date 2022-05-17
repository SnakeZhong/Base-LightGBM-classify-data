import math

import  numpy as np
import lightgbm
import pandas as pd
import sklearn.model_selection
from pandas import plotting
from pandas import DataFrame as df
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#model(useless)
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
#model(useful)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
import seaborn as sns
sns.set_style("whitegrid")
plt.style.use('seaborn')
import time

data0=pd.read_excel("data/ALL_y.xlsx") #原始数据集
data1=pd.read_excel("data/ALL_1.xlsx") #原始数据集的转置
data2=pd.read_excel("data/gear_ALL.xlsx")#通过分解原数据波形分解成36个因素



#拆分数据集
random_state=42
X=data0.iloc[:,1:5]
Y=data0.iloc[:,5]
#数据编码
encode_index=LabelEncoder()
y=encode_index.fit_transform(Y)
x_train,x_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=random_state,shuffle=True
)

X_36=data2.iloc[:,0:36]
Y_36=data2.iloc[:,36]
encode_index=LabelEncoder()
y_36=encode_index.fit_transform(Y_36)
x36_train,x36_test,y36_train,y36_test=train_test_split(
    X_36,y_36,test_size=0.2,random_state=random_state,shuffle=True
)

#选取分解的最后一个波形作为训练数据
x_9_train=x36_train.iloc[:,[8,17,26,35]]
x_9_test=x36_test.iloc[:,[8,17,26,35]]
x_8_train=x36_train.iloc[:,[7,16,25,34]]
x_8_test=x36_test.iloc[:,[7,16,25,34]]
x_7_train=x36_train.iloc[:,[6,15,24,33]]
x_7_test=x36_test.iloc[:,[6,15,24,33]]
# y_9_train=y36_train
# y_9_test=y36_test




def lgbmodel(x_train,x_test,y_train,y_test):
    lgb_model=lightgbm.LGBMClassifier\
    (
        n_estimators=1000,learning_rate=0.05,max_depth=10,num_leaves=480,min_child_samples=10
    )
    lgb_model.fit(x_train,y_train)
    decisionTree_model_pre=lgb_model.predict(x_test)
    print('The accuracy of is {0}'.format(metrics.accuracy_score(decisionTree_model_pre,y_test)))

import math
x_origin_conversion_cluster=pd.concat([data0,data2],axis=1,join='outer')
x_origin_conversion_cluster.head()
X_40=x_origin_conversion_cluster.iloc[:,[3,13,22,31,40]]
# x_cluster_8=x_origin_conversion_cluster.to_excel("data/x_cluster_5.xlsx")
def data_Resaling(data):
    for i in range(8):
        max_diff=max(data.iloc[:,i])-min(data.iloc[:,i])
        data.iloc[:,i]=(data.iloc[:,i]-max(data.iloc[:,i]))/max_diff
    return data
def data_weighting(data):
    data=data_Resaling(data)
    weight_list_4=[0.027,0.027,0.027,0.915]
    weight_list_9_init=[0.079,0.078,0.09,0.101]
    weight_list_9=[]
    datax=[]
    for i in weight_list_9_init:
        weight_list_9.append(i/sum(weight_list_9_init))
    print(weight_list_9)
    weight_list_4.extend(weight_list_9)
    for i in range(8):
        data.iloc[:,i]=(data.iloc[:,i]*weight_list_4[i])
    # for i in range(147000):
    #     datax.append(sum(data.iloc[i,0:7]))
    # df=pd.DataFrame(datax,columns=['Stand_weight_x'],dtype=float)
    # print(df.head())
    return data

Y_40=x_origin_conversion_cluster.iloc[:,41]
encode_index=LabelEncoder()
y_40=encode_index.fit_transform(Y_40)
x40_train,x40_test,y40_train,y40_test=train_test_split(
    X_40,y_40,test_size=0.95,random_state=random_state,shuffle=True
)


from sklearn.metrics import mean_squared_error
lgb_train=lightgbm.Dataset(x40_train,y40_train)
lgb_test=lightgbm.Dataset(x40_test,y40_test)

params=\
{
    'task': 'train',
    'boosting_type': 'gbdt',    #基学习器
    'metric': {'l2', 'auc'},    #采用l2正则化
    'num_leaves': 256,  #决策树叶子数
    'learning_rate': 0.15,  #学习率
    # 'feature_fraction': 0.9,    #每次迭代中随机选择特征比例
    # 'bagging_fraction': 0.8,    #每次迭代选择部分样本来训练
    # 'bagging_freq': 5,      #执行5次bagging
    'verbose': 0,
    'max_depth':15

}
evals_result={}
lgb_model_x_1 = lightgbm.train(params,
                lgb_train,
                num_boost_round=10,
                init_model='./model.txt',
                learning_rates=lambda iter: 0.15 * (0.99 ** iter),
                valid_sets=lgb_test,
                evals_result=evals_result
                               )
print('特征名称:')
print(lgb_model_x_1.feature_name())
print('特征重要度:')
print(list(lgb_model_x_1.feature_importance()))
print('在训练过程中绘图...')

ax = lightgbm.plot_metric(evals_result, metric='l2')
plt.show()

print('画出特征重要度...')
ax = lightgbm.plot_importance(lgb_model_x_1, max_num_features=10)
plt.show()
y_pred = lgb_model_x_1.predict(x40_test, num_iteration=lgb_model_x_1.best_iteration)
print(mean_squared_error(y40_test, y_pred) ** 0.5)

