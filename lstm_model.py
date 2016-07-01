#coding:utf8
from gym import spaces
import sklearn
import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import sys 
data_part1 = pd.read_csv('data/krank.part1.2015')
data_part2 = pd.read_csv('data/krank.part2.2015')
#part2 预处理第一步：NA赋值0(不是简单的补0值，是有一定业务含义的)
data_part2_pre1 = data_part2.fillna(0)
#part2 预处理第二步，对两部分数据进行了合并
data_pre2_merge = pd.merge(data_part1,data_part2_pre1)
#preprocess step 3: drop NAN
data_pre3_dorpna = data_pre2_merge.dropna()
#symbol_bool = data_pre3_dorpna['symbol'].value_counts() == 24
#如果数据是两年上面，如果用的一年下面
symbol_bool = data_pre3_dorpna['symbol'].value_counts() == 12 
symbol_index = symbol_bool[symbol_bool.values].axes[0]
print symbol_index
t = data_pre3_dorpna.isin({'symbol':list(symbol_index)})
input_data = data_pre3_dorpna[t['symbol']]
#数据scaling阶段
#（1）mktcap数量级过于大，需要降低两个数量级/100
#（2）准确率降低两个数量级/100
input_data['mktcap'] = input_data['mktcap']/100
input_data['6mr'] = input_data['6mr']/100
input_data = input_data.sort_values(['symbol','date'])
input_data.to_csv("input_data",sep = "\t")
model_train_x = []
model_train_y = []
group_data = input_data.groupby('symbol')
for key_value,group_by_key in group_data:
    print key_value
    #dframe_x = group_by_key[['krank','quality','value','momentum','growth','mktcap','sp500','sp1500','rsl1000']]
    dframe_x = group_by_key[['krank','quality','value','growth','mktcap']]
    model_train_x.append(dframe_x.values)
    #dframe_y = group_by_key[['3mrs','6mrs','3mr','6mr']]
    dframe_y = group_by_key['6mr']
    model_train_y.append(dframe_y.values)
step_size = 3 
time_size = len(model_train_x[0])
timesteps = time_size - step_size
num_samples = len(model_train_x)
data_dim = len(model_train_x[0][0])
train_x = []
train_y = []
test_x = []
test_y = []
for i in range(num_samples):
    for j in range(timesteps):
        sigle_x = []
        for k in range(j,j+step_size):
            sigle_x.append(model_train_x[i][k])
        train_x.append(sigle_x)
        train_y.append([round(model_train_y[i][j+step_size-1]/1.0,3)])
    sigle_x = []
    for j in range(timesteps,timesteps+step_size):
        sigle_x.append(model_train_x[i][j])
    test_x.append(sigle_x)
    test_y.append(round(model_train_y[i][timesteps+step_size-1]/1.0,3))
print("train sample nums:%d,label nums:%d,time steps:%d,data dim:%d"%(len(train_x),len(train_y),len(train_x[0]),len(train_x[0][0])))
print("test sample nums:%d,label nums:%d,time steps:%d,data dim:%d"%(len(test_x),len(test_y),len(test_x[0]),len(test_x[0][0])))
timesteps = len(train_x[0])
data_dim - len(train_x[0][0])
model = Sequential()
model.add(LSTM(32, return_sequences=True,
    input_shape=(timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
"""
model.add(Dense(1, activation='softmax'))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')
model.fit(train_x,train_y,nb_epoch = 100,batch_size = 40,verbose = 1,shuffle=True)
"""
model.add(Dense(32,init="glorot_uniform"))
model.add(Dense(1,input_dim=32,init="glorot_uniform"))
model.add(Activation("linear"))
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.11, nesterov=True)
model.compile(loss='mean_absolute_error', optimizer=sgd)
model.fit(train_x,train_y,nb_epoch = 10,batch_size = 4,verbose = 1,shuffle=True)
#model.fit(test_x,test_y,nb_epoch = 2000,batch_size = 4,verbose = 1,shuffle=True)
pre_label =  model.predict_proba(test_x)
right_num = 0
for i in range(len(test_x)):
    if test_y[i] >= 0:
        low_thr = test_y[i]*0.8
        high_thr = test_y[i]*1.2
    else:
        low_thr = test_y[i]*1.2
        high_thr = test_y[i]*0.8
    if (pre_label[i]>low_thr) and (pre_label[i]<high_thr):
        right_num += 1
    print test_y[i],pre_label[i],((pre_label[i]>low_thr) and (pre_label[i]<high_thr))
print("%.2f"%(1.0*right_num/len(test_x)))
