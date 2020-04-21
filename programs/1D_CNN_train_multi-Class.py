import csv
import glob
import os

# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from keras.layers import (Conv1D, Dense, Dropout, Embedding, Flatten,
                          MaxPooling1D,Activation,BatchNormalization)
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.regularizers import l2

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#正常数据标签0，滚动体故障标签1，内圈故障标签2，外圈故障标签3


with open ('/Users/alien/Documents/d盘/python/本科毕设/files/XiaoZhouCheng_Vertical_1500r.csv','r') as f:
    reader=csv.reader(f)
    Data=[]
    for row in reader:
        Data.append(row)
    Data=np.array(Data)

'''
初始化参数
'''
batch_size=20
epochs=30
num_class=4 
np.random.seed(6) #900-4，1200-3，1500-6

x_data=np.array([x[0:2000] for x in Data])
y_data=np.array([x[-1] for x in Data])

x_vals=x_data.astype(np.float64)
y_vals=y_data.astype(np.float64)

train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.5), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

y_vals_train=to_categorical(y_vals_train,num_class)
y_vals_test=to_categorical(y_vals_test,num_class)

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)
    
x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

'''
将输入数据reshape符合Conv1D的input_shape
'''
x_vals_train=x_vals_train.reshape((x_vals_train.shape[0],x_vals_train.shape[1],1))
x_vals_test=x_vals_test.reshape((x_vals_test.shape[0],x_vals_test.shape[1],1))


'''
建立训练模型
'''
model=Sequential()
#Convolution Layer (filter_shape=1*9,num_filter=60)
model.add(Conv1D(60,9,padding='same',input_shape=(x_vals_train.shape[1],1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

#Subsampling Layer (filter_shape=1*4)
model.add(MaxPooling1D(4))

#Convolution Layer (filter_shape=1*9,num_filter=40)
model.add(Conv1D(40,9,padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
#Subsampling Layer (filter_shape=1*4)
model.add(MaxPooling1D(4))

#Convolution Layer (filter_shape=1*9,num_filter=40)
model.add(Conv1D(40,9,padding='same'))
model.add(Activation('relu'))
#将最后一层卷积层的扁平化后再输入到全连接网络
model.add(Flatten())
#rate=0.5表示丢弃的比例，将50%的数据置为0，有助于防止过拟合
model.add(Dropout(0.5))

#Fully Connected MLP Layer (20 neurons)
model.add(Dense(20,activation='relu',kernel_regularizer=l2(0.01)))

#Output Layer (only 1 neuron for 2 classes)
model.add(Dense(4,activation='softmax',kernel_regularizer=l2(0.01)))

'''
编译训练模型
'''

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



History=model.fit(x_vals_train,y_vals_train,
            batch_size=batch_size,
            validation_data=(x_vals_test,y_vals_test),
            epochs=epochs,verbose=2)

score=model.evaluate(x_vals_test,y_vals_test,
            batch_size=batch_size)



#画结果图（train_loss、test_loss、train_acc、test_acc)
N=np.arange(1,epochs+1)
title='Training Loss and Accuracy on 小轴承实验台故障数据集'

plt.style.use('ggplot')
# plt.figure()
plt.plot(N,History.history['loss'],label='train_loss')
plt.plot(N,History.history['val_loss'],label='test_loss')
plt.plot(N,History.history['accuracy'],label='train_acc')
plt.plot(N,History.history['val_accuracy'],label='test_acc')
plt.title(title)
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
