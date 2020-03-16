from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Embedding
from keras.layers import Conv1D,MaxPooling1D

import csv
import numpy as np
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
读取带有标签的训练数据
'''
with open ('/Users/alien/Documents/d盘/python/本科毕设/originalDataWithLabel_700.csv','r') as f:
    reader=csv.reader(f)
    Data=[]
    for row in reader:
        Data.append(row)
    Data=np.array(Data)

with open ('/Users/alien/Documents/d盘/python/本科毕设/originalDataWithLabel_700_FE.csv','r') as f:
    reader=csv.reader(f)
    Data_test=[]
    for row in reader:
        Data_test.append(row)
    Data_test=np.array(Data_test)

#训练集是12k驱动端原始数据集
x_data=np.array([x[0:700] for x in Data])
y_data=np.array([x[-1] for x in Data])

#测试集是12k风扇端原始数据集
x_data_test=np.array([x[0:700] for x in Data_test])
y_data_test=np.array([x[-1] for x in Data_test])

x_vals_train=x_data.astype(np.float64)
y_vals_train=y_data.astype(np.float64)

x_vals_test=x_data_test.astype(np.float64)
y_vals_test=y_data_test.astype(np.float64)

'''
使结果可以重现
'''
np.random.seed(3)

'''
将数据集分为训练集/测试集=80%/20%
'''
# train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
# test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
# x_vals_train = x_vals[train_indices]
# x_vals_test = x_vals[test_indices]
# y_vals_train = y_vals[train_indices]
# y_vals_test = y_vals[test_indices]

'''
初始化参数
'''
batch_size=10
epochs=20 #epoch=30时train_acc和test_acc均能达到100%

'''
将输入数据reshape符合Conv1D的input_shape
'''
x_vals_train=x_vals_train.reshape((x_vals_train.shape[0],x_vals_train.shape[1],1))
x_vals_test=x_vals_test.reshape((x_vals_test.shape[0],x_vals_test.shape[1],1))

y_vals_train=y_vals_train.reshape((y_vals_train.shape[0],1))
y_vals_test=y_vals_test.reshape((y_vals_test.shape[0],1))



'''
建立训练模型
'''
model=Sequential()
#Convolution Layer (filter_shape=1*9,num_filter=60)
model.add(Conv1D(60,9,padding='same',activation='relu',input_shape=(x_vals_train.shape[1],1)))
#Subsampling Layer (filter_shape=1*4)
model.add(MaxPooling1D(4))

#Convolution Layer (filter_shape=1*9,num_filter=40)
model.add(Conv1D(40,9,padding='same',activation='relu'))

#Subsampling Layer (filter_shape=1*4)
model.add(MaxPooling1D(4))

#Convolution Layer (filter_shape=1*9,num_filter=40)
model.add(Conv1D(40,9,padding='same',activation='relu'))

#将最后一层卷积层的扁平化后再输入到全连接网络
model.add(Flatten())
#rate=0.5表示丢弃的比例，将50%的数据置为0，有助于防止过拟合
model.add(Dropout(0.3))

#Fully Connected MLP Layer (20 neurons)
model.add(Dense(20,activation='relu'))

#Output Layer (only 1 neuron for 2 classes)
model.add(Dense(1,activation='sigmoid'))

'''
编译训练模型
'''
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



History=model.fit(x_vals_train,y_vals_train,
            batch_size=batch_size,
            validation_data=(x_vals_test, y_vals_test),
            epochs=epochs,verbose=2)

score=model.evaluate(x_vals_test,y_vals_test,
            batch_size=batch_size)
print('acc='+str(score[1]*100))


#画结果图（train_loss、test_loss、train_acc、test_acc)
N=np.arange(1,epochs+1)
title='Training Loss and Accuracy on CWRU dataset(12k-DE)'

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





