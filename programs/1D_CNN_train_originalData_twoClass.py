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
with open ('/Users/alien/Documents/d盘/python/本科毕设/files/dataWithLabel_2000_DE.csv','r') as f:
    reader=csv.reader(f)
    Data=[]
    for row in reader:
        Data.append(row)
    Data=np.array(Data)


#训练集是12k驱动端原始数据集
x_data=np.array([x[0:2000] for x in Data])
y_data=np.array([x[-1] for x in Data])


x_vals=x_data.astype(np.float64)
y_vals=y_data.astype(np.float64)

'''
使结果可以重现
'''
np.random.seed(4)

'''
将数据集分为训练集/测试集=80%/20%
'''
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# def normalize_cols(m):
#     col_max = m.max(axis=0)
#     col_min = m.min(axis=0)
#     return (m-col_min) / (col_max - col_min)
    
# x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
# x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

'''
初始化参数
'''
batch_size=16
epochs=100 #epoch=30时train_acc和test_acc均能达到100%

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
model.add(Dropout(0.5))

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

# #保存模型
# model_json=model.to_json()
# with open ('baseModle_twoClass_new.json','w') as f:
#     f.write(model_json)

# #保存模型权重
# model.save_weights('baseModle_twoClass_new.h5')

# mod=model.get_weights()
# print(mod)