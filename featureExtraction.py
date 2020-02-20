import datetime
import glob
import math
import os

import h5py as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import fft
import scipy.io as scio

# def loadData_from_oneFile(filename):
#     data=scio.loadmat('/Users/alien/Documents/f盘/毕设数据/12k驱动端/12k_Drive_End_B007_0_118.mat')
    #   DE_time=data['X11']
#     return wave

# def getFinal_data():
#     file_list=glob.glob('/Users/alien/Documents/f盘/毕设数据/12k驱动端/'+'12k_Drive_End*')
#     file_list=sorted(file_list)
#     Data=[]
#     for fileName in file_list:
#         wave=loadData_from_oneFile(fileName)
#         Data.append(wave[0])
#     return Data

# data=hp.File('/Users/alien/Documents/f盘/毕设数据/12k驱动端/12k_Drive_End_B007_0_118.mat','r')
# print(data)
# data.close()

data=scio.loadmat('/Users/alien/Documents/f盘/毕设数据/12k驱动端/12k_Drive_End_B007_1_119.mat')
print(data.keys())

