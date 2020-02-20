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

def loadData_from_oneFile(filename):
    data=scio.loadmat(filename)
    for key in data.keys():
        if key[-7:]=='DE_time':
            wave=data[key]
            break
    return wave

def getFinal_data():
    fileList=glob.glob('/Users/alien/Documents/f盘/毕设数据/12k驱动端/'+'12k_Drive_End*')
    fileList=sorted(fileList)
    Data=[]
    for fileName in fileList:
        wave=loadData_from_oneFile(fileName)
        Data.append(wave)
    return Data

def main():
    Data=getFinal_data()
    print(Data)

if __name__ == "__main__":
    main()
