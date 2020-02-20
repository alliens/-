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
    '''
    Parameters 
    filename : the name of one file 
    ''' 
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

def myspecfft(wave,fs):
    '''
    Parameters
    ----------
    wave : a one-dimension array of one file

    fs : sampling frequence
    '''
    L=len(wave)
    print(L)
    wave=wave-np.mean(wave)
    blmanharris=2.7875*np.blackman(L)
    wave=wave*blmanharris
    Wave=fft(wave)/L
    #采样频率为12k，根据采样定理，频谱上只有0～6k是有效的，保险起见最好取12k/2.56
    f=(fs/L)*np.array(range(1,int(L/2.56),1))
    Af=2*abs(Wave[0:int(L/2.56)])
    #返回值f为频率，Af为幅值
    return f,Af

def main():
    Data=getFinal_data()
    print(Data)

if __name__ == "__main__":
    main()
