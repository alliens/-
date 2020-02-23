import datetime
import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio
import scipy.signal as ss
from scipy.fftpack import fft


def loadData_from_oneFile(filename):
    '''
    Parameters 
    ----------
    filename : the name of one file 
    ''' 
    data=scio.loadmat(filename)
    for key in data.keys():
        if key[-7:]=='DE_time':
            wave=data[key]
    return wave

def getFinal_data():
    fileList=glob.glob('/Users/alien/Documents/f盘/毕设数据/12k驱动端/'+'12k_Drive_End*')
    fileList=sorted(fileList)
    Data=[]
    for fileName in fileList:
        wave=loadData_from_oneFile(fileName)
        wave=wave.reshape(len(wave))
        Data.append(wave)
    return Data

def myspecfft(wave,fs,ifplot:bool=False):
    '''
    Parameters
    ----------
    wave : a one-dimension array of one file

    fs : sampling frequence

    ifplot : if plot the Spectrum
    '''
    L=len(wave)
    wave=wave-np.mean(wave)
    wave=wave.reshape((L))
    wave=wave*2.7875*ss.blackmanharris(L)
    Wave=fft(wave)/L
    #采样频率为12k，根据采样定理，频谱上只有0～6k是有效的，保险起见最好取12k/2.56
    f=(fs/L)*np.array(range(1,int(L/2.56)))
    Af=2*abs(Wave[1:int(L/2.56)])
    if ifplot== True:
        plt.figure()
        plt.title('Spectrum')
        plt.plot(f,Af)
    #返回值f为频率，Af为幅值
    return Af,f

def RMS_fSectionFromInputedSpec(VibSpec,df,Fs,StartF,EndF):
    '''
    Parameters
    ----------
    VibSpec : Input spectrum

    df : Interval between two frequencies on the X axis of the spectrum

    Fs : Sampling frequency

    StartF : Start index of frequency band

    EndF : End index of frequency band 
    '''
    startFindex=int(round(StartF/df))
    endFindex=int(round(EndF/df))
    squareVib=list(map(lambda num: num*num,VibSpec[startFindex:endFindex]))
    RMS_fSection=math.sqrt(sum(squareVib)/(endFindex-startFindex+1))
    return RMS_fSection

def main():
    fs=12000
    #负载分别为0,1,2,3的转速
    # RPM=[1797,1772,1750,1730]
    #四个转速/60得到的倍频均接近30
    #frequencyMul=30
    Data=getFinal_data()
    print('len of Data:'+str(len(Data)))
    Feature=[]
    for data in Data:
        # start=datetime.datetime.now()
        feature=[]
        Af,f=myspecfft(data,fs)
        #频谱横坐标两个频率的间隔
        df=f[1]-f[0]
        startFre=0
        endFre=10

        #0X-5X分为15段，每段取RMS
        for i in range(15):
            dx=10
            rms=RMS_fSectionFromInputedSpec(Af,df,fs,startFre,endFre)
            feature.append(rms)
            startFre+=dx
            endFre+=dx
        
        #5X-10X分为10段，每段取RMS
        for i in range(10):
            startFre=150
            endFre=165
            dx=15
            rms=RMS_fSectionFromInputedSpec(Af,df,fs,startFre,endFre)
            feature.append(rms)
            startFre+=dx
            endFre+=dx

        #10X-20X分为10段，每段取RMS
        for i in range(10):
            startFre=300
            endFre=330
            dx=30
            rms=RMS_fSectionFromInputedSpec(Af,df,fs,startFre,endFre)
            feature.append(rms)
            startFre+=dx
            endFre+=dx

        #20X-100X分为8段，分段取最大值，若超过采样频率限制，则取0
        for i in range(8):
            startFre=600
            endFre=900
            dx=300
            Max=max(Af[startFre:endFre+1])
            feature.append(Max)
            startFre+=dx
            endFre+=dx
        Feature.append(feature)
        # end=datetime.datetime.now()
        # print("处理每个文件的time:"+str(end-start))
    for fe in Feature:
        print(fe)

if __name__ == "__main__":
    main()
