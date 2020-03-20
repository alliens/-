import csv
import glob
#正常数据标签0，滚动体故障标签1，内圈故障标签2，外圈故障标签3
'''
def getFileList(dirPath):
    fileList=glob.glob(dirPath)
    fileList=sorted(fileList)
    return fileList

fileList=getFileList('/Users/alien/Documents/f盘/毕设数据/小轴承实验台轴承故障数据_副本/外圈1故障数据/'+'2012*')
for fileName in fileList:
    with open (fileName,'r') as f:
        data=f.readline()
        data=list(data.split(' '))
        data.pop()
        data=[float(x) for x in data]
        data=data[0:2000]
    with open ('2000.csv','a+') as csv_f:
        writer=csv.writer(csv_f)
        writer.writerow(data)
'''