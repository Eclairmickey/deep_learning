import numpy as np
import sys,os


sys.path.append(os.pardir)
path="./test.txt" #filepath
#横の数
num=7000
#学習回数
cnt=10
ignore_string=['\n',' ','[',']']
data=np.zeros((cnt,num))
with open(path) as f:
    col=0
    for s_line in f:
        if s_line=='\n':
            break
        
        #'\n'and' ' to ''  
        # split

        for c in ignore_string:
            s_line=s_line.replace(c,'')
        print(s_line)
        line_data=s_line.split(',')

        for i in range(0,num):
            data[col][i]=float(line_data[i])
    col+=1
f.close()

result=np.mean(data,axis=0)
print(result)
"""
graph write
"""



        


