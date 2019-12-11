import sys,os

def file_output(path,list):
    with open(path,mode='a') as f:
        f.write(str(list)+'\n')
    f.close()