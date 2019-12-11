import numpy as np
import sys,os

'''
path:ファイルのパス
num:ファイルの読み取りたい行数
pattern:1 10段階で読み込む
        2 1~3を0,4~6を1,7~10を2とする
'''
def file_import(path,num,pattern):
    data=np.zeros((num,11))
    cnt=np.zeros(10)
    col=0
    if pattern==0:
        spi_data=np.zeros((num,10))
    else:
        spi_data=np.zeros((num,3))
    
    with open(path) as f:
        for s_line in f:
 
            #改行のみの行にたどり着いたら終了
            """
            
            if s_line=='\n':
                break
            """
            #'\n' to ''  split
            s_line=s_line.replace('\n','')
            line_data=s_line.split(',')

            
            for i in range(0,11):
                data[col][i]=float(line_data[i])

            cnt[int(line_data[11])]+=1
            if pattern==0:
                spi_data[col][int(line_data[11])]=1
            else:
                quality=int(line_data[11])
                if quality<4:
                    spi_data[col][0]=1
                elif quality<7:
                    spi_data[col][1]=1
                elif quality<11:
                    spi_data[col][2]=1
                
            col+=1
        f.close()
        #評価の分布を表示する
        #print(cnt)
    return data,spi_data