import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.pardir)
from two_layer_net import TwoLayerNet

def file_input(path):
    data=np.zeros((75,4))
    spi_data=np.zeros((75,3))
    col=0
    with open(path) as f:
        for s_line in f:

            #改行のみの行にたどり着いたら終了
            if s_line=='\n':
                break
        
            #'\n' to ''  split
            s_line=s_line.replace('\n','')
            line_data=s_line.split(',')
        
            for i in range(0,4):
                data[col][i]=float(line_data[i])
        
            if line_data[4]=='Iris-setosa':
                spi_data[col][0]=1
            elif line_data[4]=='Iris-versicolor':
                spi_data[col][1]=1
            else:
                spi_data[col][2]=1
            col+=1
        f.close()
    return data,spi_data

#load data 
test_path="./../data_sets/test75_iris.dat"
train_path="./../data_sets/Traning75_iris.dat"
x_test,t_test=file_input(test_path)
x_train,t_train=file_input(train_path)


network=TwoLayerNet(input_size=4,hidden_size=8,output_size=3)

itres_num=1000
learning_rate = 0.1

train_loss_list=[]
test_loss_list=[]
train_acc_list=[]
test_acc_list=[]

batch_size=75
train_size = x_train.shape[0]

iter_per_epoch=max(train_size/batch_size,1)

for i in range(itres_num):
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]

    grad=network.gradient(x_batch,t_batch)
    #loss=network.loss(x_batch,t_batch)

    for key in ('W1','b1','W2','b2'):
        network.params[key]-=learning_rate*grad[key]
    
    if i%iter_per_epoch==0:
        train_loss=network.loss(x_train,t_train)
        test_loss=network.loss(x_test,t_test)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        train_acc=network.accuracy(x_train,t_train)
        test_acc=network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc,test_acc)


x=np.linspace(0,10,itres_num)
plt.plot(train_loss_list)
plt.plot(test_loss_list)
plt.show()