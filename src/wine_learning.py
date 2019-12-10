import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.pardir)
#from two_layer_net import TwoLayerNet('W1','b1','W2','b2')
from many_layer_net import ManyLayerNet

def file_input(path,num):
    data=np.zeros((num,11))
    spi_data=np.zeros((num,10))
    cnt=np.zeros(10)
    col=0

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
            spi_data[col][int(line_data[11])]=1
            col+=1
        f.close()
        print(cnt)
    return data,spi_data

#load data 
test_path="./../data_sets/wine_test.csv"
train_path="./../data_sets/wine.csv"
x_test,t_test=file_input(test_path,500)
x_train,t_train=file_input(train_path,500)
#print(x_train)

network=ManyLayerNet(input_size=11,hidden_size=12,hidden_size2=10,output_size=10)
#network=TwoLayerNet(input_size=11,hidden_size=12,output_size=10)
itres_num=10000
learning_rate = 0.1

train_loss_list=[]
test_loss_list=[]
train_acc_list=[]
test_acc_list=[]

batch_size=25
train_size = x_train.shape[0]

iter_per_epoch=max(train_size/batch_size,1)

for i in range(itres_num):
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]

    grad=network.gradient(x_batch,t_batch)
    #loss=network.loss(x_batch,t_batch)

    for key in grad:
        network.params[key]-=learning_rate*grad[key]

    train_loss=network.loss(x_train,t_train)
    test_loss=network.loss(x_test,t_test)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)

    if i%iter_per_epoch==0:
        train_acc=network.accuracy(x_train,t_train)
        test_acc=network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc,test_acc)


x=np.linspace(0,10,itres_num)

"""
plt.subplot(1,2,1)
plt.title("Lossfunction train")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.plot(train_loss_list)

plt.subplot(1,2,2)
plt.title("Lossfunction test")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.plot(test_loss_list)
"""
plt.subplot(1,2,1)
plt.title("train accurary")
plt.xlabel("iteration")
plt.ylabel("acc")
plt.plot(train_acc_list)

plt.subplot(1,2,2)
plt.title("test accurary")
plt.xlabel("iteration")
plt.ylabel("acc")
plt.plot(test_acc_list)
plt.show()