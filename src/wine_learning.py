import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.pardir)
from file_import import *
from two_layer_net import TwoLayerNet
from write_file import *
#from multi_layer_net import MultiLayerNet


test_path="./../data_sets/wine.csv"
train_path="./../data_sets/wine_test.csv"
x_test,t_test=file_import(test_path,500,0)
x_train,t_train=file_import(train_path,500,0)

l_times=10
count=0
while count<l_times:
    #network=multiLayerNet(input_size=11,hidden_size=12,hidden_size2=10,output_size=3)
    network=TwoLayerNet(input_size=11,hidden_size=30,output_size=10)
    itres_num=7000
    learning_rate = 0.1
    train_loss_list=[]
    test_loss_list=[]
    train_acc_list=[]
    test_acc_list=[]

    batch_size=100
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
            #print(train_acc,test_acc)
    network=None
    count+=1

    #ファイル書き込み部分
    file_output("./../output_data/test_acc_result",test_acc_list)
    file_output("./../output_data/train_acc_result",train_acc_list)
    file_output("./../output_data/test_loss_result",test_loss_list)
    file_output("./../output_data/train_loss_result",train_loss_list)

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