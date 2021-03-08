"""
训练线性分类器分类MNIST+LDA降维后再次线性分类，并保存权重

本程序会先运行线性分类器，后运行LDA降维再次训练
由于改自ipynb文件，结构较乱，阅读结构时请参考注释

61518407 李浩瑞 1.13
"""

import gzip
import pickle
import os,sys
import struct
import numpy as np
def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('图片数量: %d张, 图片大小: %d*%d' % (num_images, num_rows, num_cols))
    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print(offset)
    fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    #plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        #print(images[i])
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('图片数量: %d张' % ( num_images))
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def softmax_pred(y_pred):
    y_soft_pred=[]
    for item in y_pred:
        item-=np.max(item)
        y_soft_pred_item=(np.exp(item) / np.sum(np.exp(item)))
        #print(y_soft_pred_item)
        y_soft_pred.append(y_soft_pred_item)
    return y_soft_pred

def val(y_soft_pred,y):
    r,w=0,0
    for i in range(0,60000):
        item_pred,item_real = y_soft_pred[i],y[i]
        list1 = item_pred.tolist()
        max_index1 = list1.index(max(list1))+1
        list2 = item_real.tolist()
        max_index2 = list2.index(max(list2))+1
        if max_index1 == max_index2:
            r+=1
        else:
            w+=1
    return (r/(r+w))

""" 线性分类器 """
#读写文件索引
project_path = os.getcwd()   # 获取当前文件路径的上一级目录
train_image_path = project_path+r'\MNIST_data\train-images.idx3-ubyte'  # 拼接训练路径字符串
train_label_path = project_path+r'\MNIST_data\train-labels.idx1-ubyte'    # 拼接训练路径字符串
test_image_path = project_path+r'\MNIST_data\t10k-images.idx3-ubyte'  # 拼接训练路径字符串
print("Processing data...")
#读取数据
image = decode_idx3_ubyte(train_image_path)
label = decode_idx1_ubyte(train_label_path)
test_image=decode_idx3_ubyte(test_image_path)

train_X=[]
for item in image:
    temp=item.reshape(1,28*28)
    temp=np.append(temp,1)
    train_X.append(temp)
train_Y=[]
for item in label:
    temp=np.array([0 for i in range(10)])
    temp[int(item)]=1
    train_Y.append(temp)
print("Starting training...")
train_x=np.array(train_X)
train_y=np.array(train_Y)
epoch=100
D_in = 785
D_out = 10
learning_rate=5e-7
loss=0
w1 = np.random.rand(D_in,D_out)
b = np.random.rand(D_out)
x=train_x
y=train_y

for it in range(epoch):
    #forward pass前向
    y_pred=x.dot(w1)
    e=(y_pred-np.max(y_pred))
    y_pred=x.dot(w1)
    y_soft_pred=softmax_pred(y_pred)
    #定义loss
    #MSE loss，均方误差
    #MSE是非凸的，所以不一定会找到全局最优，梯度下降过程中也可能先下后上再下
    pre_loss=loss
    loss=np.square(y_soft_pred - y).sum()
    if ((it % 2)==0):
        print("progress:{0}%".format(round((it + 1) * 100 / epoch)),"iter:",it,"Loss:",loss)
        if (np.square(pre_loss-loss)<100 or (pre_loss-loss)<-5000):
            learning_rate*=0.9
            print("lr change")
    grad_Loss_yhat=2.0*(y_soft_pred - y)
    grad_Loss_w1=x.T.dot(grad_Loss_yhat)
    # Update weights
    w1 -= learning_rate * grad_Loss_w1
    if ((it % 20)==0):
        print("grad_Loss_w1:",grad_Loss_w1)
        acc=val(y_soft_pred,y)
        print("now acc is:",acc)

finalacc=val(y_soft_pred,y)
np.save(project_path+r"my_acc{}_w.npy".format(str(finalacc)[:5]),w1)
print("Trainning End! Saved:",project_path+r"my_acc{}_w.npy".format(str(finalacc)[:5]))

"""LDA"""
def LDA(X,y):
    n = 60000
    Dim = 28*28
    dim = 9
    N = 10
    m = np.mean(X,axis=0).reshape(Dim,1)
    X = [[]for i in range((10))] 
    mean = []
    S_B,S_W = np.zeros((Dim,Dim)),np.zeros((Dim,Dim))
    for i in range(n):
        X[y[i]].append(X[i])
        print("X[i]",X[i].shape)
        print("first i",i/n)
    for i in range(N):
        X[i] = np.array(X[i])
        mean.append(np.mean(X[i], axis=0).reshape(Dim, 1))
        print("second i",i/N)
    for i in range(N):
        S_B += np.matmul((mean[i]-m).reshape(len(m),1)) , ((mean[i]-m).reshape(len(m),1).T)
        print("third i",i/N)
    for i in range(N):
        print("forth i",i)
        for j in range(len(X[i])):
            S_W +=np.matmul(((X[i][j].reshape(len(X[i][j]),1)-mean[i])),((X[i][j].reshape(len(X[i][j]),1)-mean[i]).T))
    try:
        S = np.matmul((np.linalg.inv(S_W)) , S_B)
    except:
        S = np.matmul(np.linalg.pinv(S_W) , S_B)
    
    eigVal, eigVec = np.linalg.eig(S)
    
    idx = eigVal.argsort()[::-1]   
    eigVal = eigVal[idx]
    eigVec = eigVec[:,idx]
    
    va = np.zeros((dim,X.shape[1]))
    for i in range(dim):
        va[i] = eigVec[:, i]
    return va

print("Starting LDA...")
train_X=np.array(train_X)
train_y = np.array([int(y) for y in label])
va = LDA(train_X,train_y)

train_X=[]
for item in image:
    temp=item.reshape(28*28,1)
    train_X.append(temp)
train_LDA_X=[]
for item in va:
    print(item)
    temp=item.reshape(1,28*28)
    temp=np.append(temp,1)
    train_LDA_X.append(temp)
print("Staring trainning...")
epoch=150
D_in = 10
D_out = 10
learning_rate=5e-7
loss=0
w2 = np.random.rand(D_in,D_out)
train_falt_X=np.array(train_falt_X)
y=train_y
for it in range(epoch):
    y_pred=train_falt_X.dot(w2)
    e=(y_pred-np.max(y_pred))
    y_soft_pred=softmax_pred(y_pred)
    pre_loss=loss
    loss=np.square(y_soft_pred - y).sum()#都是N*10
    if ((it % 2)==0):
        print("progress:{0}%".format(round((it + 1) * 100 / epoch)),"iter:",it,"Loss:",loss)#,end="\r"
        if (np.square(pre_loss-loss)<100 or (pre_loss-loss)<-5000):
            learning_rate*=0.9
            print("lr change")
    grad_Loss_yhat=2.0*(y_soft_pred - y)
    grad_Loss_w2=train_falt_X.T.dot(grad_Loss_yhat)
    w2 -= learning_rate * grad_Loss_w2
    if ((it % 20)==0 or it == 149):
        print("grad_Loss_w1:",grad_Loss_w1)
        acc=val(y_soft_pred,y)
        print("now acc is:",acc)
finalacc=val(y_soft_pred,y)
np.save(project_path+r"my_LDA_acc{}_w.npy".format(str(finalacc)[:5]),w1)
print("Saved:",project_path+r"my_LDA_acc{}_w.npy".format(str(finalacc)[:5]))