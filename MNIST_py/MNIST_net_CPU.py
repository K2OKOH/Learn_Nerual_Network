# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 20:35:20 2021

@author: admin
"""
import numpy as np
import struct
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# 训练集文件
train_images_file = 'MNIST/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_file = 'MNIST/train-labels.idx1-ubyte'

# 测试集文件
test_images_file = 'MNIST/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_file = 'MNIST/t10k-labels.idx1-ubyte'

bin_data = open(train_images_file,'rb').read()

def load_images_file(filename):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(filename, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

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
#        plt.imshow(images[i],'gray')
#        plt.pause(0.00001)
#        plt.show()
    #plt.show()

    return images

def load_labels_file(filename):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(filename, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

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

#继承nn.Module构建自己的简单神经网络SNet
class ConvNet(torch.nn.Module):
    #构建三层全连接网络
    def __init__(self,CH_1, CH_2, CH_3, CH_4):
        super(ConvNet, self).__init__()
        #定义每层的结构
        self.features = nn.Sequential(   
            # 28*28*1
            nn.Conv2d(CH_1, CH_2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 28*28*CH_2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 14*14*CH_2
            nn.Conv2d(CH_2, CH_3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 14*14*CH_3
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 7*7*CH_3
            nn.Conv2d(CH_3, CH_4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 7*7*CH_4
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(CH_4 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 10),
        )
        
    #使用ConvNet会自动运行forward（前向传播）方法，方法连接各个隐藏层，并产生非线性
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # 读取标签和图像
    train_images = load_images_file(train_images_file)
    test_images = load_images_file(test_images_file)
    train_labels = load_labels_file(train_labels_file)
    test_labels = load_labels_file(test_labels_file)
    
    #对label进行onehot编码（多分类）
    train_label_oh = np.eye(10)[train_labels.astype(int)]
    test_label_oh = np.eye(10)[test_labels.astype(int)]
    
    
    #训练和测试进行类型转换
    train_feature_t = torch.from_numpy(train_images).unsqueeze(1).float()   #由numpy转换而来是torch.float64
    # print(train_feature_t)
    train_label_t = torch.from_numpy(train_label_oh).float()
    
    # train_feature_t_f = torch.tensor(train_feature_t,dtype=torch.float32)   #转换为torch.float32，因为要和之后模型预测的类型匹配
    # train_label_t_f = torch.tensor(train_label_t,dtype=torch.float32)
    
    test_feature_t = torch.from_numpy(test_images).unsqueeze(1).float()
    test_label_t = torch.from_numpy(test_label_oh).float()
    
    # test_feature_t_f = torch.tensor(test_feature_t,dtype=torch.float32)
    # test_label_t_f = torch.tensor(test_label_t,dtype=torch.float32)
    
    #######################训练过程##############################
    
    #输入维度，隐藏层神经元个数，输出维度
    CH_1, CH_2, CH_3, CH_4 = 1, 5, 10, 15
    
    #实例化一个用于预测的网络
    model = ConvNet(CH_1, CH_2, CH_3, CH_4)
    
    #定义损失函数，使用均方误差
    loss_fn = nn.MSELoss(reduction='sum')
    #设置学习率
    learning_rate = 1e-3
    #优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #网络参数初始化（上面定义的方法）
    # model.apply(weights_init)
    
    #10000轮迭代，可更改
    for t in range(10): 
        # 执行model.forward()前向传播
        y_pred = model(train_feature_t)
        
        #计算本次迭代的误差
        loss = loss_fn(y_pred, train_label_t)
        print("训练次数：",t,"\tloss =",loss.item())
        
        #清零梯度
        optimizer.zero_grad()
        #Backward pass反向传播
        loss.backward()
        #更新参数
        optimizer.step()
    
    #######################测试过程##############################
    
    model.eval()    #保证BN和dropout不发生变化
    
    cnt = 0 #初始化正确的计数值
    
    #输入训练集得到测试结果
    test_out = model(test_feature_t)
    _, test_out_np= torch.max(test_out,1)   #onehot解码，返回值第一个是最大值（不需要），第二个是最大值的序号

    #迭代922个测试样本输出和统计    
    for test_i in range(992):
    
        print("No.",test_i,"\npre:",test_out_np.numpy()[test_i],"\nGT:",test_labels[test_i])
        print("****************")
        if test_out_np.numpy()[test_i] == test_labels[test_i]:
            #print("correct")
            cnt += 1
    
    #正确率计算
    correct_rate = cnt/992.0
    print("correct_rate:",correct_rate)