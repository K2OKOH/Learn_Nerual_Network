# Learn_Nerual_Network
本项目都是非常简单的网络程序，学习如何构建网络。
数据集都是手写识别，分为：
pendigits：已经提取好特征的结构化数据集
MNIST：未提取特征的单通道图像非结构化数据集

## pendigits_py 文件夹
处理结构化的数据，读取excel表格中的特征数据,其中:  
**python程序:**  
- pendigits.py  -->  在CPU上 训练+测试
- pendigits_GPU.py  --> 在GPU上 训练+测试
- pendigits_test_load.py  --> 在CPU上 训练 + 保存(到SaveData文件夹中)
- pendigits_train_save.py  --> 在CPU上 读取 + 测试(从SaveData文件夹中)
**文件夹:**  
- dataset --> 存储excel手写数据集
- SaveData  --> 保存的模型


## MNIST_py 文件夹
处理非结构化的图片数据集,其中:  
**python程序:**
- MNIST_net_CPU.py  -->  在CPU上 训练+测试(慢)
- MNIST_net_GPU.py  -->  在GPU上 训练+测试(需要cuda)
- MNIST_net_GPU_Save.py  -->  在GPU上 训练 + 保存模型(到SaveModel文件夹中)
- MNIST_net_GPU_Load.py  -->  在CPU上 读取模型 + 测试(从SaveModel文件夹中)
- FeatureShow.py
  - --> 绘制网络学习到的卷积核(保存在SaveKernelMap文件夹中)
  - --> 绘制网络对单张图提取的特征(保存在SaveFeatureMap文件夹中)

**文件夹:**  
- MNIST  -->  存储MNIST手写数据集
- SaveModel  -->  训练的模型(整个模型 或 仅参数)
- SaveKernelMap  -->  学习到的卷积核,保存图像
- SaveFeatureMap  -->  各层提取的特征图,保存图像
