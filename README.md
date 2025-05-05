本文是使用了MobileNet V2模块作为特征提取网络，之后进行代价体构建，再进行3D卷积编码和反卷积。
其中训练所用的图片分别放在./Dataset之中，./Dataset/left存放左视图，./Dataset/right存放右视图，./Dataset/disp存放视差图
model.py存放模型代码，train.py存放训练代码，test.py存放测试代码，KITTI_Loader.py存放加载图片代码，./checkpoints中存放已经训练好的模型
目前问题是已经训练好的模型用来测试发现生成的视差图为全黑，并且结果不理想，不知道是test.py内容出现问题还是模型训练出现问题。
