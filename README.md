# Keras-ToAndroid

## 环境和数据集
克隆代码:
```
git clone https://github.com/Hexmagic/Keras-ToAndroid.git
```
安装keras和TensorFlow
```
pip install tensorflow 
pip install keras
```
下载数据集，这里使用的是科赛一个口罩识别数据集(源数据是Kaggle)，[数据集地址](https://www.kesci.com/mw/dataset/5eda04e4b772f5002d6e4fd2),下载后解压得到下面的结构:
```
mask
├── Test
│   ├── WithMask
│   └── WithoutMask
├── Train
│   ├── WithMask
│   └── WithoutMask
└── Validation
    ├── WithMask
    └── WithoutMask

```

## 训练和转换
使用train.py进行训练，
```
python train.py --root mask --epochs 20
```
感觉精度差不多可以早停

例如得到一个精度不错的模型`sq_5.h5`，使用to_tf.py将其转换为TensorFlow模型
```
python to_tf.py --model sq_5.h5
```
将得到的`squeezenet.pb`复制到安卓项目的asset文件夹下即可

## 更多

这里提一下如何使用自己训练的分类模型,假设你需要训练分类数量位num_classes,那么
1. 首先你要修改`train.py`中最后一个Conv2d的通道数为你需要的num_classes.
2. 修改`MainActivity.java`中的`float[] PREDICTIONS = new float[2];`中的2为num_classes
3. 修改`assets/labels.json`中的字典(类和文字的对应关系)
