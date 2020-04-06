# Supervised

# Unsupervised

# Reinforcement

# 损失函数

## 交叉熵

**基本概念**

* 信息量

  > 事件A：巴西队进入了2018世界杯决赛圈。
  >
  > 事件B：中国队进入了2018世界杯决赛圈。
  >
  > 仅凭直觉来说，显而易见事件B的信息量比事件A的信息量要大。究其原因，是因为事件A发生的概率很大，事件B发生的概率很小。所以当越不可能的事件发生了，我们获取到的信息量就越大。越可能发生的事件发生了，我们获取到的信息量就越小。

  信息量和事件发生的概率有关
  $$
  I(x_0)=-log(p(x_0))
  $$

* 熵

  > 某个时间存在n种可能 每一种可能都对应这一个概率$p(x_i)$
  >
  > 熵为该事件信息量的期望

  $$
  H(X)=-\sum_{i=1}^{n}p(x_i)log(p(x_i))
  $$

* KL散度 - 相对熵

  > 同一个事件存在两个不同的概率分布(真实概率分布 - label & 预测概率分布 - prediction)
  >
  > 可以使用KL散度 / 相对熵 来衡量两个分布的差异

  $$
  D_{KL}(p||q)=\sum_{i=1}^{n}p(x_i)log(\dfrac{p(x_i)}{q(x_i)})=\sum_{i=1}^{n}p(x_i)log(p(x_i)) - \sum_{i=1}^{n}p(x_i)log(q(x_i))
  $$

* 交叉熵

  > 公式变形后，要评估label 与 pred 之间的差距 去掉不变的$\sum_{i=1}^{n}p(x_i)log(p(x_i)) = -H(p(x))$ 所以优化过程中只需要关注交叉熵即可

  $$
  H(p,q) = -\sum_{i=1}^{n}p(x_i)log(q(x_i))
  $$

* from logits

  > 在进行交叉熵计算时 logits表示输入的预测结果没有经过激活函数的激活 默认分布在 (-inf, inf)
  >
  > 实现中就会按照内部机制的激活函数先将预测结果进行激活 `sigmoid` or `softmax`再进行计算

* label smoothing

  > 是分类问题中对错误标注学习不当的一种解决方法
  >
  > ~~~html
  > label采用one-hot编码后
  > 1. 无法保证模型的泛化能力，容易造成过拟合
  > 	如果预测结果为 [1., 5., 4.] 经过softmax [0.013, 0.721, 0.265] y = [0, 1, 0]
  >     loss = 0.327 优化时仍然让0.721 更大的接近于1 可能会造成过拟合
  > 2. 全概率和0概率鼓励所属类别和其他类别之间的差距尽可能加大，而由梯度有界可知，这种情况很难适应。会造成模型过于相信预测的类别
  > ~~~
  >
  > 
  >
  > 所以采用label smoothing缓解这一现象

  $$
  q^{'}(k|x)=(1-\varepsilon)\delta_{k,y}+\varepsilon u(k)
  $$

  > $\delta$为狄拉克函数
  >
  > 在除了零以外的点函数值都等于零，而其在整个定义域上的积分等于1

  > 相当于对label进行了一次平滑惩罚 - 不再是one-hot形式
  >
  > 比如label本来是 [0 1], 设置$\varepsilon$为0.1
  >
  > 平滑过程：$[0,1] * (1 - 0.1)$ + 0.1 / 2 = [0.05, 0.95]

**binary_crossentropy**

* 使用sigmoid激活后计算交叉熵









# 前向传播算法

# 后向传播算法

# Dropout方法

**AlphaDropout**

~~~python
tensorflow - keras - layers - AlphaDropout
~~~

**Dropout**

```
tensorflow - keras - layers - Dropout
```



# 超参数搜索

* 网格搜索
*  遗传算法
* 启发式搜索

> 应用：AutoML
>
> 1 使用循环神经网络来生成参数
>
> 2 使用强化学习来进行反馈 使用模型来训练生成参数

* 

# 归一化方法

**批归一化**

~~~~python
tensorflow - keras - layers - BatchNormalization
~~~~

**最值归一化**

~~~python
# 一维归一化
x = np.random.randint(0,100,size=100)
(x - np.min(x)) / (np.max(x) - np.min(x))

# 矩阵多维归一化
X = np.random.randint(0,100,size=(100,2))
X = np.array(X,dtype=float)
X[:,0] = (X[:,0] - np.min(X[:,0])) / (np.max(X[:,0]) - np.min(X[:,0]))# 多维的时候一维一维的来进行归一化
X[:,1] = (X[:,1] - np.min(X[:,1])) / (np.max(X[:,1]) - np.min(X[:,1]))
~~~

![1566288221184](机器学习程序方法.assets/1566288221184.png)

* 适用于分布有明显边界的情况：满分零分，图像像素的像素点0-255
* 受outlier影响较大：无明显边界：收入，存在极高的收入100,100,100,10000000效果极差

**均值方差归一化**

~~~python
X2 = np.random.randint(0,100,(50,2))
X2 = np.array(X2,dtype=float)
for i in range(2):
	X2[:,i] = (X2[:,i] - np.mean(X2[:,i])) / np.std(X2[:,i])
~~~

<img src="机器学习程序方法.assets/1566288302824.png" alt="1566288302824" style="zoom:25%;" />

* 适用于数据分布没有明显边界，有可能存在极端数据值
* 把所有数据归一到均值为0方差为1的分布中

## sklearn

**StandardScaler 均值方差归一化**

~~~python
from sklearn.preprocessing import StandardScaler
standarScaler = StandardScaler()
# 先进行数据处理获得归一化所需要的一些数据然后利用transform进行归一化处理
standarScaler.fit(X_train)
# 对应四个特征的均值
standarScaler.mean_
# 描述数据的分布范围，方差仅仅是一种指标
standarScaler.scale_
# 归一化处理
X_train = standarScaler.transform(X_train)
# 测试集也不能忘
X_test_standard = standarScaler.transform(X_test)
~~~

**实现**

~~~python
# Auther:Jiancong Cui_Butterflier

import numpy as np
class StandardScaler:

	def __init__(self):
		self.mean_ = None
		self.scale_ = None
	def fit(self,X):
		assert X.ndim == 2,"The dimension of X must be 2"

		self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
		self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])

		return self

	def tranform(self,X):
		assert X.ndim == 2,"The dimension of X must be 2"
		assert self.mean_ is not None and self.scale_ is not None,\
			"must fit before transform"
		assert X.shape[1] == len(self.mean_),\
			"the feature number of X must be equal to mean_ and std_"
		resX = np.empty(shape=X.shape,dtype=float)

		for col in range(X.shape[1]):
			resX[:,col] = (X[:,col] - self.mean_[col]) / self.scale_[col]

		return resX
~~~

# TensorFlow

* 基础API

* 基础数据类型

* 自定义损失函数

* 自定义层次

* `tf.function`

  > 1 - 将python函数编译成图结构
  >
  > 2 - 易于将模型导出为 GraphDef + checkpoint / SavedModel
  >
  > 3 - 使得eager execution 可以默认打开
  >
  > 4 - 替代并加强session

# keras

## layers

**Dense**

> * units
>
> * activation
>
> * use_bias
>   * Boolean
> * kernel_initializer
> * bias_initializer
> * kernel_regularizer
> * bias_regularizer
> * activity_regularizer
> * kernel_constraint
> * bias_constraint

# 模型

## Wide & Deep 

* 稀疏特征

  > one-hot编码表示的离散值特征
  >
  > 可做叉乘获取共现信息 - 实现记忆效果

  * 优点

    > 有效 - 广泛用于工业界 - 推荐算法 - 点击率预估

  * 缺点

    > 需要人工设计特征
    >
    > 可能过拟合 - 记住了每一个样本

* 密集特征

  > 向量表达
  >
  > <img src="机器学习程序方法.assets/image-20200310195800296.png" alt="image-20200310195800296" style="zoom:50%;" />
  >
  > 应用：Word2vec工具

  * 优点

    > 带有语义信息(上下文) - 不同向量之间有相关新
    >
    > 兼容没有出现过的特征组合
    >
    > 更少的人工参与

  * 缺点

    > 过度泛化 - 实验效果不好

* 模型结构

  > Wide <- one-hot表达的稀疏特征
  >
  > Deep <- 多层神经网络的结果

  <img src="机器学习程序方法.assets/image-20200310200237444.png" alt="image-20200310200237444" style="zoom:50%;" />


* TensorFlow

~~~python
input = keras.layers.Input(shape=X.shape[1:])
hidden1 = keras.layers.Dense(unitNum, activation = 'relu')(input)
hidden2 = keras.layers.Dense(unitNum, activation = 'relu')(hidden1)

concat = keras.layers.concatenate([input, hidden2])
output = keras.layers.Dense(1)(concat)

model = keras.model.Model(inputs = [input], outputs = [output])

model.conpile(loss = 'mean_squared_error', optimize = 'sgd', metrics = ['accuracy'])
callbacks = [
    keras.callbacks.EarlyStopping(patience = 5, min_delta = 1e-2)
]

history = model.fit(x_train, y_train, validation_data = (x_valid, y_valid), epochs = 100, callbacks = callbacks, verbose = 2)
~~~

# AlexNet

**ImageNet classification with deep convolutional neural networks**

# VGG

**Very Deep Convolutional Networks for Large-Scale Image Recognition**

# NIN

**network in network**

# ResNet

**Deep Residual Learning for Image Recogniton**

# Inception V1 - V4

# MobileNet

**Efficient Convolutional Neural Networks for Mobile Vision**

# NASNet

**Learning Transferable Architectures for Scalable Image Recognition**

# ShakeShake

**Shake - Shake regularization**

# 循环神经网络

**Embedding**

> * One - hot
>
>   > word - find index in the dictionary - [ 0 0 0 0 0 1 - - - 0]
>
> * Dense Embedding
>
>   > Word - index - [1.2, 4.2, 2.9 - - - , 0.1]
>   >
>   > 一开始随机编码 - 后来训练

**变长输入**

> * padding / 截断
>
>   > Word index - [3, 2, 5, 9, 1]
>   >
>   > Padding - [3, 2, 5, 9, 1, 0, 0, 0, 0, 0] - 0表示未知词
>
> * 合并
>
>   > <img src="机器学习程序方法.assets/image-20200319171317157.png" alt="image-20200319171317157" style="zoom:33%;" />
>   >
>   > 

## LSTM

# 卷积神经网络

## CNN

> [ 卷积层 +(可选)池化层 ] * N + 全连接层 * M (N >= 1, M>= 0)

**全卷积神经网络**

> 卷积 + 池化操作可能会使得输入的规模越来越小 - 添加反卷积操作以恢复输入的规模
>
> [ CNN + (Optional) Pooling ] * N + 反卷积层 * K 
>
> - 应用
>
>   > 物体分割 - 输入输出的尺寸一致 - 判断像素点属于哪个物体

<img src="机器学习程序方法.assets/image-20200318122539728.png" alt="image-20200318122539728" style="zoom:33%;" />

**卷积操作**

> 解决的问题：全连接层导致训练参数过多
>
> 卷积操作 - 局部连接 - 参数共享 - 共用卷积核
>
> 理论基础 - 图像的区域性
>
> padding - 外围补零 - 使输出size不变
>
> Stride - 滑动步长
>
> * 规律
>
>   Output_size = Origin_size - kernel_size + 1
>
>   Output_size = Origin_size + (padding_size * 2) - (kernel_size - 1)

**池化操作**

> * Max - pool
>
> * Mean - pool
>
> * 特点
>
>   > 1. 不重叠 - 池化层的大小一般等于 Stride
>   > 2. 不补零 - 最后不够计算时直接丢弃
>   > 3. 没有用于求导的参数
>   > 4. 用于减少图像的尺寸 - 减少计算量
>   > 5. 一定程度平移鲁棒

## 深度可分离卷积

<img src="机器学习程序方法.assets/image-20200319110122500.png" alt="image-20200319110122500" style="zoom:33%;" />

> BN - BatchNormalization
>
> 启发自Inception V3

<img src="机器学习程序方法.assets/image-20200319110519902.png" alt="image-20200319110519902" style="zoom:33%;" />

> * 思想
>
> 1. 视野域 - 输出单元与多少输入单元相关 - 可叠加

<img src="机器学习程序方法.assets/image-20200319110611992.png" alt="image-20200319110611992" style="zoom:33%;" />

> 2. 提升效率 - 相对于不做分支提升了效率

**深度可分离卷积模型结构**

<img src="机器学习程序方法.assets/image-20200319110949068.png" alt="image-20200319110949068" style="zoom:33%;" /><img src="机器学习程序方法.assets/image-20200319111229251.png" alt="image-20200319111229251" style="zoom:33%;" />

> * 计算量 - 普通卷积
>
>   > $D_k ·D_k·M·N·D_F·D_F$ 
>   >
>   > $D_k$ - 卷积核 $D_F$ - 输入图像 
>   >
>   > M - 输入通道 N - 输出通道
>   >
>   > 在M个通道上同时进行卷积核操作 $D_k ·D_k·M$ ， 滑动与输入图像大小有关$D_F·D_F$ 
>   >
>   > 多个卷积核 - N
>
> * 计算量 - 深度可分离卷积 - 分支输入
>
>   > 深度可分离 - $D_k ·D_k·M·D_F·D_F$ 
>   >
>   > 1 * 1 卷积 - $M·N·D_F·D_F$
>
> * 优化比例
>
>   > $$
>   > \dfrac{D_k ·D_k·M·D_F·D_F+M·N·D_F·D_F}{D_k ·D_k·M·N·D_F·D_F}
>   > = \dfrac{1}{N}+\dfrac{1}{D_K^{2}}
>   > $$



## 数据增强

## 迁移学习

# 小知识

**`__init__.py`的作用**

> 1. 标识该目录是一个python的模块包 - `model package`
>
> 2. 简化模块导入操作
>
>    > 如果目录中包含了 **__init__.py** 时，当用 import 导入该目录时，会执行 **__init__.py** 里面的代码
>    >
>    > ~~~python
>    > # mypackage/__init__.py
>    > print("You have imported mypackage")
>    > -------------------------------------------------
>    > import mypackage
>    > >>> You have imported mypackage
>    > ~~~
>    >
>    > 