# 逻辑回归的各种使用方法

## 线性逻辑回归
### 1. LR的基本原理和求解方法

 LR模型中,通过特征权重向量对特征向量的不同维度上的取值进行加权,并用逻辑函数将其压缩到0~1的范围,作为该样本为正样本的概率.逻辑函数为:

**逻辑函数曲线:**

![image](http://latex.codecogs.com/gif.latex?\alpha(x)={1\over{1+e^{-x}}})

给定M个训练样本![image](http://latex.codecogs.com/gif.latex?$(X_1,Y_1),(X_2,Y_2)...(X_M,Y_M)$)
其中![image](http://latex.codecogs.com/gif.latex?$X_j=\{{X_{ji}|i=1,2...N}\}$)
为N维的实际向量,在LR模型中,第j个样本为正样本的概率是:

![image](http://latex.codecogs.com/gif.latex?P(y_j=1|W,X_j)={1\over{1+e^{-W^TX_j}}})

其中W是N维的特征权重向量,也就是LR问题中要求解的模型参数.

### 2. 求解LR问题

  寻找一个合适的特征权重向量W,使得对于训练里面的正样本
![image](http://latex.codecogs.com/gif.latex?$P(y_j=1|W,Xj)$)
值尽量大,对于训练集里面的负样本,这值尽量小.用联合概率来表示:

![image](http://latex.codecogs.com/gif.latex?max_wp(W)=\prod_{j=1}^M{1\over{1+e^{-y_j})

对上式求log取符号,则等价于:

![image](http://latex.codecogs.com/gif.latex?min_Wf(W)=\prod_{j=1}^Mlog(1+e^{-y_jW^TX_j}))

上式就是LR求解的目标函数.寻找合适的W令目标函数![image](http://latex.codecogs.com/gif.latex?$f(W)$)最小，是一个无约束最优化问题，解决这个问题的通用做法是随机给定一个初始的![image](http://latex.codecogs.com/gif.latex?$W_0$)，通过迭代，在每次迭代中计算目标函数的下降方向并更新![image](http://latex.codecogs.com/gif.latex?$W$)，直到目标函数稳定在最小的点。
```flow
st=>start: Start
op=>operation: Step 1:令迭代次数t=0,随机给
定初始特征权重向量w_0
op1=>operation: Step 2:t=t+1
op2=>operation: Step 3:计算第t次迭代中
损失函数的搜索方向D_t
op3=>operation: Step 4:更新特征权重
向量W_t=W_t-1+aDt
cond=>condition: 是否不满足终止条件?
e=>end

st->op->op1->op2->op3->cond
cond(yes)->e
cond(no)->op1
```
不同的优化算法的区别就在于目标函数下降方向![image](http://latex.codecogs.com/gif.latex?$D_t$)的计算,下降方向是通过对目标函数在当前的W下求一阶倒数![image](http://latex.codecogs.com/gif.latex?(梯度，Gradient))和求二阶导数![image](http://latex.codecogs.com/gif.latex?(海森矩阵，Hessian-Matrix))得到。常见的算法有梯度下降法、牛顿法、拟牛顿法。

####  (1) 梯度下降法(Gradient Descent)
 
 梯度下降法直接采用目标函数在当前W的梯度的反方向作为下降方向：
 
![image](http://latex.codecogs.com/gif.latex?D_t=-G_t=-\nabla_Wf(W_t))

其中目标函数的梯度,及其计算方法

![image](http://latex.codecogs.com/gif.latex?G_t=\nabla_Wf(W_t))

![image](http://latex.codecogs.com/gif.latex?G_t=\nabla_wf(W_t)=\sum_{j=1}^M[\sigma(y_jW_t^TX_j)-1]y_jX_j)

![image](http://latex.codecogs.com/gif.latex?G_t=\nabla_wf(W_t)=\sum_{j=1}^M[{1\over1+e^{y_iW_t^TX_j}}-1]y_jX_j)

![image](http://latex.codecogs.com/gif.latex?w^{t+1}=w^t-\alpha\sum_{i=1}^m({1\over1+{e^{-w^tx_i}}}-y_i)x_i)

```python
from numpy import *
def sigmoid(inX):
    return 1.0/(1.0+exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat   = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights
````

#### 牛顿法

![image](http://latex.codecogs.com/gif.latex?\theta(x)=g(\theta^Tx)={1\over1+e^{-\theta^Tx}})

![image](http://latex.codecogs.com/gif.latex?L(\theta)=p(\vec{y}|X;\theta))

![image](http://latex.codecogs.com/gif.latex?L(\theta)=\prod_{i=1}^mp(y^i|x^i;\theta))

![image](http://latex.codecogs.com/gif.latex?L%28%5Ctheta%29%3D%5Cprod_%7Bi%3D1%7D%5Em%28h_%7B%5Ctheta%7D%28x%5Ei%29%29%5E%7By%5Ei%7D%281-h_%7B%5Ctheta%7D%28x%5Ei%29%29%5E%7B1-y%5Ei%7D)

![image](http://latex.codecogs.com/gif.latex?l%28%5Ctheta%29%3DlogL%28%5Ctheta%29%3D%5Csum_%7Bi%3D1%7D%5Emy%5Eilogh%28x%5Ei%29&plus;%281-y%5Ei%29log%281-h%28x%5Ei%29%29)

![image](http://latex.codecogs.com/gif.latex?%7B%5Cpartial%5Cover%7B%5Cpartial%5Ctheta_j%7D%7Dl%28%5Ctheta%29%3D%28y-h_%7B%5Ctheta%7D%28x%29%29x_j)

![image](http://latex.codecogs.com/gif.latex?%5Ctheta%5E%7Bt&plus;1%7D%3D%5Ctheta%5Et-H%5E%7B-1%7D%5Cnabla_%7B%5Ctheta%7DL)

![iamge](http://latex.codecogs.com/gif.latex?H%3D%7B1%5Cover%7Bm%7D%7D%5Csum_%7Bi%3D1%7D%5Em%5Cleft%5B%28h_%7B%5Ctheta%7D%28x%5Ei%29%29%281-h_%7B%5Ctheta%7D%28x%5Ei%29%29x%5Ei%28x%5Ej%29%5ET%5Cright%5D)

![image](http://latex.codecogs.com/gif.latex?\nabla_{\theta}L=-{1\over{m}}{\partial\over{\partial\theta_j}}l(\theta))

```

```matlab
theta = zeros(n, 1);%initialization
z = x * theta;
h = ones(m,1)./(1 + exp(-z));%logistic model
cost_function_pre = sum(y .* log(h)+(1-y).*log(ones(m,1)-h), 1);%likelihood function

%compute first-derivation and second derivation
G = zeros(n , 1);
H = zeros(n, n);
for i = 1:n
    dif = y - h;
    G(i) = sum(dif.* x(:, i), 1)/m;

    const_sum = h.*(ones(m,1)-h);
    for j = 1:n
        H(i,j) = -sum(const_sum.* x(:, i).*x(:,j), 1)/m;
    end
end
theta = theta - inv(H)*G;
```

## Multi-Class Logistic（多分类的Logistic问题）

假设对于一个样本`$x$`,它可能属于`$k$`个分类,其估计值分别为`$f_1(x)...f_k(x)$`,LR变化是一个平滑且数据规范化(使得向量的长度为1)的过程,结果为属于类别k的概率`$p_k(x)$`

![image](http://latex.codecogs.com/gif.latex?f_k(x)=\theta_k^Tx)

![image](http://latex.codecogs.com/gif.latex?p_k(y=k|x)={e^{f_k(x)}\over\sum_{l=1}^ke^{f_l(x)}})

该模型有这样一个性质,对于所有参数加上一个向量`$v$`,后验概率的值不变.即

![image](http://latex.codecogs.com/gif.latex?{e^{(\theta_1+v)^T}\over{\sum_{k=1}^c}e^{(\theta_k+v)^Tx}}={e^{\theta_1^T}\over{\sum_{k=1}^c}e^{\theta_k^Tx}})

加入正则项后的代价函数可以表述为


![image](http://latex.codecogs.com/gif.latex?J(\theta)=-{1\over{m}}\left(\sum_{i=1}^n\sum_{k=1}^c1\{y_k=k\}{log{\theta_kx_i}\over{\sum_{l=1}^c\theta_l^Tx_i}}\right))

其中，![image](http://latex.codecogs.com/gif.latex?$1{yi=k}$)是指示函数，若![image](http://latex.codecogs.com/gif.latex?$yi=k$)为真，则等于1，否则等于0。对于参数的求解，与两类问题的求解方法一样。
```matlab
num_of_data = 90;
num_of_outputs = 3;

X = zeros(num_of_data,2);

X = randi(20,num_of_data/3,2);
X = [X;randi([40 60],num_of_data/3,2)];
temp = [randi(20,num_of_data/3,1) randi([40 60],num_of_data/3,1)];
X = [X;temp];
X = [ones(num_of_data,1) X];

y = zeros(num_of_data,num_of_outputs);
y_0 = zeros(num_of_data,1);
y_1 = zeros(num_of_data,1);
y_2 = zeros(num_of_data,1);

y_0(1:num_of_data/3) = 1;
y_1(num_of_data/3+1:num_of_data*2/3)=1;
y_2(num_of_data*2/3+1:end) = 1;

y = [y_0 y_1 y_2];


m = size(X,1);
iterations = 20000;
learning_rate = 0.005;

theta = ones(size(X,2),3);
hypothesis = (1./(1+exp(-X*theta)));

xmin = -20;
xmax = 100;
ymin = -20;
ymax = 100;

figure
axis([xmin, xmax, ymin, ymax])
hold on

scatter(X(1:num_of_data/3,2),X(1:num_of_data/3,3), 'x', 'b');
scatter(X(num_of_data/3+1:num_of_data*2/3,2),X(num_of_data/3+1:num_of_data*2/3,3), 'x','r');   
scatter(X(num_of_data*2/3:end,2),X(num_of_data*2/3:end,3), 'x','m');
     

y_intercept = -theta(1,:)./theta(3,:);
slope = -y_intercept./(-theta(1,:)./theta(2,:));
 
line1 = plot([xmin xmax],[y_intercept(1)+slope(1)*xmin y_intercept(1)+slope(1)*xmax]);
line2 = plot([xmin xmax],[y_intercept(2)+slope(2)*xmin y_intercept(2)+slope(2)*xmax]);
line3 = plot([xmin xmax],[y_intercept(3)+slope(3)*xmin y_intercept(3)+slope(3)*xmax]);

%% Classification
J_hist = zeros(iterations,3);

for j = 1:iterations
    
    %Cost function of logistic regression
    cost = - y.*log(hypothesis)-(1-y).*log(1-hypothesis);
    J_hist(j,:) = (1/m)*sum(cost);
 
    %Gradient descent of the parameters theta
    theta = theta - learning_rate*(1/m)*X'*(hypothesis-y);
    hypothesis = (1./(1+exp(-X*theta)));
    
     %if statement to update graph every 100 iterations of gradient descent
     if (mod(j, 100) == 0)
         
         x1 = [-theta(1,:)./theta(2,:); zeros(1,3)];
         x2 = [zeros(1,3); -theta(1,:)./theta(3,:)];
        
         y_intercept = -theta(1,:)./theta(3,:);
         slope = -y_intercept./(-theta(1,:)./theta(2,:));
         
         delete(line1)
         delete(line2)
         delete(line3)
         
         line1 = plot([xmin xmax],[y_intercept(1)+slope(1)*xmin y_intercept(1)+slope(1)*xmax]);
         line2 = plot([xmin xmax],[y_intercept(2)+slope(2)*xmin y_intercept(2)+slope(2)*xmax]);
         line3 = plot([xmin xmax],[y_intercept(3)+slope(3)*xmin y_intercept(3)+slope(3)*xmax]);
         
         %Pausing to observe classification lines on graph
         pause(0.01);
     
     end
end
```

## 并行逻辑回归

由逻辑回归问题的求解方法中可以看出，无论是梯度下降法、牛顿法、拟牛顿法，计算梯度都是其最基本的步骤，并且![image](http://latex.codecogs.com/gif.latex?$L-BFGS$)通过两步循环计算牛顿方向的方法，避免了计算海森矩阵。因此逻辑回归的并行化最主要的就是对目标函数梯度计算的并行化。可以看出，目标函数的梯度向量计算中只需要进行向量间的点乘和相加，可以很容易将每个迭代过程拆分成相互独立的计算步骤，由不同的节点进行独立计算，然后归并计算结果。

将M个样本的标签构成一个M维的标签向量，![image](http://latex.codecogs.com/gif.latex?$M$)个![image](http://latex.codecogs.com/gif.latex?$N$)维特征向量构成一个![image](http://latex.codecogs.com/gif.latex?$M*N$)的样本矩阵，如图3所示。其中特征矩阵每一行为一个特征向量(![image](http://latex.codecogs.com/gif.latex?$M$)行)，列为特征维度(![image](http://latex.codecogs.com/gif.latex?$N$)列)。

![样本标签向量 & 特征向量](http://cms.csdnimg.cn/article/201402/13/52fc49a17df7e.jpg)样本标签向量 & 特征向量

如果将样本矩阵按行划分，将样本特征向量分布到不同的计算节点，由各计算节点完成自己所负责样本的点乘与求和计算，然后将计算结果进行归并，则实现了“按行并行的LR”。按行并行的LR解决了样本数量的问题，但是实际情况中会存在针对**高维特征向量**进行逻辑回归的场景（如广告系统中的特征维度高达上亿），仅仅按行进行并行处理，无法满足这类场景的需求，因此还需要按列将**高维的特征向量拆分成若干小的向量**进行求解。

####  (1) 数据分割

假设所有计算节点排列成m行n列（m*n个计算节点），按行将样本进行划分，每个计算节点分配M/m个样本特征向量和分类标签；按列对特征向量进行切分，每个节点上的特征向量分配N/n维特征。如图4所示，同一样本的特征对应节点的行号相同，不同样本相同维度的特征对应节点的列号相同。

![image](http://cms.csdnimg.cn/article/201402/13/52fc4a094e233.jpg)并行LR中的数据分割

一个样本的特征向量被拆分到同一行不同列的节点中，即：![image](http://latex.codecogs.com/gif.latex?$X_{r,k}=<X_{(r,1),k},...,X_{(r,c),k},...,X_{(r,n),k}>$)
其中Xr,k表示第r行的第k个向量，![image](http://latex.codecogs.com/gif.latex?$X_{(r,c),k}$)表示![image](http://latex.codecogs.com/gif.latex?$X_{r,k}$)在第![image](http://latex.codecogs.com/gif.latex?$c$)列节点上的分量。同样的，用![image](http://latex.codecogs.com/gif.latex?$W_c$)表示特征向量![image](http://latex.codecogs.com/gif.latex?$W$)在第![image](http://latex.codecogs.com/gif.latex?$c$)列节点上的分量，即：

![image](http://latex.codecogs.com/gif.latex?W=<W_1,...,W_c,...,W_n>)

#### (2) 并行计算

观察目标函数的梯度计算公式,其依赖于两个计算结果：特征权重向量Wt和特征向量Xj的点乘，标量
![image](http://latex.codecogs.com/gif.latex?[\sigma(y_iW_t^TX_i)-1]y_i$)
和特征向量![image](http://latex.codecogs.com/gif.latex?$X_j$)的相乘。可以将目标函数的梯度计算分成两个并行化计算步骤和两个结果归并步骤：

##### ① 各节点并行计算点乘，计算

![image](http://latex.codecogs.com/gif.latex?d_{(r,c),k,t}=W_{c,t}^TX_{(r,c)k}\in\mathbb{R})

其中![image](http://latex.codecogs.com/gif.latex?k=1,2,…,M/m)，![image](http://latex.codecogs.com/gif.latex?$d_{(r,c),k,t}$)表示第t次迭代中节点(r,c)上的第k个特征向量与特征权重分量的点乘，![image](http://latex.codecogs.com/gif.latex?$W_{c,t}$)为第t次迭代中特征权重向量在第c列节点上的分量。

##### ②对行号相同的节点归并点乘结果：


![image](http://latex.codecogs.com/gif.latex?d_{r,k,t}=W_t^TX_r^k=\sum_{c=1}^nd_{(r,c),k,t}=\sum_{c=1}^nX_{(r,c),k}\in\mathbb{R})

计算得到的点乘结果需要返回到该行所有计算节点中，如图5所示。
![image](http://cms.csdnimg.cn/article/201402/13/52fc4affbe5df.jpg)

点乘结果归并

##### ③ 各节点独立算标量与特征向量相乘：

![image](http://latex.codecogs.com/gif.latex?G_{(r,c),t}=\sum_{k=1}^{M\over{m}}[\sigma(y_{r,k}d_{r,k,t})-1]y_{r,k}X_{(r,c),k}\in\mathbb{R}^{N\over{n}})

可以理解为由第r行节点上部分样本计算出的目标函数梯度向量在第c列节点上的分量。

##### ④ 对列号相同的节点进行归并
```math
G_{(r,c),t}=\sum_{r=1}^mG_{(r,c),t}\in\mathbb{R}^{N\over{n}}
```
`$G_{c,t}$`就是目标函数的梯度向量```$G_t$```在第c列节点上的分量，对其进行归并得到目标函数的梯度向量：

![image](http://latex.codecogs.com/gif.latex?G_t=<G_{1,t},...,G_{c,t},...,G_{n,t}>\in\mathbb{R}^N)


![image](http://cms.csdnimg.cn/article/201402/13/52fc4ba6123a5.jpg)

![image](http://cms.csdnimg.cn/article/201402/13/52fc4bde78780.jpg)
```python
# -*- coding: utf-8 -*-
from pyspark import SparkContext
from math import *

theta = [0, 0, 0]    #初始theta值
alpha = 0.001    #学习速率

def inner(x, y):
    return sum([i*j for i,j in zip(x,y)])
        
def func(lst):
    h = (1 + exp(-inner(lst, theta)))**(-1)
    return map(lambda x: (h - lst[-1]) * x, lst[:-1])


sc = SparkContext('local')

rdd = sc.textFile('/home/freyr/logisticRegression.txt')\
        .map(lambda line: map(float, line.strip().split(',')))\
        .map(lambda lst: [1]+lst)


for i in range(400):
    partheta = rdd.map(func)\
                   .reduce(lambda x,y: [i+j for i,j in zip(x,y)])

    for j in range(3):
        theta[j] = theta[j] - alpha * partheta[j]

print 'theta = %s' % theta
```

## 理解L-BFGS算法

L-BFGS(Limited-Memory BFGS)是BFGS算法在受限内存时的一种近似算法，而BFGS是数学优化中一种无约束最优化算法。

### 无约束优化
---
无约束最优化的基本形式是，给定一个多元函数`$f(x)$`，求出该函数的最小值点![image](http://latex.codecogs.com/gif.latex?$x^*$)。形式化地来说，即：

![image](http://latex.codecogs.com/gif.latex?x^*=argmin_xf(x))

### 牛顿法
---

在使用计算机求解一个优化问题时，通常使用迭代方法。即我们的算法不断迭代，产生一个序列![image](http://latex.codecogs.com/gif.latex?$x_1,x_2,...,x_k$)，若该序列能收敛到![image](http://latex.codecogs.com/gif.latex?$x^*$),则算法是有效的.

假设现在已经有点![image](http://latex.codecogs.com/gif.latex?$x_n$),如何构造![image](http://latex.codecogs.com/gif.latex?$x_{n+1}$),使得![image](http://latex.codecogs.com/gif.latex?$f(x_n)<f(x_{n+1})$)

泰勒公式,将![image](http://latex.codecogs.com/gif.latex?$f(x)$)在固定点![image](http://latex.codecogs.com/gif.latex?$x_n$)处展开,则有:

![image](http://latex.codecogs.com/gif.latex?f%28x_n&plus;%5CDelta%7Bx%7D%29%5Capprox%7Bf%28x_n%29&plus;%5CDelta%7Bx%5ET%7D%5Cnabla%7Bf%28x_n%29%7D&plus;%7B1%5Cover%7B2%7D%7D%5CDelta%7Bx%5ET%7D%28%5Cnabla%5E2%7Bf%28x_n%29%7D%29%5CDelta%7Bx%7D%7D)

其中,![image](http://latex.codecogs.com/gif.latex?$\nabla{f(x_n)}$)和`$\nabla^2{f(x_n)}$`分别为目标函数在点![image](http://latex.codecogs.com/gif.latex?$x_n$)处的梯度和Hessian矩阵.![image](http://latex.codecogs.com/gif.latex?$\left||\Delta{x}||\to{0}$),上面的近似展开式是成立的.

![image](http://latex.codecogs.com/gif.latex?h_n{(\Delta{x})}=f(x_n+\Delta{x})=f(x_n)+\Delta{x^Tg_n}+{1\over{2}}\Delta{x^T}\mathit{H}_n\Delta{x})

当前任务转化为只要找到使![image](http://latex.codecogs.com/gif.latex?$h_n(\Delta{x})$)最小点![image](http://latex.codecogs.com/gif.latex?$\Delta{x}$)即可.

![image](http://latex.codecogs.com/gif.latex?%7B%5Cpartial%7Bh_n%7B%28%5CDelta%7Bx%7D%29%7D%7D%5Cover%7B%5Cpartial%7B%5CDelta%7Bx%7D%7D%7D%7D%20%3D%20g_n&plus;H_n%5CDelta%7Bx%7D%3D0)

![image](http://latex.codecogs.com/gif.latex?\Delta{x}^*=-H_n^{-1}g_n)

![image](http://latex.codecogs.com/gif.latex?x_{n+1}=x_n-\alpha(H_n^{-1}g_n))


### 牛顿法的不足

---
当维度很高时（百万或千万级），此时计算Hessian矩阵的逆几乎是不可能的(存储都很难).

### 拟牛顿法

---

[online latex editor](https://www.overleaf.com/10295655cjzyctddqcmt#/38180161/)
![flow](https://github.com/VisBlank/logistic-regression/blob/master/image/alflow.png?raw=true)

[梯度下降法的三种形式BGD、SGD以及MBGD](http://www.cnblogs.com/maybe2030/p/5089753.html)

[更好的总结,翻墙](http://ruder.io/optimizing-gradient-descent/)

[梯度下降算法中的Adagrad和Adadelta](http://blog.csdn.net/joshuaxx316/article/details/52062291)

Batch gradient descent: Use all examples in each iteration；

Stochastic gradient descent: Use 1 example in each iteration；

Mini-batch gradient descent: Use b examples in each iteration.

[LR Adadelta](https://github.com/saiias/Adadelta)

### 逻辑回归在线

---


![image](http://latex.codecogs.com/gif.latex?g_i=(p_t-y_t)x_i)

![image](http://latex.codecogs.com/gif.latex?\sigma_i={1\over{\alpha}}\sqrt{n_i+g_i^2}-\sqrt{n_i})

![image](http://latex.codecogs.com/gif.latex?z_i\gets{z_i}+g_i-\sigma_iw_{t,i})

[ftrl Follow-the-regularized-Leader 推导](https://www.52ml.net/16256.html)

[ftrl Follow-the-regularized-Leader github](https://github.com/wuyunhao/admm/blob/521721b722d8f1f38593ab1999b7ae3b1e5eaf7d/src/ftrl.cc)
```c++
void FtrlSolver::Update(FtrlSolver::real_t predict, const FtrlSolver::Row& x, const std::vector<FtrlSolver::real_t>& reg_offset) {
  int label = x.label > 0? 1:0;
  //g[i] = (p - y)*x[i]
  auto pre_loss = predict - label;
  
  for(auto i = 0u; i < x.length; ++i) {
    auto loss = pre_loss; 
    auto sigma = (sqrt(squared_sum_[x.index[i]] + loss*loss) - sqrt(squared_sum_[x.index[i]]))/alpha_;
    //z[i] = z[i] + g[i] - sigma*w[i]
    mid_weight_[x.index[i]] += loss - sigma * weight_[x.index[i]];
    //n[i] = n[i] + g[i]^2;
    squared_sum_[x.index[i]] += loss * loss;
  }
}
```
![image](http://img.mp.itc.cn/upload/20161117/b1f77046e3f54f0f8991b771c1ba5982_th.jpeg)

