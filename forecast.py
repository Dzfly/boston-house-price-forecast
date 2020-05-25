import numpy as np
import matplotlib.pyplot as plt


def load_data():
    # 读取以空格分开的文件，变成一个连续的数组
    firstdata = np.fromfile('housing.data', sep=' ')
    # 添加属性
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                     'MEDV']
    # 列的长度
    feature_num = len(feature_names)
    # print(firstdata.shape)  输出结果:(7084, )
    # print(firstdata.shape[0] // feature_nums)  输出结果:506
    # 构造506*14的二维数组
    data = firstdata.reshape([firstdata.shape[0] // feature_num, feature_num])

    # 训练集设置为总数据的80%
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    # print(training_data.shape)

    # axis=0表示列
    # axis=1表示行
    # \表示换行，无需输入
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), training_data.sum(axis=0) / \
                               training_data.shape[0]
    # 查看训练集每列的最大值、最小值、平均值
    # print(maximums, minimums, avgs)

    # 对所有数据进行归一化处理
    for i in range(feature_num):
        # print(maximums[i], minimums[i], avgs[i])
        # 归一化，减去平均值是为了移除共同部分，凸显个体差异
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 覆盖上面的训练集
    training_data = data[:offset]
    # 剩下的20%为测试集
    test_data = data[offset:]
    return training_data, test_data


class Network(object):
    def __init__(self, num_of_weights):
        # 随即产生w的初始值
        # seed(0)表示设置了随机种子，保证程序每次运行结果的一致性
        np.random.seed(0)
        # self.w的结构为num_of_weights行，1列
        self.w = np.random.randn(num_of_weights, 1)
        # b初始化为0
        self.b = 0.

    def forward(self, x):
        # dot()功能：向量点积和矩阵乘法
        # 根据下面x的取值可以确定x和z的结构
        z = np.dot(x, self.w) + self.b
        return z

    def loss(self, z, y):
        # 根据下面y的取值可以确定y的结构
        error = z - y
        # num_samples为总行数404
        num_samples = error.shape[0]
        # cost为均方误差，用来评价模型的好坏
        cost = error * error
        # 计算损失时需要把每个样本的损失都考虑到
        # 对单个样本的损失函数进行求和，并除以样本总数
        cost = np.sum(cost) / num_samples
        return cost

    def gradient(self, x, y):
        # 调用forward函数，得到z
        z = self.forward(x)
        # 计算w梯度，得到一个13维向量，每个分量分别代表该维度的梯度
        gradient_w = (z - y) * x
        # 均值函数mean：求均值
        # axis 不设置值，对 m*n 个数求均值，返回一个实数
        # axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
        # axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
        gradient_w = np.mean(gradient_w, axis=0)
        # 增加维度，变成 n * 1 的矩阵
        gradient_w = gradient_w[:, np.newaxis]
        # 计算b的梯度
        gradient_b = (z - y)
        # b为一个数值，不需要再增加维度
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b

    # 确定损失函数更小的点
    # 更新梯度
    def update(self, gradient_w, gradient_b, eta=0.01):
        # 更新参数
        # 相减：参数需要向梯度的反方向移动。
        # eta：控制每次参数值沿着梯度反方向变动的大小，即每次移动的步长，又称为学习率。
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

    # 迭代100次，每次移动0.01
    def train(self, x, y, iterations=100, eta=0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)
            # 循环输出i末尾为9，间隔为10的数据
            if (i + 1) % 10 == 0:
                print('iter {}, loss{}'.format(i, L))
        return losses


# 获取数据
training_data, test_data = load_data()
# 取训练集全部行的前13列
x = training_data[:, :-1]
# 取训练集全部行的最后一列
y = training_data[:, -1:]
# 创建网络
net = Network(13)
num_iterations = 1000
# 启动训练，迭代次数为1000，步长为0.01
losses = net.train(x, y, iterations=num_iterations, eta=0.01)

# 画出损失函数的变化趋势
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()
