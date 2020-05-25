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
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        # self.w[5] = -100.
        # self.w[9] = -100.
        self.b = 0.

    def forward(self, x):
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
        z = self.forward(x)
        # 取数据的行数
        N = x.shape[0]
        # 计算w的梯度，总数相加再除以N
        gradient_w = 1. / N * np.sum((z - y) * x, axis=0)
        # 增加维度
        gradient_w = gradient_w[:, np.newaxis]
        # 计算b的梯度，同上
        gradient_b = 1. / N * np.sum(z - y)
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

    # num_epoches为训练的轮数，eta为步长
    def train(self, training_data, num_epoches, batch_size=10, eta=0.01):
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epoches):
            # 打乱样本顺序
            np.random.shuffle(training_data)
            # 将train_data分成多个mini_batch
            # 循环取值，每次取出batch_size条数据
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batche in enumerate(mini_batches):
                # 取mini_batch的前13列
                x = mini_batche[:, :-1]
                # 取mini_batch的最后1列
                y = mini_batche[:, -1:]
                # 前向计算
                a = self.forward(x)
                # 计算损失
                loss = self.loss(a, y)
                # 计算梯度
                gradient_w, gradient_b = self.gradient(x, y)
                # 更新参数
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.format(epoch_id, iter_id, loss))
        return losses


# 获取数据
training_data, test_data = load_data()
# 创建网络
net = Network(13)
# 启动训练，训练50轮，每轮样本数目为100，步长为0.1
losses = net.train(training_data, num_epoches=50, batch_size=100, eta=0.1)

# 画出损失函数的变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()
