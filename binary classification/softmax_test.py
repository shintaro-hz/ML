from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata, fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np

np.random.seed(34)

# logの中身が0になるのを防ぐ
def np_log(x):
    return np.log(np.clip(a=x, a_min=1e-10, a_max=x))

# softmax
def softmax(x):
    x -= x.max(axis=1, keepdisms=True)
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

#minstのデータ（手書きの数字の画像）をダウンロード
mnist = fetch_openml(name='mnist_784')

x_mnist = mnist.data.astype('float32') / 255.
t_mnist = np.eye(N=10)[mnist.target.astype('int32')]

x_train_mnist, x_test_mnist, t_train_mnist, t_test_mnist = train_test_split(x_mnist, t_mnist, test_size=10000)
x_train_mnist, x_valid_mnist, t_train_mnist, t_valid_mnist = train_test_split(x_train_mnist, t_train_mnist, test_size=10000)


# 重み (入力の次元数: 784, 出力の次元数: 10)
W_mnist = np.random.uniform(low=-0.08, high=0.08, size=(784, 10)).astype('float32')
b_mnist = np.zeros(shape=(10,)).astype('float32')

def train_mnist(x, t, eps=1.0):
    """
    :param x: np.ndarray, 入力データ, shape=(batch_size, 入力の次元数)
    :param t: np.ndarray, 教師ラベル, shape=(batch_size, 出力の次元数)
    :param eps: float, 学習率
    """
    global W_mnist, b_mnist
    
    batch_size = x.shape[0]
    
    # 順伝播
    y = softmax(np.matmul(x, W_mnist) + b_mnist) # shape: (batch_size, 出力の次元数)
    
    # 逆伝播
    cost = (- t * np_log(y)).sum(axis=1).mean()
    delta = y - t # shape: (batch_size, 出力の次元数)
    
    # パラメータの更新
    dW = np.matmul(x.T, delta) / batch_size # shape: (入力の次元数, 出力の次元数)
    db = np.matmul(np.ones(shape=(batch_size,)), delta) / batch_size # shape: (出力の次元数,)
    W_mnist -= eps * dW
    b_mnist -= eps * db

    return cost

def valid_mnist(x, t):
    y = softmax(np.matmul(x, W_mnist) + b_mnist)
    cost = (- t * np_log(y)).sum(axis=1).mean()
    
    return cost, y

#学習
for epoch in range(3):
    # オンライン学習
    x_train_mnist, t_train_mnist = shuffle(x_train_mnist, t_train_mnist)
    for x, t in zip(x_train_mnist, t_train_mnist):
        cost = train_mnist(x[None, :], t[None, :])
    cost, y_pred = valid_mnist(x_valid_mnist, t_valid_mnist)
    print('EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(
        epoch + 1,
        cost,
        accuracy_score(t_valid_mnist.argmax(axis=1), y_pred.argmax(axis=1))
    ))

