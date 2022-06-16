import numpy as np
import networkx as nx
from more_itertools import peekable
from pairing_functions import cantor, szudzik

if __name__ == '__main__':
    # np.set_printoptions(threshold=np.inf)
    # test_network=nx.read_gpickle("/home/xiaofu/FutureCoord/data/experiments/germany50/germany50.gpickle")
    # print(test_network)
    # #marrival平均到达概率
    # with open('/home/xiaofu/FutureCoord/data/experiments/abilene/abilene.npy', 'rb') as file:
    #     #这个npy文件可能是167天每天分43个时间段，12个节点的网络，可能是连接概率吧,注意这个矩阵不对称因此是双向流矩阵
    #     endpoints = np.load(file)
    #     print((endpoints).shape)
    # seed=10
    # print(cantor.pair(1,2))
    # print(cantor.pair(1, 2))
    # print(cantor.pair(1, 2))
    # print(cantor.pair(1, 2))
    a=iter([1,2,3,4,5])
    p=peekable(a)
    while True:
        print(type(next(p)))