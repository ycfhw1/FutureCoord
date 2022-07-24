import numpy as np
import networkx as nx
from more_itertools import peekable
from pairing_functions import cantor, szudzik

if __name__ == '__main__':
    # np.set_printoptions(threshold=np.inf)
    # test_network=nx.read_gpickle("/home/hello/桌面/xiaofu/FutureCoord/utils/chinanet.gpickle")
    # print(test_network.graph)
    # #marrival平均到达概率
    # with open('/home/hello/下载/germany50.npy', 'rb') as file:
    #     newendpoints=np.empty(shape=(6,43,50,50))
    #     #这个npy文件可能是167天每天分43个时间段，12个节点的网络，可能是连接概率吧,注意这个矩阵不对称因此是双向流矩阵
    #     index1=0
    #     endpoints = np.load(file)
    #     for index2 in range(6):
    #         for index3 in range(43):
    #             newendpoints[index2][index3]=endpoints[index1].reshape(50,50)
    #             index1=index1+1
    #     print(index1)
    #     np.save("germany50.npy",newendpoints)
    # with open('/home/hello/桌面/xiaofu/FutureCoord/data/rates/trace/2.npy', 'rb') as file:
    #     traffic=np.load(file)
    #     print(traffic)
    # seed=10
    # print(cantor.pair(1,2))
    # print(cantor.pair(1, 2))
    # print(cantor.pair(1, 2))
    # print(cantor.pair(1, 2))
    # a=iter([1,2,3,4,5])
    # p=peekable(a)
    # while True:
    #     print(type(next(p)))
    # results = np.random.random(size=100)
    # results_list=[]
    # for result in results:
    #     if result > 0.5:
    #         result = 1
    #         results_list.append(result)
    #     else:
    #         result = 0
    #         results_list.append(result)
    # print(results_list)
    # net=nx.read_gpickle("/home/hello/桌面/xiaofu/FutureCoord/data/experiments/abilene/abilene2.gpickle")
    # #NODES:NAME,COORDS,COMPUTE,MEMORY
    # print(net.nodes(data=True))
    # for u,d in net.nodes(data=True):
    #     d['compute']=d['compute']*5
    #     d['memory']=d['memory']*5
    # for u,v,w in net.edges(data=True):
    #     w['datarate']=w['datarate']*10
    #     w['propagation'] = w['propagation'] /10
    # nx.write_gpickle(net,"/home/hello/桌面/xiaofu/abilene.gpickle")
    nums = [4, 1, 5, 2, 9, 6, 8, 7]

    m_sorted = sorted(enumerate(nums), key=lambda x: x[1])
    print(m_sorted)
    sorted_inds = [m[0] for m in m_sorted]

    sorted_nums = [m[1] for m in m_sorted]
    # print(sorted_inds)
    # print(sorted_nums)
