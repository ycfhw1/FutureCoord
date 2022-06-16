import numpy as np


class GRC:
    def __init__(self, damping, alpha, **kwargs):
        self.name='GRC'
        self.damping = damping
        self.alpha = alpha
        #sgrc代表物理节点的评价,rgrc代表请求的评价分数
        self.sgrc, self.rgrc =  [], []

    def learn(self, **kwargs):
        pass

    def predict(self, env, **kwargs):
        if not self.rgrc:
            self.rgrc = self.request_grc(env)

        vgrc = self.rgrc.pop(0)

        # in contrast to the original paper, we recompute the substrate GRC vector after every placement decision
        # since their setting assumes (1) resource demands irrespective of placements
        # and (2) more than one VNF instance may be served by the same node   
        self.sgrc = self.substrate_grc(env)

        argsort = sorted(range(len(self.sgrc)), key=self.sgrc.__getitem__)
        argsort.reverse()

        action = next(node for node in argsort if node in env.valid_routes)
        return action

    def substrate_grc(self, env):
        num_nodes = len(env.net.nodes())

        # compute (normalized) remaining computing and memory resources
        #计算（标准化）剩余的计算和内存资源
        compute = np.asarray(list(env.computing.values()))
        max_compute = np.asarray(list(c for _, c in env.net.nodes('compute')))
        compute = compute / np.sum(max_compute)

        memory = np.asarray(list(env.memory.values()))
        max_memory = np.asarray(list(m for _, m in env.net.nodes('memory')))
        memory = memory / np.sum(max_memory)

        # compute aggregated resource vector (accounts for multiple resources)
        #计算聚合资源向量（占多个资源）
        resources = self.alpha * compute + (1 - self.alpha) * memory

        # determine datarate transition matrix
        datarate = np.zeros(shape=(num_nodes, num_nodes))
        for u, v, data in env.net.edges(data=True):
            datarate[u, v] = data['datarate']
            datarate[v, u] = data['datarate']
        
        # determince grc vector for substrate network
        total_datarate = np.sum(datarate, axis=0)
        datarate = datarate / total_datarate[:, np.newaxis]
        #@矩阵乘法运算符np.linalg.inv矩阵求逆
        substrate_grc = (1 - self.damping) * np.linalg.inv(np.eye(num_nodes) - self.damping * datarate) @ resources
        
        return list(substrate_grc)
    #打分算法，对于请求的VNF，按照VNF长度进行排序
    def request_grc(self, env):
        num_vnfs = len(env.request.vtypes)

        # in our scenario, requested resources depend on the placement, i.e. consider an aggregation of resource demands
        #一维数组，代表了每一种VNF所给当前节点带来的负载变化量
        resources = np.asarray([self._mean_resource_demand(env, env.request, vtype) for vtype in env.request.vtypes])
        resources = resources / np.sum(resources)

        # normalized transition matrix for linear chain of VNFs is the identity matrix
        #VNFs线性链的归一化转移矩阵是单位矩阵
        datarate = np.eye(num_vnfs)
        #np.linalg.inv矩阵求逆,@代表矩阵乘法
        request_grc = (1 - self.damping) * np.linalg.inv(np.eye(num_vnfs) - self.damping * datarate) @ resources
        return list(request_grc)     

    def _mean_resource_demand(self, env, req, vtype):
        config = env.vnfs[vtype]
        demand = []

        for node in env.net.nodes():
            # compute resource demand after placing VNF of `node`
            #supplied_rate代表放置前的总数率,当前节点，指定VNF类型,之前服务的总速率
            supplied_rate = sum([service.datarate for service in env.vtype_bidict[(node, vtype)]])
            after_cdem, after_mdem = env.score(supplied_rate + req.datarate, config)
            prev_cdem, prev_mdem = env.score(supplied_rate, config)
            #计算流增长带来的计算资源的增长量
            cincr = (after_cdem - prev_cdem) / env.net.nodes[node]['compute']
            #计算流增长带来的存储资源的增长量
            mincr = (after_mdem - prev_mdem) / env.net.nodes[node]['memory']

            # compute aggregated increase (accounts for multiple rather than single resource type)
            #计算累计增长（考虑多个而非单一资源类型）
            incr = self.alpha * cincr + (1 - self.alpha) * mincr

            # filter invalid placements (infinite resource demands)
            #筛选无效位置（无限资源需求）对每一个节点都进行了遍历，计算放置一个VNF所可能带来的影响
            if incr >= 0.0:
                demand.append(incr)

        return np.mean(demand)