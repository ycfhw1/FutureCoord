import heapq
import logging
import time
from typing import List, Dict, Tuple
from itertools import islice, chain

import gym
import numpy as np
import networkx as nx
from gym import spaces
from munch import munchify
from tabulate import tabulate
from collections import Counter
from more_itertools import peekable

from coordination.environment.bidict import BiDict
from coordination.environment.traffic import Request, Traffic


class ServiceCoordination(gym.Env):

    def __init__(self, net_path: str, process: Traffic, vnfs: List, services: List):
        # initalize constants from graph description
        # 从network描述当前网络
        self.time_windows = []
        self.net_path = net_path
        self.net = nx.read_gpickle(self.net_path)
        self.NUM_NODES = self.net.number_of_nodes()
        self.MAX_DEGREE = max([deg for _, deg in self.net.degree()])
        self.MAX_COMPUTE = self.net.graph['MAX_COMPUTE']
        self.MAX_LINKRATE = self.net.graph['MAX_LINKRATE']  # in MB/s
        self.MAX_MEMORY = self.net.graph['MAX_MEMORY']  # in MB
        self.HOPS_DIAMETER = self.net.graph['HOPS_DIAMETER']  # in ms
        self.PROPAGATION_DIAMETER = self.net.graph['PROPAGATION_DIAMETER']  # in ms
        # 表示当前的处理过程
        self.process: Traffic = process
        self.vnfs: List[dict] = vnfs
        self.services: List[List[int]] = services
        self.NUM_SERVICES = len(self.services)
        self.MAX_SERVICE_LEN = max([len(service) for service in self.services])
        # REJECT_ACTION拒绝操作,这个东西为什么跟节点数有关有待分析
        self.REJECT_ACTION = self.NUM_NODES + 1
        self.planning_mode = False

        # track resource requirements of prior service request; track admission or dismissal
        # 跟踪先前服务请求的资源需求；跟踪录取或解聘
        self.occupied = {'compute': 0.0, 'memory': 0.0, 'datarate': 0.0}
        # 初始阶段所有请求都是false不被接受
        self.admission = {'deployed': False, 'finalized': False}

        # initialize action and observation space
        # 初始化动作和观察空间
        self.ACTION_DIM = len(self.net.nodes) + 1
        self.action_space = spaces.Discrete(self.ACTION_DIM)
        self.NODE_REPR_SIZE = 7
        # 设置服务观测空间
        self.SERVICE_REPR_SIZE = len(
            self.services) + len(self.vnfs) + len(self.net.nodes) + 4
        # 设置全局观测对象graph_repr
        self.GRAPH_REPR_SIZE = len(self.vnfs) + 3
        # 总的观测空间
        self.OBS_SIZE = len(self.net.nodes) * self.NODE_REPR_SIZE + \
                        self.SERVICE_REPR_SIZE + self.GRAPH_REPR_SIZE
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float16)

        # setup basic debug logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())

        # setup layout for rendering functionality; setup step info
        # 设置渲染功能的布局；设置步骤信息
        self.pos = None
        self.info = None
        self.observation_time = 1
        # reset方法生成了一个具体的请求他是万物之祖哦，done的逻辑要到progress_time的逻辑里去
        self.reset()

    def compute_node_state(self, node) -> np.ndarray:
        '''Define node level statistics for state representation.'''
        # 定义状态表示的节点统计信息，对应于NODE_REPR=7
        if not node in self.valid_routes:
            nstate = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
            return nstate
        # (1) node is a valid placement & valid route to node exists?
        # 当前节点是否还能放置VNF是否还有到其他节点的合法路由
        valid = float(node in self.valid_routes)

        # (2) residual latency of request after placement on node
        # 在节点上放置请求后的剩余延迟,特定节点到这个节点的路由表
        route = self.valid_routes[node]
        latency = sum([self.propagation[frozenset({u, v})] for u, v in route])
        latency = np.nanmin([latency / self.request.resd_lat, 1.0])

        # (3) quantify datarate demands for placement with #hops to node
        # 量化节点跳数放置的数据速率需求,确定具体的跳数
        hops = len(route) / self.HOPS_DIAMETER

        # (4) increase of allocated compute/memory after potential placement on node
        # 在节点上潜在放置后，分配的计算/内存增加,vnum到底是什么啊,virtualnumber,在服务功能链中遍历用的,第一个vnf，第二个vnf
        vnum = len(self.vtype_bidict.mirror[self.request])
        vtype = self.request.vtypes[vnum]
        config = self.vnfs[vtype]
        #current_time all datarate used by the current node for this vtype
        supplied_rate = sum(
            [service.datarate for service in self.vtype_bidict[(node, vtype)]])
        # consumedemand,根据数据速率计算vnf需求
        after_cdem, after_mdem = self.score(
            supplied_rate + self.request.datarate, config)
        prev_cdem, prev_mdem = self.score(supplied_rate, config)
        cdemand = np.clip((after_cdem - prev_cdem) /
                          self.MAX_COMPUTE, a_min=0.0, a_max=1.0)
        mdemand = np.clip((after_mdem - prev_mdem) /
                          self.MAX_MEMORY, a_min=0.0, a_max=1.0)

        # (5) residual compute/memory after placement on node
        # 放置在节点上后的剩余计算/内存
        resd_comp = np.clip(
            (self.computing[node] - cdemand) / self.MAX_COMPUTE, a_min=0.0, a_max=1.0)
        resd_mem = np.clip(
            (self.memory[node] - mdemand) / self.MAX_MEMORY, a_min=0.0, a_max=1.0)

        return [valid, latency, hops, cdemand, mdemand, resd_comp, resd_mem]

    def compute_state(self) -> np.ndarray:
        '''Compute state representation of environment for RL agent.'''
        # RL代理所感知到的计算环境
        if self.planning_mode or self.done:
            return np.zeros(self.OBS_SIZE)

        node_stats = [self.compute_node_state(node) for node in self.net.nodes]
        node_stats = list(chain(*node_stats))

        # encode request level statistics:
        # (1) encode resources bound to partial embedding; i.e. resources that will be released by REJECT_ACTION
        vnum = len(self.vtype_bidict.mirror[self.request])
        placed = self.vtype_bidict.mirror[self.request]
        crelease, mrelease = 0.0, 0.0
        for node, vtype in placed:
            config = self.vnfs[vtype]
            supplied_rate = sum(
                [service.datarate for service in self.vtype_bidict[(node, vtype)]])
            prev_cdem, prev_mdem = self.score(supplied_rate, config)
            after_cdem, after_mdem = self.score(
                supplied_rate - self.request.datarate, config)
            crelease += prev_cdem - after_cdem
            mrelease += prev_mdem - after_mdem

        crelease /= self.MAX_SERVICE_LEN * self.MAX_COMPUTE
        mrelease /= self.MAX_SERVICE_LEN * self.MAX_MEMORY

        # (2) one-hot encoding of requested service type
        # 请求的服务类型的一个热编码,1*4
        stype = [1.0 if service ==
                        self.request.service else 0.0 for service in self.services]

        # (3) count encoding of to-be-deployed VNFs for requested service
        # 对于这个服务需要新增的VNF实例个数,确定了vnum是用来在request里面遍历的
        counter = Counter(self.request.vtypes[vnum:])
        vnf_counts = [counter[vnf] /
                      self.MAX_SERVICE_LEN for vnf in range(len(self.vnfs))]

        # (4) noramlized datarate demanded by service request
        # 服务请求所需的非标准化数据速率
        datarate = self.request.datarate / self.MAX_LINKRATE

        # (5) residual end-to-end latency
        # 当前剩余端到端时延
        resd_lat = self.request.resd_lat / self.PROPAGATION_DIAMETER

        # (6) one-hot encoding of request's egress node
        egress_enc = [1.0 if node ==
                             self.request.egress else 0.0 for node in self.net.nodes]

        service_stats = [crelease, mrelease, *stype, *
        vnf_counts, datarate, resd_lat, *egress_enc]

        # encode graph level statistics:
        # (1) number of deployed instances for each type of VNF
        num_deployed = [sum([float((node, vtype) in self.vtype_bidict)
                             for node in self.net.nodes]) for vtype in range(len(self.vnfs))]
        num_deployed = [count / len(self.net.nodes) for count in num_deployed]

        # (2) mean compute, memory & link utilization
        mean_cutil = np.mean(
            [self.computing[node] / self.MAX_COMPUTE for node in self.net.nodes])
        mean_mutil = np.mean(
            [self.memory[node] / self.MAX_MEMORY for node in self.net.nodes])
        mean_lutil = np.mean([self.datarate[frozenset(
            {src, trg})] / self.MAX_LINKRATE for src, trg in self.net.edges])

        graph_stats = [*num_deployed, mean_cutil, mean_mutil, mean_lutil]
        # ServiceCoordination只有一个服务一个一个一个！！！！
        return np.asarray(list(chain(node_stats, service_stats, graph_stats)))

    # 想啥呢，action当然是agent返回来的，你这个deployment都有了，玩个锤子agent
    def step(self, action):
        rejected = (action == self.REJECT_ACTION)
        # reset tracked information for prior request when `action` deploys the next service's initial component
        # 当“操作”部署下一个服务的初始组件时，重置先前请求的跟踪信息,这里可能跟接入控制模块有关
        if not self.request in self.vtype_bidict.mirror:
            self.occupied = {'compute': 0.0, 'memory': 0.0, 'datarate': 0.0}
            self.admission = {'deployed': False, 'finalized': False}

        # check whether action is valid; terminate episode otherwise
        # 检查行动是否有效；否则终止,done代表放置一个节点是可以被接受的状态.一旦每一步被拒绝，那么这个service总的reject个数加1
        if not (action in self.valid_routes or rejected):
            # do not allow invalid actions for planning agents
            if self.planning_mode:
                self.logger.critical('Invalid action in planning mode.')
                raise RuntimeError('Invalid action taken.')

            # terminate the episode for model-free agents
            self.done = True
            self.info[self.request.service].num_invalid += 1
            self.logger.debug('Invalid action.')
            return self.compute_state(), 0.0, self.done, {}

        # action voluntarily releases partial service embedding; progress to next service
        # 为了下一次服务主动释放部分服务嵌入；
        elif rejected:
            self.info[self.request.service].num_rejects += 1
            self.logger.debug('Service embedding rejected.')
            self.admission = {'deployed': False, 'finalized': True}

            # release service; progress in time to next service; (implicit action cache update)
            self.release(self.request)
            reward = self.compute_reward(True, False, self.request)
            self.done = self.progress_time2()
            return self.compute_state(), reward, self.done, {}

        # the given action extends the partial embedding by another VNF placement
        # 给定的操作通过另一个VNF放置扩展了部分嵌入
        final_placement = self.place_vnf(action)
        # case: placement of final VNF; check whether valid route to egress node exists
        # 案例：最终VNF的放置；检查到出口节点的有效路由是否存在(前面place_vnf检查了VNF是否放置完毕,这一步是最后一个VNF到终点)
        if final_placement:
            try:
                # compute valid shortest path (latency) routing towards egress
                # NOTE: throws error if not reachable or shortest path has (cumulated) weight > cutoff
                _, route = nx.single_source_dijkstra(
                    self.net, source=action, target=self.request.egress, weight=self.get_weights,
                    cutoff=self.request.resd_lat)
                # case: a valid route to the service's egress node exists; service deployment successful
                # update network state, i.e. steer traffic towards egress
                route = ServiceCoordination.get_edges(route)
                self.steer_traffic(route)

                # register successful service embedding for deletion after duration passed
                exit_time = self.request.arrival + self.request.duration
                # 服务成功，进行压栈操作
                heapq.heappush(self.deployed, (exit_time, self.request))

                # update meta-information for deployed service before progressing in time
                self.update_info()
                self.logger.debug('Service deployed successfully.')
                self.admission = {'deployed': True, 'finalized': True}

                # progress in time after successful deployment; (implicit action cache update)
                # 操作成功获得收益
                reward = self.compute_reward(True, True, self.request)
                self.done = self.progress_time2()
                return self.compute_state(), reward, self.done, {}
            except (nx.NetworkXNoPath) as e:
                # case: no valid route to the service's egress node exists
                # 前面都放好了，没有到出口的路径
                self.info[self.request.service].no_egress_route += 1
                self.logger.debug('No valid route to egress remains.')
                self.admission = {'deployed': False, 'finalized': True}

                # release service; progress in time to next service; (implicit action cache update)
                # 隐式操作缓存更新
                self.release(self.request)
                reward = self.compute_reward(True, False, self.request)
                self.done = self.progress_time2()
                return self.compute_state(), reward, self.done, {}

        # case: partial embedding not finalized; at least one VNF placement remains to be determined
        reward = self.compute_reward(False, False, self.request)
        self.update_actions()

        # case: partial embedding cannot be extended farther after placement action; proceed to next service
        if not self.valid_routes:
            self.info[self.request.service].no_extension += 1
            self.logger.debug('Cannot extend partial embedding farther.')
            self.admission = {'deployed': False, 'finalized': True}

            # progress in time after the previous service was released; (implicit action cache update)
            self.release(self.request)
            self.done = self.progress_time2()
            return self.compute_state(), reward, self.done, {}

        # case: valid actions remain after extension of partial embedding
        self.logger.debug('Proceed with extension of partial embedding.')
        return self.compute_state(), reward, self.done, {}

    def reset(self) -> np.ndarray:
        '''Reset environment after episode is finished.'''
        # load graph from path, i.e. reset resource utilizations
        # trace是流量请求的集合,在做simulation时，带过来的是traffic对象，是没有经过sample的吧,实际在peekable的时候就做了sample
        self.observation_time = int(0)
        self.num_requests = 0
        # 用于捕获上一个时间窗口的时间窗口，下一个时间窗口送到部署模块进行部署
        self.requests_old_time = []
        # 用于捕获下一个时间窗口的请求
        self.requests_new_time = []
        self.trace = peekable(iter(self.process))
        #收集0-5s所有请求
        self.t_window()
        #手动生成了第一次请求
        self.request = self.requests_old_time[0]
        # set service & VNF properties of request upon arrival
        # reset environment's progress parameters
        # 全局是否终结,这个逻辑由progress_time2给
        self.done = False

        KEYS = ['accepts', 'requests', 'skipped_on_arrival', 'no_egress_route', 'no_extension', 'num_rejects',
                'num_invalid', 'cum_service_length', 'cum_route_hops', 'cum_compute',
                'cum_memory', 'cum_datarate', 'cum_max_latency', 'cum_resd_latency']
        self.info = [munchify(dict.fromkeys(KEYS, 0.0))
                     for _ in range(len(self.services))]
        self.info[self.request.service].requests += 1

        # reset resource utilization parameters,读取每个节点的属性信息
        self.computing = {node: data['compute']
                          for node, data in self.net.nodes(data=True)}
        self.memory = {node: data['memory']
                       for node, data in self.net.nodes(data=True)}
        self.datarate = {frozenset(
            {src, trg}): data['datarate'] for src, trg, data in self.net.edges(data=True)}

        # save propagation delays for each edge & processing delay per node
        self.propagation = {frozenset(
            {src, trg}): data['propagation'] for src, trg, data in self.net.edges(data=True)}

        # reset datastructures that tracks currently deployed services; tracks valid placements & their routes
        # 重置跟踪当前部署的服务对象集合；跟踪有效的位置及其路线
        self.deployed, self.valid_routes = [], {}
        # bidirectional mapping among (node, vtype) and services
        # (1) the dictionary maps (node, vtype) to sets of services that the VNF `vtype` serves on server `node`
        # -> dict is used to check whether an instance still serves any request or can be released otherwise
        # (2) the mirrored dict maps services to lists of (node, vtype), i.e. to its virtual node embeddings
        # -> used to determine what instances serve the service; when a service is terminated, all instances
        # serving it are shrinked or terminated entirely
        # (node,vtype)和服务的映射关系,正向映射代表是否实例还在被服务使用
        # 反向映射代表服务所应用的实例,当服务终止时，所有实例将被移除
        self.vtype_bidict = BiDict(None, val_btype=list)
        self.vtype_bidict = BiDict(self.vtype_bidict, val_btype=list)

        # bidirectional mapping among edges services and their routes
        # (1) the dictionary maps services to routes, i.e. to a list of links [(src, trg), ...]
        # -> dict is used to track virtual link embeddings per service
        # 正向映射代表服务和虚拟边映射的关系
        # (2) the mirrored dict maps set{(src, trg)} -> set{services steered across edge}
        # 反向映射代表某物理边所承载的所有服务集合
        # -> mirrored dict is used to track what services are steered across an edge (to obtain their max. latency)
        self.routes_bidict = BiDict(
            None, val_btype=list, key_map=lambda key: frozenset(key))
        self.routes_bidict = BiDict(self.routes_bidict, val_btype=list)

        # register ingress of first service request & compute initially valid actions
        # 注册第一个服务请求的入口并计算初始有效操作
        self.routes_bidict[self.request] = (None, self.request.ingress)
        self.update_actions()
        # check (edge case) whether first generated request is invalid
        if not self.valid_routes:
            # progress_time逻辑被重构成窗口内是否还有请求未尝试部署
            self.progress_time2()
        return self.compute_state()
    #compte the reward according to the service type,this is chaned by me,it may be changed
    def compute_reward(self, finalized: bool, deployed: bool, req: Request) -> float:
        '''Reward agents upon the acceptance of requested services.'''
        # 部署成功，奖励1.0
        if deployed:
            if req.service == 1:
                return 2.0
            elif req.service == 2:
                return 3.0
            elif req.service == 3:
                return 4.0
            else:
                return 1.0

        return 0.0

    # 时间窗口主逻辑,观测当前窗口所有的请求,这个方法只收集请求，不负责移除请求
    def t_window(self):
        self.requests_old_time=self.requests_new_time
        self.requests_new_time=[]
        while True:
            try:
                self.request = next(self.trace)
                # 捕获到当前时间窗口所有请求
                if self.request.arrival > self.observation_time and self.request.arrival < self.observation_time + 2:
                    self.requests_old_time.append(self.request)
                    self.request.resd_lat = self.request.max_latency
                    self.request.vtypes = self.services[self.request.service]
                    self.num_requests += 1
                # 下一个时间窗口的第一个请求
                else:
                    self.requests_new_time.append(self.request)
                    self.num_requests += 1
                    break
            except StopIteration:
                # 遇到StopIteration就退出循环,process里面没有东西了，取不到请求了哦
                self.done=True
                break

    # 更新时间窗口
    def update_t_window(self):
        if self.observation_time > 43:
            print("something is error")
        self.observation_time = self.observation_time + 2
        self.t_window()

    def network_temperature(self):
        compute_state = self.compute_state()
    #基于滑动时间窗口的部署方案
    #用于判断一个时间窗口的宏观逻辑，窗口内的逻辑由progress_time给
    def progress_time2(self) -> bool:
        '''Proceed in time to the succeeding service request, update the network accordingly.'''
        # 及时处理后续服务请求，相应地更新网络。
        # progress until the episode ends or an action must be taken
        while self.observation_time < 43:
            # determine resource demands of request upon their initial arrival
            #获取要部署的第一个请求
            if self.requests_old_time[-1]==self.request:
                self.update_t_window()
                if self.requests_old_time!=[]:
                    self.request=self.requests_old_time[0]
                else:
                    return True
                self.routes_bidict[self.request] = (None, self.request.ingress)
                self.info[self.request.service].requests += 1

                # set requested VNFs upon arrival of request
                self.request.resd_lat = self.request.max_latency
                self.request.vtypes = self.services[self.request.service]

                # update progress parameters of environment
            elif self.requests_old_time.index(self.request)!=-1:
                current_index=self.requests_old_time.index(self.request)
                #更新了self.request
                self.request=self.requests_old_time[current_index+1]
                self.routes_bidict[self.request] = (None, self.request.ingress)
                self.info[self.request.service].requests += 1

                # set requested VNFs upon arrival of request
                self.request.resd_lat = self.request.max_latency
                self.request.vtypes = self.services[self.request.service]

                # update progress parameters of environment
            while self.deployed and self.deployed[0][0] < self.observation_time:
                # 数组进行弹出操作
                rel_time, service = heapq.heappop(self.deployed)
                self.release(service)
            # stop progressing in time if valid actions exist for the arriving request
            # 如果对到达的请求存在有效操作，请及时停止进行
            self.update_actions()
            # 有valid_routes还有救,返回一个false，给你个机会去部署，true，你没机会了，直接pass
            if self.valid_routes:
                return False
            # 刚到达就跳过
            self.info[self.request.service].skipped_on_arrival += 1
        # episode is done when the trace simulation is complete
        return True

    def release(self, req: Request) -> None:
        '''Release network resources bound to the request.'''
        # case: to-be-deployed request is rejected upon its arrival
        # mirror key是请求对象,没有反之，后面那个二元组，代表（node，vtype）,反过来也是一样的.正向的时候key可以不一样，但是value可以一样
        if req not in self.vtype_bidict.mirror:
            return

        # release compute & memory resources at nodes with VNF instances that serve the request
        serving_vnfs = Counter(self.vtype_bidict.mirror[req])
        for (node, vtype), count in serving_vnfs.items():
            config = self.vnfs[vtype]

            # NOTE: account for sharing of VNFs when computing the updated rates by the counter
            # 通过计数器计算更新的费率时，考虑VNF的共享
            supplied_rate = sum(
                [req.datarate for req in self.vtype_bidict[(node, vtype)]])
            updated_rate = supplied_rate - count * req.datarate

            # score resources by difference in scores after datarate has been released
            prev_cdem, prev_mdem = self.score(supplied_rate, config)
            after_cdem, after_mdem = self.score(updated_rate, config)

            self.computing[node] += prev_cdem - after_cdem
            self.memory[node] += prev_mdem - after_mdem

        # remove to-be-released request from mapping
        del self.vtype_bidict.mirror[req]

        # release datarate along routing path and update datastructure
        route = self.routes_bidict.pop(req, [])
        for src, trg in route[1:]:
            self.datarate[frozenset({src, trg})] += req.datarate

    # 每一对VNF之间都选了最短路径.route代表VNF之间的虚拟链接映射到的物理链接
    def steer_traffic(self, route: List) -> None:
        '''Steer traffic from node-to-node across the given route.'''
        # route是经过的所有边，在这里，就是具体的部署了，上面已经规划好了
        for (src, trg) in route:
            # update residual datarate & latency that remains after steering action
            self.datarate[frozenset({src, trg})] -= self.request.datarate
            self.request.resd_lat -= self.propagation[frozenset({src, trg})]

            # register link to routing (link embeddings) of `self.request`
            self.routes_bidict[self.request] = (src, trg)
        # track increase of resources occupied by the service deployment
        datarate = len(route) * self.request.datarate
        occupied = {'compute': 0.0, 'memory': 0.0, 'datarate': datarate}
        self.occupied = {key: self.occupied[key] + occupied[key] for key in occupied}

    def compute_resources(self, node: int, vtype: int) -> Tuple[int]:
        '''Calculate increased resource requirements when placing a VNF of type `vtype` on `node`.'''
        # calculate the datarate served by VNF `vtype` before scheduling the current flow to it
        # 在将当前流调度到VNF`vtype`之前，计算它所服务的数据速率,先判断
        config = self.vnfs[vtype]
        supplied_rate = sum([service.datarate for service in self.vtype_bidict[(node, vtype)]])

        # calculate the resource requirements before and after the scaling
        # 计算缩放前后的资源需求
        before = self.score(supplied_rate, config)
        after = self.score(supplied_rate + self.request.datarate, config)
        compute, memory = np.subtract(after, before)

        return compute, memory

    # 放置VNF
    def place_vnf(self, node: int) -> bool:
        '''Deploys the to-be-placed VNF on `node` and establishes its connection to the service.'''
        # 将要放置的VNF部署到“node”上，并建立其与服务的连接。
        vnum = len(self.vtype_bidict.mirror[self.request])
        vtype = self.request.vtypes[vnum]

        # update provisioned compute and memory resources
        # 更新配置的计算和内存资源
        compute, memory = self.compute_resources(node, vtype)
        self.computing[node] -= compute
        self.memory[node] -= memory

        # track increase of resources occupied by the service deployment
        occupied = {'compute': compute, 'memory': memory, 'datarate': 0.0}
        self.occupied = {key: self.occupied[key] + occupied[key] for key in occupied}

        # steer traffic across shortest path (in terms of latency) route
        route = self.valid_routes[node]
        self.steer_traffic(route)

        # update data structure that track relations among VNFs, services and nodes
        # 更新跟踪VNF、服务和节点之间关系的数据结构
        self.vtype_bidict[(node, vtype)] = self.request
        # the service is completely deployed; register demanded resources for deletion after duration is exceeded
        if len(self.vtype_bidict.mirror[self.request]) >= len(self.request.vtypes):
            return True

        return False

    def get_weights(self, u: int, v: int, d: Dict) -> float:
        '''Link (propagation) delay invoked when steering traffic across the edge.'''
        # 引导流量时引发的传播延迟
        # check if it does not provision sufficient datarate resources
        if self.datarate[frozenset({u, v})] < self.request.datarate:
            return None

        # compute propagation & queuing delay based on link utilization & requested datarate
        delay = self.propagation[frozenset({u, v})]
        return delay

    def update_actions(self) -> None:
        '''Update the set of valid placement actions and their respective routings.'''
        # return if simulation episode is already done
        if self.done:
            return

        # compute latencies by shortest path of propagation delays across amenable edges
        # 通过可修正边上传播延迟的最短路径计算延迟,这里的source不是指的你的那个源，这个source是在部署情况下的起点值
        _, source = self.routes_bidict[self.request][-1]
        # routes代表当前服务源节点到其他节点的最短路径,请注意，只从源节点出发.选取最短路径
        lengths, routes = nx.single_source_dijkstra(
            self.net, source=source, weight=self.get_weights, cutoff=self.request.resd_lat)

        # filter routes to deployment nodes where the routing delay exceeds the maximum end-to-end latency of the request
        # 筛选路由到路由延迟超过请求最大端到端延迟的部署节点,s哦，我选取了可达的所有最短路径.我在选下一跳，下一个VNF放在哪
        routes = {node: route for node, route in routes.items() if lengths[node] <= self.request.resd_lat}
        # check whether reachable nodes also provision enough resources for the deployment
        # 检查可访问节点是否也为部署提供了足够的资源,考虑转发路由的存在.mirror用来做反向逻辑
        vnum = len(self.vtype_bidict.mirror[self.request])
        vtype = self.request.vtypes[vnum]
        cdemands, mdemands = {}, {}

        for node in routes:
            compute, memory = self.compute_resources(node, vtype)
            cdemands[node] = compute
            mdemands[node] = memory

        # valid nodes must provision enough compute and memory resources for the deployment
        valid_nodes = [node for node in routes if cdemands[node] <=
                       self.computing[node] and mdemands[node] <= self.memory[node]]

        # cache valid routes for the upcoming time step
        # 缓存即将到来的时间区间的有效路由,route5: [11, 1, 5].valid_routes:5: [(11, 1), (1, 5)]
        self.valid_routes = {node: ServiceCoordination.get_edges(route) for node,
                                                                            route in routes.items() if
                             node in valid_nodes}

    def replace_process(self, process):
        '''Replace traffic process used to generate request traces.'''
        # 替换用于生成请求跟踪的流量过程
        self.process = process
        self.reset()

    def render(self, mode: str = 'None', close: bool = False) -> str:
        if mode == 'None' or self.done:
            return

        elif mode == 'textual':
            rtable = [['Compute'] + [str(round(c, 2))
                                     for _, c in self.computing.items()]]
            rtable += [['Memory'] + [str(round(m, 2))
                                     for _, m in self.memory.items()]]

            tnodes = [{t for t in range(len(self.vnfs)) if (
                n, t) in self.vtype_bidict} for n in self.net.nodes()]
            rtable += [['Type'] + tnodes]

            services = [len([num for num, service in enumerate(self.vtype_bidict.mirror) for t in range(
                len(self.vnfs)) if (n, t) in self.vtype_bidict.mirror[service]]) for n in self.net.nodes()]
            rtable += [['Service'] + services]
            cheaders = ['Property'] + [str(node)
                                       for node, _ in self.computing.items()]
            ctable = tabulate(rtable, headers=cheaders, tablefmt='github')
            cutil = 1 - np.mean([self.computing[n] / self.net.nodes[n]
            ['compute'] for n in self.net.nodes])
            mutil = 1 - \
                    np.mean([self.memory[n] / self.net.nodes[n]['memory']
                             for n in self.net.nodes])

            max_cap = [self.net.edges[e]['datarate'] for e in self.net.edges]
            cap = [self.datarate[frozenset({*e})] for e in self.net.edges]
            dutil = np.asarray(cap) / np.asarray(max_cap)
            dutil = 1 - np.mean(dutil)
            graph_stats = f'Util (C): {cutil}; Util (M): {mutil}; Util (D): {dutil}'

            vnum = len(self.vtype_bidict.mirror[self.request])
            vtype = self.request.vtypes[vnum]
            str_repr = '\n'.join(
                (ctable, f'Time: {self.time}', f'Request: {str(self.request)}->{vtype}',
                 f'Available Routes: {self.valid_routes}', f'Graph Stats: {graph_stats}' '\n\n'))
            return str_repr

        if self.pos is None:
            self.pos = nx.spring_layout(self.net, iterations=400)

        # Render the environment to the screen,edge是物理边
        edges = {edge: frozenset({*edge}) for edge in self.net.edges}
        datarate = {edge: self.datarate[edges[edge]]
                    for edge in self.net.edges}
        propagation = {
            edge: self.propagation[edges[edge]] for edge in self.net.edges}

        def link_rate(edge):
            return round(datarate[edge], 2)

        def delay(edge):
            return round(propagation[edge], 2)

        _, service_pos = self.routes_bidict[self.request][-1]
        valid_placements = self.valid_routes.keys()

        def color(node):
            if node in valid_placements and node == service_pos:
                return 'cornflowerblue'
            elif node in valid_placements:
                return 'seagreen'
            elif node == service_pos:
                return 'red'
            return 'slategrey'

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 8))
        color_map = [color(node) for node in self.net.nodes]
        edge_labels = {edge: (link_rate(edge), delay(edge))
                       for edge in self.net.edges}
        node_labels = {node: (node, round(self.computing[node], 2), round(
            self.memory[node], 2)) for node in self.net.nodes()}
        nx.draw(self.net, self.pos, labels=node_labels, node_color=color_map,
                with_labels=True, node_size=400, ax=ax)
        nx.draw_networkx_edge_labels(
            self.net, self.pos, edge_labels=edge_labels, ax=ax)

    def update_info(self) -> None:
        service = self.request.service
        self.info[service].accepts += 1
        self.info[service].cum_service_length += len(self.request.vtypes)
        self.info[service].cum_route_hops += len(
            self.routes_bidict[self.request])
        self.info[service].cum_datarate += self.request.datarate
        self.info[service].cum_max_latency += self.request.max_latency
        self.info[service].cum_resd_latency += self.request.resd_lat

    @staticmethod
    # 如果stop为None，则一直迭代到最后位置。islice原型（可迭代对象，start，stop）
    def get_edges(nodes: List) -> List:
        return list(zip(islice(nodes, 0, None), islice(nodes, 1, None)))

    @staticmethod
    def score(rate, config):
        '''Score the CPU and memory resource consumption for a given VNF configuration and requested datarate.'''
        # 为给定VNF配置和请求的数据速率的CPU和内存资源消耗评分。
        # set VNF resource consumption to zero whenever their requested rate is zero
        if rate <= 0.0:
            return (0.0, 0.0)
        # VNF有服务的最大传输速率
        # VNFs cannot serve more than their max. transfer rate (in MB/s)
        elif rate > config['max. req_transf_rate']:
            return (np.inf, np.inf)

        rate = rate / config['scale']

        # score VNF resources by polynomial fit(按多项式拟合)
        # VNF demand
        compute = config['coff'] + config['ccoef_1'] * rate + config['ccoef_2'] * \
                  (rate ** 2) + config['ccoef_3'] * \
                  (rate ** 3) + config['ccoef_4'] * (rate ** 4)
        memory = config['moff'] + config['mcoef_1'] * rate + config['mcoef_2'] * \
                 (rate ** 2) + config['mcoef_3'] * \
                 (rate ** 3) + config['mcoef_3'] * (rate ** 4)

        return (max(0.0, compute), max(0.0, memory))