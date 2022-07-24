import heapq
import logging
from collections import Counter
from itertools import chain, islice
from typing import List, Dict, Tuple

import gym
import networkx as nx
import numpy as np
from gym import spaces
# from more_itertools import peekable
from tabulate import tabulate
from munch import munchify
from coordination.environment.bidict import BiDict
from coordination.environment.traffic import Traffic, Request, TrafficStub


# 部署方案好像要改


class ServiceCoordination5(gym.Env):
    # substrate network traffic request vnfs services
    def __init__(self, net_path: str, process: Traffic, vnfs: List, services: List):
        self.net_path = net_path
        self.net = nx.read_gpickle(self.net_path)
        self.NUM_NODES = self.net.number_of_nodes()
        self.MAX_DEGREE = max([deg for _, deg in self.net.degree()])
        self.REJECT_ACTION = self.NUM_NODES + 1
        # GLOBAL attribute
        self.MAX_COMPUTE = self.net.graph['MAX_COMPUTE']
        self.MAX_LINKRATE = self.net.graph['MAX_LINKRATE']  # in MB/s
        self.MAX_MEMORY = self.net.graph['MAX_MEMORY']  # in MB
        self.HOPS_DIAMETER = self.net.graph['HOPS_DIAMETER']  # in ms
        self.PROPAGATION_DIAMETER = self.net.graph['PROPAGATION_DIAMETER']  # in ms
        # discrete event generator
        self.process: Traffic = process
        self.PROCESS_STUB = TrafficStub(self.process.sample())
        # 8 queue in a list
        self.all_waiting_queues = []
        self.vnfs: List[dict] = vnfs
        self.services: List[List[int]] = services
        self.NUM_SERVICES = len(self.services)
        self.MAX_SERVICE_LEN = max([len(service) for service in self.services])
        # current source
        self.occupied = {'compute': 0.0, 'memory': 0.0, 'datarate': 0.0}
        # 强行截断
        self.MAX_QUEUE_LENGTH = 200
        # admission_control
        self.admission = {'deployed': False, 'finalized': False}
        # 我选队列来部署，不决定是否拒绝
        self.ACTION_DIM = 8
        # use 8 discrete action
        self.action_space = spaces.Discrete(self.ACTION_DIM)
        # state_space:substrate_network(cpu,mem,network)*Queue(等待队列程度)*成功率(当前成功率)
        self.NETWORK_SIZE = 3
        # 新到请求，被拒绝请求
        self.QUEUE_SIZE = 8
        self.num_requests = 0
        # 4类业务
        self.SUCCUESS_SIZE = 4
        self.update_interval = 0
        # TODO:加入网络温度
        self.OBS_SIZE = self.NETWORK_SIZE + self.QUEUE_SIZE + self.SUCCUESS_SIZE
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float16)
        # setup basic debug logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())
        # setup layout for rendering functionality; setup step info
        self.pos = None
        self.info = None
        # it is used to save the all the new-arrival requests
        self.request_list = []
        # it is used to save the all the new-arrival requests
        self.reject_list = []
        self.reward = 0
        # current time window
        self.start_time = 0
        self.TIME_WINDOW_LENGTH = 5
        self.end_time = self.start_time + self.TIME_WINDOW_LENGTH
        self.TIME_HORIZON = 50
        self.valid_actions={}
        # 重置环境
        self.reset()

    def time_window(self, start_time, end_time):
        for request in self.PROCESS_STUB:
            # THE NEW ATTIVAL REQUESTS WILL BE INITIALIZED IN ARRRIVAL TIME.HOWEVER WHEN REJECTED,SOME PROPERTIES WILL BE REMOVED
            #WE WILL USE initial_one_REJECT_request FUMCTION TO REINITIAL IT
            if request.arrival > start_time and request.arrival < end_time:
                self.request_list.append(request)
                self.initial_one_request(request)

    def reset(self):
        # 离散事件发生器
        # self.trace = peekable(iter(self.process))
        self.queue_new_arrival_memory = []
        self.queue_new_arrival_cpu = []
        self.queue_new_arrival_datarate = []
        self.queue_new_arrival_latency = []
        self.queue_reject_memory = []
        self.queue_reject_cpu = []
        self.queue_reject_datarate = []
        self.queue_reject_latency = []
        self.done = False
        self.time = 5
        self.update_interval = 0
        self.request_list = []
        self.reject_list=[]
        self.start_time = 0
        self.end_time = self.start_time + self.TIME_WINDOW_LENGTH
        self.all_waiting_queues=[]
        # record the datarate in every instance of every service_type in everynode!
        self.rate_list = []
        for i in range(self.net.number_of_nodes()):
            self.rate_list.append([])
            # vtype 6 service_rate 4,node 50
            for vtype in range(6):
                self.rate_list[i].append([0.0, 0.0, 0.0, 0.0])
        # 这个就是给monitor模块用的，step用于返回信息。最后的info
        KEYS = ['accepts', 'requests', 'skipped_on_arrival', 'no_egress_route', 'no_extension', 'num_rejects',
                'num_invalid', 'cum_service_length', 'cum_route_hops', 'cum_compute',
                'cum_memory', 'cum_datarate', 'cum_max_latency', 'cum_resd_latency']
        self.info = [munchify(dict.fromkeys(KEYS, 0.0))
                     for _ in range(len(self.services))]

        # 重置底层网络
        self.computing = {node: data['compute']
                          for node, data in self.net.nodes(data=True)}
        self.memory = {node: data['memory']
                       for node, data in self.net.nodes(data=True)}
        self.datarate = {frozenset(
            {src, trg}): data['datarate'] for src, trg, data in self.net.edges(data=True)}

        # save propagation delays for each edge & processing delay per node
        self.propagation = {frozenset(
            {src, trg}): data['propagation'] for src, trg, data in self.net.edges(data=True)}

        # 请求的到达时间和请求的对象实例
        self.deployed, self.valid_routes = [], {}
        # 记录当前部署的节点情况
        self.vtype_bidict = BiDict(None, val_btype=list)
        self.vtype_bidict = BiDict(self.vtype_bidict, val_btype=list)
        # 记录请求和路径节点的映射
        self.routes_bidict = BiDict(None, val_btype=list, key_map=lambda key: frozenset(key))
        self.routes_bidict = BiDict(self.routes_bidict, val_btype=list)
        # capture the first time_window requests
        self.time_window(self.start_time, self.end_time)
        self.combine_request_list()
        self.valid_actions=self.valid_actions_method()
        return self.compute_state_admission()

    # this function should return the next_placement_state and the placement result
    def place_single_vnf(self, action):
        rejected = (action == self.REJECT_ACTION)
        # reset tracked information for prior request when `action` deploys the next service's initial component
        if not self.request in self.vtype_bidict.mirror:
            self.occupied = {'compute': 0.0, 'memory': 0.0, 'datarate': 0.0}
            self.admission = {'deployed': False, 'finalized': False}

        # check whether action is valid; terminate episode otherwise
        if not (action in self.valid_routes or rejected):
            # do not allow invalid actions for planning agents

            # terminate the episode for model-free agents
            # only means this request has finished
            # an invalid action we should reject it immediately the result should be false
            self.info[self.request.service].num_invalid += 1
            self.logger.debug('Invalid action.')
            reward_step = 0
            result = False
            return [], reward_step, result

        # action voluntarily releases partial service embedding; progress to next service
        #
        elif rejected:
            self.info[self.request.service].num_rejects += 1
            self.logger.debug('Service embedding rejected.')
            self.admission = {'deployed': False, 'finalized': True}

            # release service; progress in time to next service; (implicit action cache update)
            self.release(self.request)
            # self.done = self.progress_time()
            reward_step = 0
            result = False
            return [], reward_step, result

        # the given action extends the partial embedding by another VNF placement
        final_placement = self.place_vnf(action)

        # case: placement of final VNF; check whether valid route to egress node exists
        if final_placement:
            try:
                # compute valid shortest path (latency) routing towards egress
                # NOTE: throws error if not reachable or shortest path has (cumulated) weight > cutoff
                _, route = nx.single_source_dijkstra(
                    self.net, source=action, target=self.request.egress, weight=self.get_weights,
                    cutoff=self.request.resd_lat)
                # case: a valid route to the service's egress node exists; service deployment successful
                # update network state, i.e. steer traffic towards egress
                route = ServiceCoordination5.get_edges(route)
                self.steer_traffic(route)

                # register successful service embedding for deletion after duration passed
                exit_time = self.request.arrival + self.request.duration
                heapq.heappush(self.deployed, (exit_time, self.request))

                # update meta-information for deployed service before progressing in time
                self.update_info()
                self.logger.debug('Service deployed successfully.')
                self.admission = {'deployed': True, 'finalized': True}

                # progress in time after successful deployment; (implicit action cache update)
                reward_step = self.compute_reward(True, True, self.request)
                # self.done = self.progress_time()
                result = True
                return [], reward_step, result

            except nx.NetworkXNoPath:
                # case: no valid route to the service's egress node exists
                self.info[self.request.service].no_egress_route += 1
                self.logger.debug('No valid route to egress remains.')
                self.admission = {'deployed': False, 'finalized': True}

                # release service; progress in time to next service; (implicit action cache update)
                self.release(self.request)
                reward = self.compute_reward(True, False, self.request)
                # self.done = self.progress_time()
                result = False
                return [], reward, result

        # case: partial embedding not finalized; at least one VNF placement remains to be determined
        reward_step = self.compute_reward(False, False, self.request)
        self.update_actions()
        # case: partial embedding cannot be extended farther after placement action; proceed to next service
        if not self.valid_routes:
            self.info[self.request.service].no_extension += 1
            self.logger.debug('Cannot extend partial embedding farther.')
            self.admission = {'deployed': False, 'finalized': True}

            # progress in time after the previous service was released; (implicit action cache update)
            self.release(self.request)
            # self.done = self.progress_time()
            reward_step = self.compute_reward(False, False, self.request)
            result = False
            return [], reward_step, result

        # case: valid actions remain after extension of partial embedding
        result = True
        self.logger.debug('Proceed with extension of partial embedding.')
        return self.compute_state(), reward_step, result

    # 更换请求
    def replace_process(self, process):
        '''Replace traffic process used to generate request traces.'''
        self.process = process
        self.PROCESS_STUB = TrafficStub(self.process.sample())
        self.reset()

    def compute_reward(self, finalized: bool, deployed: bool, req: Request) -> float:
        '''Reward agents upon the acceptance of requested services.'''
        # todo: calculate the reward by the request,the basic is accrding to the text.it should be (profit),(utilazation),(fairness)
        if deployed:
            return 1.0

        return 0.0


    def release(self, req: Request) -> None:
        '''Release network resources bound to the request.'''
        # case: to-be-deployed request is rejected upon its arrival
        if req not in self.vtype_bidict.mirror:
            return

        # release compute & memory resources at nodes with VNF instances that serve the request
        # Counter({(46, 0): 1, (9, 2): 1, (12, 5): 1, (42, 4): 1})
        serving_vnfs = Counter(self.vtype_bidict.mirror[req])
        for (node, vtype), count in serving_vnfs.items():
            config = self.vnfs[vtype]
            # NOTE: account for sharing of VNFs when computing the updated rates by the counter
            # current supplied_date，当然是减，我们是在释放请求阿，亲！this is processing rate,however if a instance doesn't have any requests,it can shutdown
            # supplied_rate = sum(
            #     [req.datarate for req in self.vtype_bidict[(node, vtype)]])
            # updated_rate = supplied_rate - count * req.datarate

            # score resources by difference in scores after datarate has been released
            prev_cdem, prev_mdem = self.score(self.rate_list[node][vtype], config)
            for req in self.vtype_bidict[(node, vtype)]:
                self.rate_list[node][vtype][req.service] = self.rate_list[node][vtype][req.service] - count * req.datarate
            after_cdem, after_mdem = self.score(self.rate_list[node][vtype], config)
            # return the computing result
            self.computing[node] += prev_cdem - after_cdem
            # return the memory result
            self.memory[node] += prev_mdem - after_mdem

        # remove to-be-released request from mapping
        del self.vtype_bidict.mirror[req]
        # release datarate along routing path and update datastructure
        route = self.routes_bidict.pop(req, [])
        for src, trg in route[1:]:
            self.datarate[frozenset({src, trg})] += req.datarate

    # 当某一个请求结束时，更新统计数据
    def update_info(self) -> None:
        service = self.request.service
        self.info[service].accepts += 1
        self.info[service].cum_service_length += len(self.request.vtypes)
        self.info[service].cum_route_hops += len(self.routes_bidict[self.request])
        self.info[service].cum_datarate += self.request.datarate
        self.info[service].cum_max_latency += self.request.max_latency
        self.info[service].cum_resd_latency += self.request.resd_lat

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

        # Render the environment to the screen
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
        nx.draw_networkx_edge_labels(self.net, self.pos, edge_labels=edge_labels, ax=ax)

    # 如果剩余带宽可以满足，那么返回传播时延
    def get_weights(self, u: int, v: int, d: Dict) -> float:
        '''Link (propagation) delay invoked when steering traffic across the edge.'''

        # check if it does not provision sufficient datarate resources
        if self.datarate[frozenset({u, v})] < self.request.datarate:
            return None

        # compute propagation & queuing delay based on link utilization & requested datarate
        delay = self.propagation[frozenset({u, v})]
        return delay

    def compute_resources(self, node: int, vtype: int) -> Tuple[int]:
        '''Calculate increased resource requirements when placing a VNF of type `vtype` on `node`.'''
        # calculate the datarate served by VNF `vtype` before scheduling the current flow to it
        config = self.vnfs[vtype]
        # supplied_rate = sum([service.datarate for service in self.vtype_bidict[(node, vtype)]])
        # calculate the resource requirements before and after the scaling
        # calculate the demand
        before = self.score(self.rate_list[node][vtype], config)
        # update_the_rate_list
        self.rate_list[node][vtype][self.request.service] += self.request.datarate
        # compute_the_demand after placement
        after = self.score(self.rate_list[node][vtype], config)
        compute, memory = np.subtract(after, before)

        return compute, memory

    # we should check if it is a valid placement_action,so we should compute_node_state,we use it as a function of placement
    def compute_node_state(self, node) -> np.ndarray:
        '''Define node level statistics for state representation.'''
        if not node in self.valid_routes:
            nstate = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
            return nstate

        # (1) node is a valid placement & valid route to node exists?
        valid = float(node in self.valid_routes)

        # (2) residual latency of request after placement on node
        route = self.valid_routes[node]
        latency = sum([self.propagation[frozenset({u, v})] for u, v in route])
        latency = np.nanmin([latency / self.request.resd_lat, 1.0])

        # (3) quantify datarate demands for placement with #hops to node
        hops = len(route) / self.HOPS_DIAMETER

        # (4) increase of allocated compute/memory after potential placement on node
        vnum = len(self.vtype_bidict.mirror[self.request])
        vtype = self.request.vtypes[vnum]
        config = self.vnfs[vtype]
        # supplied_rate = sum(
        #     [service.datarate for service in self.vtype_bidict[(node, vtype)]])
        prev_cdem, prev_mdem = self.score(self.rate_list[node][vtype], config)
        self.rate_list[node][vtype][self.request.service] = self.rate_list[node][vtype][self.request.service] + self.request.datarate
        after_cdem, after_mdem = self.score(self.rate_list[node][vtype], config)
        # calculate the cdemand
        cdemand = np.clip((after_cdem - prev_cdem) /
                          self.MAX_COMPUTE, a_min=0.0, a_max=1.0)
        # calculate the mdemand
        mdemand = np.clip((after_mdem - prev_mdem) /
                          self.MAX_MEMORY, a_min=0.0, a_max=1.0)

        # (5) residual compute/memory after placement on node
        resd_comp = np.clip(
            (self.computing[node] - cdemand) / self.MAX_COMPUTE, a_min=0.0, a_max=1.0)
        resd_mem = np.clip(
            (self.memory[node] - mdemand) / self.MAX_MEMORY, a_min=0.0, a_max=1.0)

        return [valid, latency, hops, cdemand, mdemand, resd_comp, resd_mem]

    # this is the satate used by the placement,however it is not used in the step function
    def compute_state(self) -> np.ndarray:
        '''Compute state representation of environment for RL agent.'''
        node_stats = [self.compute_node_state(node) for node in self.net.nodes]
        node_stats = list(chain(*node_stats))

        # encode request level statistics:
        # (1) encode resources bound to partial embedding; i.e. resources that will be released by REJECT_ACTION
        vnum = len(self.vtype_bidict.mirror[self.request])
        placed = self.vtype_bidict.mirror[self.request]
        crelease, mrelease = 0.0, 0.0
        for node, vtype in placed:
            config = self.vnfs[vtype]
            # supplied_rate = sum(
            #     [service.datarate for service in self.vtype_bidict[(node, vtype)]])
            prev_cdem, prev_mdem = self.score(self.rate_list[node][vtype], config)
            self.rate_list[node][vtype][self.request.service] -= self.request.datarate
            after_cdem, after_mdem = self.score(self.rate_list[node][vtype], config)
            crelease += prev_cdem - after_cdem
            mrelease += prev_mdem - after_mdem
        # normalize,crelease,mrelease
        crelease /= self.MAX_SERVICE_LEN * self.MAX_COMPUTE
        mrelease /= self.MAX_SERVICE_LEN * self.MAX_MEMORY

        # (2) one-hot encoding of requested service type
        stype = [1.0 if service ==
                        self.request.service else 0.0 for service in self.services]

        # (3) count encoding of to-be-deployed VNFs for requested service
        counter = Counter(self.request.vtypes[vnum:])
        vnf_counts = [counter[vnf] /
                      self.MAX_SERVICE_LEN for vnf in range(len(self.vnfs))]

        # (4) noramlized datarate demanded by service request
        datarate = self.request.datarate / self.MAX_LINKRATE

        # (5) residual end-to-end latency
        resd_lat = self.request.resd_lat / self.PROPAGATION_DIAMETER

        # (6) one-hot encoding of request's egress node
        egress_enc = [1.0 if node ==
                             self.request.egress else 0.0 for node in self.net.nodes]
        # vnf_counts represent the remaining vnf types
        service_stats = [crelease, mrelease, *stype, *vnf_counts, datarate, resd_lat, *egress_enc]
        # (1) number of deployed instances for each type of VNF
        num_deployed = [sum([float((node, vtype) in self.vtype_bidict)
                             for node in self.net.nodes]) for vtype in range(len(self.vnfs))]
        num_deployed = [count / len(self.net.nodes) for count in num_deployed]
        mean_cutil = np.mean(
            [self.computing[node] / self.MAX_COMPUTE for node in self.net.nodes])
        mean_mutil = np.mean(
            [self.memory[node] / self.MAX_MEMORY for node in self.net.nodes])
        mean_lutil = np.mean([self.datarate[frozenset(
            {src, trg})] / self.MAX_LINKRATE for src, trg in self.net.edges])

        graph_stats = [*num_deployed, mean_cutil, mean_mutil, mean_lutil]

        return np.asarray(list(chain(node_stats, service_stats, graph_stats)))

    @staticmethod
    def get_edges(nodes: List) -> List:
        return list(zip(islice(nodes, 0, None), islice(nodes, 1, None)))

    @staticmethod
    # rate_list is accoding to the special node and vtype
    def score(rate_list: list, config):
        '''Score the CPU and memory resource consumption for a given VNF configuration and requested datarate.'''
        # set VNF resource consumption to zero whenever their requested rate is zero
        # for rate in rate_list:
        #     if rate < 0.0:
        #         return (0.0, 0.0)
        #
        #     # VNFs cannot serve more than their max. transfer rate (in MB/s)
        #     elif rate > config['max. req_transf_rate']:
        #         return (np.inf, np.inf)
        # average rate only one!
        # for rate in rate_list:
        #     rate = rate / config['scale']

        # score VNF resources by polynomial fit,非线性关系
        compute = config['coff'] + config['ccoef_1'] * rate_list[0] + config['ccoef_2'] * \
                  (rate_list[1]) + config['ccoef_3'] * \
                  (rate_list[2]) + config['ccoef_4'] * (rate_list[3]) /config['scale']
        memory = config['moff'] + config['mcoef_1'] * rate_list[0] + config['mcoef_2'] * \
                 (rate_list[1]) + config['mcoef_3'] * \
                 (rate_list[2]) + config['mcoef_3'] * (rate_list[3])/config['scale']

        return (max(0.0, compute), max(0.0, memory))

    def place_request(self, request: Request) -> bool:
        # single step result,trying to place the vnf one step
        global result
        # if the request was rejected before the final placment
        for i in range(len(self.request.vtypes)):
            action = self.predict_random()
            # placement is done in one step
            state, step_reward, result = self.place_single_vnf(action)
            if not result:
                return state, step_reward, result
        # only step_reward is not None the result is True after all the placement
        return state, step_reward, result

    # place vnf randomly
    def predict_random(self, **kwargs):
        """Samples a valid action from all valid actions."""
        # sample process
        valid_nodes = np.asarray([node for node in self.valid_routes])
        return np.random.choice(valid_nodes)

    def place_vnf(self, node: int) -> bool:
        vnum = len(self.vtype_bidict.mirror[self.request])
        vtype = self.request.vtypes[vnum]

        # update provisioned compute and memory resources
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
        self.vtype_bidict[(node, vtype)] = self.request

        # the service is completely deployed; register demanded resources for deletion after duration is exceeded
        if len(self.vtype_bidict.mirror[self.request]) == len(self.request.vtypes):
            return True

        return False

    def steer_traffic(self, route: List) -> None:
        '''Steer traffic from node-to-node across the given route.'''

        for (src, trg) in route:
            # update residual datarate & latency that remains after steering action
            self.datarate[frozenset({src, trg})] -= self.request.datarate
            self.request.resd_lat -= self.propagation[frozenset({src, trg})]

            # register link to routing (link embeddings) of `self.request`
            self.routes_bidict[self.request] = (src, trg)

        # track increase of resources occupied by the service deployment
        datarate = len(route) * self.request.datarate
        occupied = {'compute': 0.0, 'memory': 0.0, 'datarate': datarate}
        # {'compute': 0.24598152161708728, 'memory': 768.0, 'datarate': 516.8745179973857}
        self.occupied = {key: self.occupied[key] + occupied[key] for key in occupied}

    def update_actions(self) -> None:
        '''Update the set of valid placement actions and their respective routings.'''
        # return if simulation episode is already done
        if self.done:
            return

        # compute latencies by shortest path of propagation delays across amenable edges
        # SOURCE:8
        _, source = self.routes_bidict[self.request][-1]
        # all candidate nodes
        # {8: 0, 11: 0.15430500052566462, 13: 0.3437491777902329, 2: 0.35707748853512133, 31: 0.4124103533704826, 37: 0.5030940551965568, 3: 0.5843501633478713, 25: 0.6353675928592062, 32: 0.6760395849780623, 49: 0.7085530236466248, 41: 0.7594958203022628, 18: 0.8549415707925507, 5: 0.871545738099399, 19: 0.8978817704237891, 34: 0.9212570930293987, 10: 1.0074296241720928, 22: 1.0196577590535458, 16: 1.0266557663712659, 43: 1.0297492850629224, 20: 1.0331334967709591, 40: 1.0457565811621277, 45: 1.0472349789538118, 44: 1.0492127092039008, 1: 1.0590295521644042, 14: 1.0854660739669235, 9: 1.093285762483712, 35: 1.1415510163095033, 12: 1.1603106123610458, 26: 1.1907463656011061, 24: 1.1983889665552838, 48: 1.203296226672668, 33: 1.2113384805366214, 28: 1.2218790047642565, 47: 1.233365330785172, 4: 1.2383230324686378, 29: 1.2506988426677474, 21: 1.2527392953289849, 39: 1.2580420645835757, 6: 1.2771533725283974, 23: 1.3057044419516144, 27: 1.348042876801162, 30: 1.3560566196352284, 38: 1.3872327479467736, 0: 1.3928588300638047, 7: 1.408466492933099, 42: 1.451759545593431, 46: 1.4634045165613112, 15: 1.513905458744372, 17: 1.5145350375183448, 36: 1.608286920926357}
        lengths, routes = nx.single_source_dijkstra(
            self.net, source=source, weight=self.get_weights, cutoff=self.request.resd_lat)
        # routes:11:[8,11] candidate paths and all the nodes passed through
        # filter routes to deployment nodes where the routing delay exceeds the maximum end-to-end latency of the request
        # 检查是否有时延未超标的节点,
        routes = {node: route for node, route in routes.items() if lengths[node] <= self.request.resd_lat}
        # check whether reachable nodes also provision enough resources for the deployment
        vnum = len(self.vtype_bidict.mirror[self.request])
        vtype = self.request.vtypes[vnum]
        cdemands, mdemands = {}, {}
        # the candidate node compute_resources
        for node in routes:
            compute, memory = self.compute_resources(node, vtype)
            cdemands[node] = compute
            mdemands[node] = memory

        # valid nodes must provision enough compute and memory resources for the deployment
        valid_nodes = [node for node in routes if cdemands[node] <=
                       self.computing[node] and mdemands[node] <= self.memory[node]]
        # get the candidate nodes
        for node in valid_nodes:
            try:
                _, check_route = nx.single_source_dijkstra(self.net, source=node, target=self.request.egress,
                                                           weight=self.get_weights, cutoff=self.request.resd_lat)
            except Exception as e:
                valid_nodes.pop(valid_nodes.index(node))
        # cache valid routes for the upcoming time step
        # check the valid routes
        # {8: [], 11: [(8, 11)], 13: [(8, 13)], 2: [(8, 2)], 3: [(8, 11), (11, 3)], 31: [(8, 11), (11, 31)], 25: [(8, 13), (13, 25)], 49: [(8, 2), (2, 37), (37, 49)], 37: [(8, 2), (2, 37)], 32: [(8, 11), (11, 31), (31, 32)], 34: [(8, 2), (2, 37), (37, 34)], 41: [(8, 2), (2, 37), (37, 41)], 43: [(8, 11), (11, 3), (3, 43)], 20: [(8, 11), (11, 3), (3, 20)], 10: [(8, 13), (13, 25), (25, 10)], 5: [(8, 11), (11, 31), (31, 32), (32, 5)], 19: [(8, 13), (13, 25), (25, 19)], 18: [(8, 13), (13, 25), (25, 18)], 45: [(8, 2), (2, 37), (37, 49), (49, 45)], 1: [(8, 2), (2, 37), (37, 34), (34, 1)], 40: [(8, 2), (2, 37), (37, 41), (41, 40)], 16: [(8, 13), (13, 25), (25, 19), (19, 16)], 21: [(8, 11), (11, 31), (31, 32), (32, 5), (5, 21)], 22: [(8, 11), (11, 31), (31, 32), (32, 5), (5, 22)], 4: [(8, 11), (11, 31), (31, 32), (32, 5), (5, 4)], 44: [(8, 13), (13, 25), (25, 19), (19, 44)], 26: [(8, 2), (2, 37), (37, 34), (34, 26)], 14: [(8, 13), (13, 25), (25, 10), (10, 14)], 35: [(8, 13), (13, 25), (25, 10), (10, 35)], 6: [(8, 11), (11, 31), (31, 32), (32, 5), (5, 22), (22, 6)], 39: [(8, 13), (13, 25), (25, 10), (10, 35), (35, 39)], 28: [(8, 13), (13, 25), (25, 19), (19, 44), (44, 28)], 9: [(8, 13), (13, 25), (25, 19), (19, 16), (16, 9)], 27: [(8, 11), (11, 3), (3, 43), (43, 27)], 24: [(8, 2), (2, 37), (37, 49), (49, 45), (45, 24)], 47: [(8, 2), (2, 37), (37, 34), (34, 1), (1, 47)], 30: [(8, 2), (2, 37), (37, 49), (49, 45), (45, 30)], 12: [(8, 13), (13, 25), (25, 10), (10, 14), (14, 12)], 48: [(8, 13), (13, 25), (25, 10), (10, 14), (14, 48)], 33: [(8, 13), (13, 25), (25, 19), (19, 16), (16, 9), (9, 33)], 23: [(8, 13), (13, 25), (25, 19), (19, 16), (16, 9), (9, 23)], 29: [(8, 13), (13, 25), (25, 10), (10, 14), (14, 12), (12, 29)], 42: [(8, 13), (13, 25), (25, 19), (19, 16), (16, 9), (9, 23), (23, 42)], 17: [(8, 2), (2, 37), (37, 49), (49, 45), (45, 24), (24, 17)], 0: [(8, 13), (13, 25), (25, 10), (10, 14), (14, 48), (48, 0)], 46: [(8, 13), (13, 25), (25, 19), (19, 44), (44, 28), (28, 46)], 38: [(8, 11), (11, 31), (31, 32), (32, 5), (5, 22), (22, 6), (6, 38)], 7: [(8, 11), (11, 31), (31, 32), (32, 5), (5, 22), (22, 6), (6, 7)], 15: [(8, 11), (11, 3), (3, 43), (43, 27), (27, 15)], 36: [(8, 11), (11, 31), (31, 32), (32, 5), (5, 22), (22, 6), (6, 38), (38, 36)]}
        # every step's valid_routes
        self.valid_routes = {node: ServiceCoordination5.get_edges(route) for node, route in routes.items() if node in valid_nodes}

    def step(self, action):
        if isinstance(action,tuple):
            action=action[0]
        else:
            action=int(action)
        # updating the request,all the request has initalized in the updating_time_window
        if self.all_waiting_queues[action]!=[]:
            self.request = self.all_waiting_queues[action][0]
            del self.all_waiting_queues[action][0]
        else:
            self.valid_actions=self.valid_actions_method()
            return self.compute_state_admission(),0.0,self.done,{}
        if self.request.rejected==False:
            self.info[self.request.service].requests += 1
        else:
            pass
        #because i removed the self.routes_bidict and self.vtype_bidict,so i should to reinitial it
        if self.routes_bidict[self.request] == [] and self.request.rejected == True:
            self.initial_one_rejected_request(self.request)
        if self.request.service==0:
            self.request.datarate=0.2
            self.request.resd_lat=10000
            self.request.max_latency=10000
            self.request.duration=1
        self.update_actions()
        #because a request is removed by it can't access the endpoint because of there is no route
        if not self.valid_routes:
            self.add_reject_list(self.request)
            self.valid_actions = self.valid_actions_method()
            self.time = self.time + self.update_interval
            self.release_request()
            return self.compute_state_admission(), 0.0, self.done, {}
        # whether this place is right or not currently i should ignore it
        self.compute_state()
        state_placement, reward_step, result = self.place_request(self.request)
        # the request is accepted
        if result:
            reward = reward_step
        # the request is rejected.However it should be send to the reject queue accroding to the servece type
        else:
            self.add_reject_list(self.request)
            reward = reward_step
        self.valid_actions=self.valid_actions_method()
        # the time_window has  finished yet
        # the time_window has not finished we should update the time

        self.time = self.time + self.update_interval
        self.release_request()
        return self.compute_state_admission(), reward, self.done, {}

    def check_time_window_queue(self):
        flag=0
        for queue in self.all_waiting_queues:
            if queue == []:
                flag+=1
            else:
                flag=flag
        return flag

    def sort_new_arrival(self):
        # new-arrival memory cpu latency datarate
        #clear all the queues in the last timestep
        self.queue_new_arrival_memory=[]
        self.queue_new_arrival_cpu=[]
        self.queue_new_arrival_latency=[]
        self.queue_new_arrival_datarate=[]
        for req in self.request_list:
            if req.service == 0 and len(self.queue_new_arrival_memory) < self.MAX_QUEUE_LENGTH:
                self.queue_new_arrival_memory.append(req)
            if req.service == 1 and len(self.queue_new_arrival_cpu) < self.MAX_QUEUE_LENGTH:
                self.queue_new_arrival_cpu.append(req)
            if req.service == 2 and len(self.queue_new_arrival_latency) < self.MAX_QUEUE_LENGTH:
                self.queue_new_arrival_latency.append(req)
            if req.service == 3 and len(self.queue_new_arrival_datarate) < self.MAX_QUEUE_LENGTH:
                self.queue_new_arrival_datarate.append(req)
    # TODO:decrease the reward if the request is rejected
    def sort_rejected(self):
        # in the first timestep the reject_list is []
        self.queue_reject_memory=[]
        self.queue_reject_cpu=[]
        self.queue_reject_latency=[]
        self.queue_reject_datarate=[]
        test_list=[]
        if self.reject_list != []:
            for req in self.reject_list:
                test_list.append(req.service)
                if req.service == 0:
                    self.update_profit(req)
                    self.queue_reject_memory.append(req)
                if req.service == 1:
                    self.update_profit(req)
                    self.queue_reject_cpu.append(req)
                if req.service == 2:
                    self.update_profit(req)
                    self.queue_reject_latency.append(req)
                if req.service == 3:
                    self.update_profit(req)
                    self.queue_reject_datarate.append(req)
        self.reject_list=[]

    def check_time_end(self):
        if self.start_time <= self.TIME_HORIZON:
            return False
        else:
            return True

    # new arrival requests.
    def update_time_window(self):
        #rejected_request in the last timestep
        self.request_list=[]
        self.start_time += self.TIME_WINDOW_LENGTH
        self.end_time += self.TIME_WINDOW_LENGTH
        # update the self.time,we should release the request according to the length of all queues
        self.time = self.end_time
        self.time_window(self.start_time, self.end_time)

    # memory,cpu,latency,datarate currently this function is only used when updating the time window
    def combine_request_list(self):
        # dispatch the request according to the service type
        length = 0
        self.sort_new_arrival()
        self.sort_rejected()
        # update the last all_waiting_queues
        self.all_waiting_queues = []
        self.all_waiting_queues.append(self.queue_new_arrival_memory)
        self.all_waiting_queues.append(self.queue_new_arrival_cpu)
        self.all_waiting_queues.append(self.queue_new_arrival_latency)
        self.all_waiting_queues.append(self.queue_new_arrival_datarate)
        self.all_waiting_queues.append(self.queue_reject_memory)
        self.all_waiting_queues.append(self.queue_reject_cpu)
        self.all_waiting_queues.append(self.queue_reject_latency)
        self.all_waiting_queues.append(self.queue_reject_datarate)
        for queue in self.all_waiting_queues:
            length += len(queue)
        self.update_interval = self.TIME_WINDOW_LENGTH / length
        length_list=[]
        for queue in self.all_waiting_queues:
            length_list.append(len(queue))
    #in the next timestep the request will be removed from the reject_list
    def add_reject_list(self, request: Request):
        if request.departure_time > self.start_time +2*self.TIME_WINDOW_LENGTH:
            self.reject_list.append(request)
            request.rejected=True
    def compute_state_admission(self):
        mean_cutil = np.mean(
            [self.computing[node] / self.MAX_COMPUTE for node in self.net.nodes])
        mean_mutil = np.mean(
            [self.memory[node] / self.MAX_MEMORY for node in self.net.nodes])
        mean_lutil = np.mean([self.datarate[frozenset(
            {src, trg})] / self.MAX_LINKRATE for src, trg in self.net.edges])
        state_new_arrival_memory = len(self.queue_new_arrival_memory) / self.MAX_QUEUE_LENGTH
        state_new_arrival_datarate = len(self.queue_new_arrival_datarate) / self.MAX_QUEUE_LENGTH
        state_new_arrival_cpu = len(self.queue_new_arrival_cpu) / self.MAX_QUEUE_LENGTH
        state_new_arrival_latency = len(self.queue_new_arrival_latency) / self.MAX_QUEUE_LENGTH
        state_reject_memory = len(self.queue_reject_memory) / self.MAX_QUEUE_LENGTH
        state_reject_datarate = len(self.queue_reject_datarate) / self.MAX_QUEUE_LENGTH
        state_reject_cpu = len(self.queue_reject_cpu) / self.MAX_QUEUE_LENGTH
        state_reject_latency = len(self.queue_reject_latency) / self.MAX_QUEUE_LENGTH
        state_queue = [state_new_arrival_memory, state_new_arrival_datarate, state_new_arrival_cpu,
                       state_new_arrival_latency, state_reject_memory, state_reject_datarate, state_reject_cpu,
                       state_reject_latency]
        state_network = [mean_cutil, mean_mutil, mean_lutil]
        accept_rate_list = self.calculate_accept_rate()
        print("当前队列利用率")
        print(state_queue)
        print("当前网络利用状态")
        print(state_network)
        print("当前服务接受情况")
        print(accept_rate_list)
        print("+++++++++++++++++++++++++++++++++")
        return np.asarray(list(chain(state_queue, state_network, accept_rate_list)))

    def release_request(self):
        while self.deployed and self.deployed[0][0] < self.time:
            rel_time, service = heapq.heappop(self.deployed)
            self.release(service)

    def initial_one_request(self, request: Request):
        self.routes_bidict[request] = (None, request.ingress)
        self.info[request.service].requests += 1
        request.schedule_time=self.end_time
        # set requested VNFs upon arrival of request
        request.resd_lat = request.max_latency
        # set the profit for the request
        self.set_profit(request)
        request.vtypes = self.services[request.service]
        self.num_requests += 1
        self.set_departure_time(request)
        request.rejected=False
    def initial_one_rejected_request(self, request: Request):
        if self.routes_bidict[request] == []:
            self.routes_bidict[request] = (None, request.ingress)

    def set_departure_time(self, request: Request):
        if request.service == 0:
            request.departure_time = request.schedule_time + self.TIME_WINDOW_LENGTH * 5
        elif request.service == 1:
            request.departure_time = request.schedule_time + self.TIME_WINDOW_LENGTH * 4
        elif request.service == 2:
            request.departure_time = request.schedule_time + self.TIME_WINDOW_LENGTH * 3
        elif request.service == 3:
            request.departure_time = request.schedule_time + self.TIME_WINDOW_LENGTH * 2

    def update_profit(self, request: Request):
        if request in self.reject_list:
            request.profit = request.profit * 0.5

    def set_profit(self, request: Request):
        if request.service == 0:
            request.profit = 5
        elif request.service == 1:
            request.profit = 4
        elif request.service == 2:
            request.profit = 3
        elif request.service == 3:
            request.profit = 2

    def calculate_accept_rate(self):
        accept_rate = []
        for i in range(4):
            if (self.info[i].requests)!=0:
                accept_rate_single = self.info[i].accepts / (self.info[i].requests)
                accept_rate.append(accept_rate_single)
            else:
                accept_rate.append(0.0)
        return accept_rate

    def valid_queues(self,action):
        if self.all_waiting_queues[action] == []:
            return False
        return True

    def valid_actions_method(self):
        valid_actions= {}
        for queue in self.all_waiting_queues:
            if queue != []:
                valid_actions[self.all_waiting_queues.index(queue)]=len(queue)
        if self.needed_updating_time_window():
            self.update_time_window()
            self.combine_request_list()
            for queue in self.all_waiting_queues:
                if queue != []:
                    valid_actions[self.all_waiting_queues.index(queue)] = len(queue)
        if self.start_time==45:
            self.done = True
        return valid_actions

    def needed_updating_time_window(self):
        if self.check_time_window_queue()==6 and self.end_time<self.TIME_HORIZON:
            return True
        elif self.start_time>=self.TIME_HORIZON:
            return False