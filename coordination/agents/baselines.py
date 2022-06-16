from copy import deepcopy
from itertools import combinations_with_replacement
from collections import defaultdict

import gym
import torch as th
import numpy as np
import numpy.ma as ma
import networkx as nx
from stable_baselines3 import DQN
from stable_baselines3.a2c import A2C
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.utils import is_vectorized_observation
from stable_baselines3.ppo import PPO
from sb3_contrib.tqc import TQC
from sb3_contrib.qrdqn import QRDQN
from stable_baselines3.ppo.policies import MlpPolicy

from coordination.environment.traffic import Request


class RandomPolicy:
    def __init__(self, seed=None, **kwargs):
        np.random.seed(seed)
        self.name='Random'
    #没有神经网络的直接predict下一步动作
    def learn(self, **kwargs):
        pass

    def predict(self, env, **kwargs):
        """Samples a valid action from all valid actions."""
        #sample process
        valid_nodes = np.asarray([node for node in env.valid_routes])
        return np.random.choice(valid_nodes)


class AllCombinations:
    def __init__(self, **kwargs):
        self.actions = []
        self.name='AllCombinations'

    def learn(self, **kwargs):
        pass

    def predict(self, env, **kwargs):
        if not self.actions:
            self.actions = self._predict(env, **kwargs)
            self.actions = list(self.actions)

        action = self.actions.pop(0)
        return action

    def _predict(self, env, **kwargs):
        position = env.request.ingress
        #record the current vnf types [list]
        #记录当前 request VNF类型
        vtypes = [vnf for vnf in env.request.vtypes]
        nodes = env.net.nodes()        

        # insert upcoming request after to-be-deployed requests, so that environment 
        # does not progress in time immediately after finalized deployment
        #在待部署请求之后插入即将到来的请求，以便环境不会在最终部署后立即进行
        env = deepcopy(env)
        #(0,1)之间的一个浮点数
        arrival = env.request.arrival + np.nextafter(0, 1)
        stub = Request(arrival, env.request.duration, 0.0, float('inf'), (0, 0), 0)
        env.trace = iter([stub])
        #测定当前计算总平均
        compute = sum(env.computing.values())
        #测定当前计算总计算
        memory = sum(env.memory.values())
        capacity = sum(env.datarate.values())
        min_placements = None
        min_score = float('inf')

        # set score coefficients proportional to inverse of available capcaity
        #设置分数系数与可用容量的反比
        c_max = max(c for _, c in env.net.nodes('compute'))
        #当前的总可用率
        c_avail = compute / sum(c for _, c in env.net.nodes('compute'))
        #可用率和最大节点（不是可用的，就是总的）的乘积
        c_alpha = 1 / (c_avail * c_max)

        m_max = max(m for _, m in env.net.nodes('memory'))
        m_util = memory / sum(m for _, m in env.net.nodes('memory'))
        m_alpha = 1 / (m_util * m_max)

        d_max = max(data['datarate'] for _, _, data in env.net.edges(data=True))
        d_util = capacity / sum(data['datarate'] for _, _, data in env.net.edges(data=True))
        d_alpha = 1 / (d_util * d_max)
        #生成指定长度序列长度为vtype,pu,后面这个是VNF服务长度,nodes代表所有节点items，后面那个代表生成指定长度序列
        for placements in combinations_with_replacement(nodes, len(vtypes)):
            sim_env = deepcopy(env)
            #在这个虚拟环境中接受的总请求数
            accepts = sum(info.accepts for info in sim_env.info)
            # simulate placement actions on environment
            for num, placement in enumerate(placements):
                if not placement in sim_env.valid_routes.keys():
                    break
                sim_env.step(placement)
            # update assessment only for accepted requests
            #仅针对已接受的请求更新评估
            if accepts >= sum(info.accepts for info in sim_env.info):
                continue
                
            delta_compute = compute - sum(sim_env.computing.values())
            delta_memory = memory - sum(sim_env.memory.values())
            delta_capacity = capacity - sum(sim_env.datarate.values())

            score = c_alpha * delta_compute + m_alpha * delta_memory + d_alpha * delta_capacity
            if score < min_score:
                min_score = score
                #计算当前最小的花销方案
                min_placements = placements

        # case: no valid placement action is available: choose REJECT_ACTION
        if min_placements is None:
            return [env.REJECT_ACTION]

        if not isinstance(min_placements, tuple):
            min_placements = (min_placements)

        if min_score < float('inf'):
            return min_placements


class GreedyHeuristic:

    def __init__(self, **kwargs):
        self.name='GreedyHeuristic'

    def learn(self, **kwargs):
        pass

    def predict(self, env, **kwargs):
        valid_actions = env.valid_routes.keys()
        _, pnode = env.routes_bidict[env.request][-1]

        # compute delays (including processing delays) associated with actions
        #计算与操作相关的延迟（包括处理延迟）
        lengths, _ = nx.single_source_dijkstra(env.net, source=pnode, weight=env.get_weights, cutoff=env.request.resd_lat)
        
        # choose (valid) VNF placement with min. latency increase
        action = min(valid_actions, key=lengths.get)

        return action

class MaskedPPO(PPO):
    def predict(self, observation: th.Tensor, deterministic: bool = False, env=None, **kwargs):
        if deterministic and not env is None:
            observation = np.asarray(observation).reshape((-1,) + self.env.observation_space.shape)
            observation = th.as_tensor(observation).to('cpu')
            
            # get action mask of valid choices from environment 
            valid_actions = np.full(env.ACTION_DIM, False)
            valid = list(env.valid_routes.keys())
            valid_actions[valid] = True
            #stochastic differential equation随机微分方程
            latent_pi, _, latent_sde = self.policy._get_latent(observation)
            distribution = self.policy._get_action_dist_from_latent(latent_pi, latent_sde)
            #输出符合条件的动作的下标
            valid, = np.where(valid_actions)
            #将valid_actions进行重新编号
            actions = th.arange(valid_actions.size)
            log_prob = distribution.log_prob(actions).detach().cpu().numpy()
            #np.ma代表掩码数组，用于筛选valid_actions
            action = ma.masked_array(log_prob, ~valid_actions, fill_value=np.NINF).argmax()
            return action
        
        action, _ = super(PPO, self).predict(observation, deterministic)
        return action

    def load(self, path, device='auto'):
        # when loading a pre-trained policy from `path`, do nothing upon call to `learn`
        self.policy = MlpPolicy.load(path, device)

        def stub(*args, **kwargs):
            pass

        self.learn = stub


class MaskedA2C(A2C):
    def predict(self, observation: th.Tensor, deterministic: bool = False, env=None, **kwargs):
        if deterministic and not env is None:
            observation = np.asarray(observation).reshape((-1,) + self.env.observation_space.shape)
            observation = th.as_tensor(observation).to('cpu')

            # get action mask of valid choices from environment
            valid_actions = np.full(env.ACTION_DIM, False)
            valid = list(env.valid_routes.keys())
            valid_actions[valid] = True
            # stochastic differential equation随机微分方程
            latent_pi, _, latent_sde = self.policy._get_latent(observation)
            distribution = self.policy._get_action_dist_from_latent(latent_pi, latent_sde)
            # 输出符合条件的动作的下标
            valid, = np.where(valid_actions)
            # 将valid_actions进行重新编号
            actions = th.arange(valid_actions.size)
            log_prob = distribution.log_prob(actions).detach().cpu().numpy()
            # np.ma代表掩码数组，用于筛选valid_actions
            action = ma.masked_array(log_prob, ~valid_actions, fill_value=np.NINF).argmax()
            return action

        action, _ = super(A2C, self).predict(observation, deterministic)
        return action

    def load(self, path, device='auto'):
        # when loading a pre-trained policy from `path`, do nothing upon call to `learn`
        self.policy = MlpPolicy.load(path, device)

        def stub(*args, **kwargs):
            pass

        self.learn = stub