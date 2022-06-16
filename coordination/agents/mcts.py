from copy import deepcopy
from typing import Dict, List, Tuple
from abc import abstractmethod
from more_itertools import peekable

import numpy as np
from pairing_functions import cantor

from coordination.agents.baselines import RandomPolicy 
from coordination.environment.traffic import Traffic
from coordination.environment.deployment import ServiceCoordination


class ANode:
    #对于MCTS，初始化时需要parent和对应的action
    def __init__(self, parent, action):
        self.parent: ANode = parent
        self.action: int = action
        self.children: List[ANode] = []
        self.visits: int = 0
        self.avalue: float = 0.0


class MCTS:
    def __init__(self, C: float, max_searches: int, seed: int = None, **kwargs: Dict) -> None:
        '''Initialize Monte Carlo tree search algorithm.'''
        #C为MCTS的超参数
        self.C = C
        #用于生成随机数的种子生成器
        self.rng = np.random.default_rng(seed)
        #最大搜索次数
        self.max_searches = max_searches
    
    def learn(self, **kwargs: Dict) -> None:
        #调用非默认ROLLOUT时，先训练
        '''When a non-default rollout policy is applied, call the training hook first.'''
        pass
    #重置整棵树
    def reset(self) -> None:
        '''Reset grown search tree.'''
        self.root = None
    #在虚拟环境中评价某个节点的价值
    @abstractmethod
    def evaluate(self, sim_env: ServiceCoordination) -> float:
        '''Heuristic evaluation function to assess node value.'''
        pass

    def select_and_expand(self, sim_env: ServiceCoordination) -> Tuple:
        '''Traverse the search tree downwards and expand it by another unvisted node.'''
        #向下遍历搜索树，并通过另一个未列出的节点将其展开。
        node = self.root
        rewards = []
        
        while not sim_env.done:
            valid = [action for action in sim_env.valid_routes]
            # enable REJECT_ACTION only upon the arrival of a novel request
            #非法输入，输入请求不对,直接拒绝,valid动作集合
            if not sim_env.request in sim_env.vtype_bidict.mirror:
                valid += [sim_env.REJECT_ACTION]
            #[1,3,5]如果访问过[1,3]，那么需要加上5
            valid_visited = [child for child in node.children if child.action in valid]
            #探索操作,valid_vistited代表实际访问过的，而valid代表所有可能性
            # case: not all valid actions have been visited before; select among unexplored nodes
            if len(valid_visited) < len(valid):
                choices = set(valid) - set(child.action for child in valid_visited)
                action = self.rng.choice(list(choices))
                child = ANode(node, action)
                node.children.append(child)

                sim_env.step(action)
                #评价的是这个state,在alphazero架构里这是有神经网络估计出来的Q值，但是概率呢？？？？？？？？？？
                reward = self.evaluate(sim_env)
                rewards.append(reward)
                return child, rewards
            #用网络估算动作和对应的策略P，还有节点的Q值,网络唯一的作用是给出概率和价值
            # case: all valid actions have been visited before; progress in search tree,所有的子节点都被访问好了
            avalues = np.asarray([child.avalue for child in valid_visited])
            visits = np.asarray([child.visits for child in valid_visited])    
            
            # choose child node according to UCT formula
            #计算子节点UCT
            uct = avalues + self.C * np.sqrt((2 * np.log(np.sum(visits))) / visits)
            choice = np.argmax(uct)
            #是树嘛，置换根节点用于迭代，返回的是sim_env end时候的根结点和他的收益
            node = valid_visited[choice]

            # update simulation environment and search; proceed in search tree
            #更新虚拟环境和搜索过程，在搜索树中继续
            sim_env.step(node.action)
            reward = self.evaluate(sim_env)
            #增加的是每一步的reward
            rewards.append(reward)
      # reached end-of-episode
        return node, rewards
    #反向计算所有reward，这个node是ANode不是底层节点，你看下面的pseudo_action，这个值绝对不是1-12的底层物理节点
    def backpropagate(self, node: ANode, rewards: List) -> None:
        '''Backpropagate simulated rewards along the traversed search tree.'''
        #反向计算所有节点的奖励
        rewards = np.cumsum(rewards[:: -1])

        for rew in rewards:
            node.visits += 1
            #avalue和reward的转换公式
            node.avalue = ((node.visits - 1) / node.visits) * node.avalue + (rew / node.visits)
            node = node.parent
    #估计的是状态？？？？？？？？
    def rollout(self, sim_env: ServiceCoordination) -> float:
        '''Obtain a Monte Carlo return estimate by simulating the episode until termination.'''
        #“通过模拟事件直到终止，获得蒙特卡罗回报估计。,在估计某一轮结束时的估计奖励”,注意这里是在预估
        mc_return = 0.0
        obs = sim_env.compute_state()

        # simulate rollout phase until termination
        while not sim_env.done:
            # use default (rollout) policy to select an action
            #rollout用预测的方法做,说是rpolicy，其实就是调了一个predict函数
            action =  self.rpolicy.predict(observation=obs, deterministic=True, env=sim_env, process=None)       
            obs, reward, _, _ = sim_env.step(action)

            # cumulate simulated rewards for Monte Carlo return estimate
            reward = self.evaluate(sim_env)    
            mc_return += reward

        return mc_return
    #对于trace进行接下来的流量,所有的流量请求集合
    @abstractmethod
    def flow_forecast(self, sim_env: ServiceCoordination, trace: List) -> List:
        '''Prepare forecast flows for simulating service coordination.'''
        pass
    #利用rollout反馈value函数
    def predict(self, env: ServiceCoordination, process: Traffic, **kwargs: Dict):
        '''Use Monte Carlo tree search to select the next service deployment decision.'''
        #选用分数最高的子节点来执行动作
        # initialize planning procedure for current simulation environment
        self.root = ANode(None, None)

        for _ in range(self.max_searches):
            # setup simulation environment for MCTS
            sim_env = deepcopy(env)
            sim_env.planning_mode = True

            # replace future flow arrivals either with forecast flows or with no later arrivals
            #用预测流量或没有延迟到达的流量替换未来的流量到达,预测出的是流量，可迭代的对象
            trace = self.flow_forecast(sim_env, process.sample())
            sim_env.trace = peekable(iter(trace))
            # select an unexplored node that is a descendent of known node
            node, rewards = self.select_and_expand(sim_env)

            # use rollout policy if no value function is provided，由根结点和收益反向传递给路径上的所有节点
            rol_return = self.rollout(sim_env)
            rewards[-1] += rol_return

            self.backpropagate(node, rewards)

        # select child of root node that was visited most
        #执行时所选取的访问最多的子节点
        child = np.argmax([child.visits for child in self.root.children])
        return self.root.children[child].action

#C是为超参数，max_searches代表最大搜索次数，max_requests代表处理的最大请求数，rpolicy代表agent采取的策略，没有重写rollout方法，也就是说rollout是一样的
class FutureCoord(MCTS):
    def __init__(self, C: float, max_searches: int, max_requests: int, rpolicy, seed: int =None, **kwargs: Dict):
        super().__init__(C, max_searches, seed)
        self.rpolicy = rpolicy
        #处理最大的请求数
        self.max_requests = max_requests
        self.name='FutureCoord'
    #回调函数
    def learn(self, **kwargs):
        '''Train default policy if `self.rpolicy` is trainable.'''
        if callable(getattr(self.rpolicy, 'learn', None)):
            self.rpolicy.learn(**kwargs)
    #预测后面max—requests的流量请求
    def flow_forecast(self, sim_env: ServiceCoordination, trace: List) -> List:
        '''Simulate `max_requests` forecast flows during the rollout phase.'''
        # only simulate requests that arrive after the to-be-deployed request
        #trace request list,筛选出当前时间之后max_requests的流量（万一预测不出来呢？？？？？？？？）
        trace = [req for req in trace if sim_env.time < req.arrival]
        #the simulation max_requests is 30,流量超出，就选取最大的流量进行simulation
        # simulate at maximum `max_requests` requests  
        trace = trace[: self.max_requests]
        return trace
    #对当前请求的一个判断(true),FutureCoord采用了稀疏反馈（从环境中进行了稀疏反馈）
    def evaluate(self, sim_env: ServiceCoordination) -> float:
        '''Uses sparse 0/1 feedback from environment.'''
        #使用来自环境的稀疏0/1反馈。deployed代表这个流是否部署成功，而finalized代表这个流是否部署结束
        return float(sim_env.admission['deployed'])
    #在select_and_expand的时候会去做一个判断
    def select_and_expand(self, sim_env: ServiceCoordination) -> Tuple:
        '''Traverse the search tree downwards and expand it by another unvisted node. Inserts delimiter nodes upon arrivals.'''
        #向下遍历搜索树，并通过另一个未列出的节点将其展开。到达时插入分隔符节点。
        node = self.root
        rewards = []
        
        # track number of processed requests to determine whether new request has arrived
        #跟踪已处理请求的数量，以确定新请求是否已到达
        num_requests = sim_env.num_requests
        
        while not sim_env.done:
            #是不是具体的放置操作，看step
            # case: next action relates to deployment of NEXT request()这个if-else是用来判断是训练还是真实玩的
            if num_requests < sim_env.num_requests:
                # check whether another request with the same ingress has been simulated before
                #判断同一个进出口的相同服务是否之前就判断过,相当于一个查询hash的操作,相当于我在树上先查找一下有没有部署这个的经验，ingress和egress和request
                #这里有一个问题哦，action是一个放置操作，但是节点是和请求有关？
                pseudo_action = cantor.pair(sim_env.request.ingress, sim_env.request.service, sim_env.request.egress)
                # case: requested service arrives at prev. unseen ingress; insert fitting pseudo node
                #情况：之前已经放置过此类服务（同入口，出口，服务类型）;到达时间可以不同。插入拟合伪节点
                visited = [child.action for child in node.children]

                if not pseudo_action in visited:
                    child = ANode(node, pseudo_action)
                    node.children.append(child)
                    rewards.append(0.0)
                    return child, rewards

                # case: fitting pseudo node is already in search tree; proceed selection
                #case：拟合伪节点已经在搜索树中;继续选择
                node = next(child for child in node.children if child.action == pseudo_action)
                rewards.append(0.0)

                # update number of simulated service requests
                num_requests = sim_env.num_requests

            # case: next action extends deployment of (real) to-be-deployed request
            #这个if判断是来判断创造新请求对象，还是request对象不变，具体一个请求中的放置操作
            else:
                #valid_routes代表从上一跳可以到达的下一跳的节点号，candidate_nodes
                valid = [action for action in sim_env.valid_routes]
                # enable REJECT_ACTION only upon the arrival of a novel request
                if not sim_env.request in sim_env.vtype_bidict.mirror:
                    valid += [sim_env.REJECT_ACTION]

                valid_visited = [child for child in node.children if child.action in valid]

                # case: not all valid actions have been visited before; select among missing nodes
                if len(valid_visited) < len(valid):
                    choices = set(valid) - set(child.action for child in valid_visited)
                    action = int(self.rng.choice(list(choices)))
                    child = ANode(node, action)
                    node.children.append(child)
                    #更新子节点的value
                    sim_env.step(action)
                    reward = self.evaluate(sim_env)    
                    rewards.append(reward)
                    return child, rewards

                # case: all valid actions have been visited before; progress in search tree
                avalues = np.asarray([child.avalue for child in valid_visited])
                visits = np.asarray([child.visits for child in valid_visited])    
                
                # choose child node according to UCT formula 
                uct = avalues + self.C * np.sqrt((2 * np.log(np.sum(visits))) / visits)
                choice = np.argmax(uct)
                node = valid_visited[choice]

                # update simulation environment and search; proceed in search tree 
                sim_env.step(node.action)
                reward = self.evaluate(sim_env)    
                rewards.append(reward)
        
        # reached end-of-episode
        return node, rewards

class MavenS(MCTS):

    def __init__(self, C: float, max_searches: int, greediness: float, coefficients: Tuple, seed: int = None, **kwargs: Dict):
        super().__init__(C, max_searches, seed)
        self.greediness = greediness
        self.alpha, self.beta, self.gamma = coefficients
        self.rpolicy = RandomPolicy(seed=seed)

    def flow_forecast(self, sim_env: ServiceCoordination, trace: List) -> List:
        '''MavenS does not consider future flow arrivals.'''
        return []
    #rollout是估计，evaluation是实际收益，其实也没错啦，evaluate是
    def evaluate(self, sim_env: ServiceCoordination) -> float:
        '''Calculate deployment costs from occupied compute, memory and datarate resources upon admission.'''

        # case: intermediate deployment decision
        if not sim_env.admission['finalized']:
            return 0.0

        # case: to-be-deployed request was rejected,deployed代表的reward
        elif sim_env.admission['finalized'] and not sim_env.admission['deployed']:
            return self.greediness

        # case: to-be-deployed request was successfully deployed
        compute = sim_env.occupied['compute'] / max(c for _, c in sim_env.net.nodes('compute'))
        memory = sim_env.occupied['memory'] / max(c for _, c in sim_env.net.nodes('memory'))
        datarate = sim_env.occupied['datarate'] / max(data['datarate'] for _, _, data in sim_env.net.edges(data=True))
        
        # compute reward as resource cost given by weighted (normalized) increase of resources 
        return (-1) * (self.alpha * compute + self.beta * memory + self.gamma * datarate)


class FutureMavenS(FutureCoord):
    def __init__(self, C: float, max_searches: int, greediness: float, coefficients: Tuple, max_requests: int, rpolicy, seed: int = None, **kwargs: Dict):
        super().__init__(C, max_searches, max_requests, rpolicy, seed)
        self.greediness = greediness
        self.alpha, self.beta, self.gamma = coefficients
        self.name='FutureMavenS'
    #部署花销
    def evaluate(self, sim_env: ServiceCoordination) -> float:
        '''Calculate deployment costs from occupied compute, memory and datarate resources upon admission.'''
        #在入站时根据占用的计算、内存和数据速率资源计算部署成本。
        # case: intermediate deployment decision
        if not sim_env.admission['finalized']:
            return 0.0

        # case: to-be-deployed request was rejected
        elif sim_env.admission['finalized'] and not sim_env.admission['deployed']:
            return self.greediness

        # case: to-be-deployed request was successfully deployed
        compute = sim_env.occupied['compute'] / max(c for _, c in sim_env.net.nodes('compute'))
        memory = sim_env.occupied['memory'] / max(c for _, c in sim_env.net.nodes('memory'))
        datarate = sim_env.occupied['datarate'] / max(data['datarate'] for _, _, data in sim_env.net.edges(data=True))
        
        # compute reward as resource cost given by weighted (normalized) increase of resources 
        return (-1) * (self.alpha * compute + self.beta * memory + self.gamma * datarate)
