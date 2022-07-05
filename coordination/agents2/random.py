import numpy as np


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
        requests_list_length=len(env.get_requests())
        results=np.random.random(size=requests_list_length)
        results_list = []
        for result in results:
            if result > 0.5:
                result = 1
                results_list.append(result)
            else:
                result = 0
                results_list.append(result)
        return results_list