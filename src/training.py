import numpy as np


class con_bandit():
    '''stationary bandit problem'''
    def __init__(self):
        self.state = 0
        self.bandits = np.array([[.2,.9,.6,-5],[.1,-5,1,.25],[-5,.5,.5,.3]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]
        
    def getState(self):
        self.state = np.random.randint(self.num_bandits)
        
        return self.state
    
    def pullArm(self, action):
        bandit = self.bandits[self.state, action]
        result = np.random.rand()
        if result > bandit:
            # return a postive reward
            return 1
        else:
            # return a negative reward
            return -1
        
