import numpy as np
import torch as th
import torch.nn as nn 


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


class agent(nn.Module):
    
    def __init__(self, input_s, output_s):
        super(agent, self).__init__()
        self.net = nn.Sequential(
                    nn.Linear(input_s, 2*input_s),
                    nn.Dropout(0.3),
                    nn.ELU(),
                    nn.Linear(2*input_s, output_s),
                    nn.Softmax(dim=1))
        
    def forward(self, x):
        
        return self.net(x)
    
    
class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        
    def forward(self, output, action, reward):
        loss = -(th.log(output[1, action]*reward))
        return loss    
#net = network(3, 3)
#x = th.ones((1, 3))
#y = net.forward(x)
#print(y)



        
        
        
        
        
        
        
        
        
        
        