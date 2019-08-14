import numpy as np
import torch as th
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt


class con_bandit():
    '''stationary bandit problem'''
    def __init__(self):
        self.state = 0
        self.bandits = np.array([[.9,.9,.6,-5],[.7,-5,1,.8],[-5,.9,.6,.8]])
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
                    nn.Sigmoid(),
                    nn.Linear(2*input_s, output_s),
                    nn.Sigmoid())
        
    def forward(self, x):
        
        return self.net(x)
    
    
class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        
    def forward(self, output, action, reward):
        loss = -(th.log(output[0, action])*reward)
        return loss    
        
''' training loop '''
env = con_bandit()
learner = agent(env.num_bandits, env.num_actions).cuda()
learner.eval()
loss_fun = loss()
one_hot = F.one_hot(th.arange(0, env.num_bandits)).float().cuda()
epochs = 30000
e = 0.5 # epsilon for exploration
#optimizer = th.optim.SGD(learner.parameters(), lr = 0.01, 
#                         momentum=0.9)
optimizer = th.optim.RMSprop(learner.parameters(), lr = 0.001, momentum=0.8)
lr_sche = th.optim.lr_scheduler.MultiStepLR(optimizer, [6000, 8000], gamma=0.1)
total_reward = np.zeros([env.num_bandits,env.num_actions])

loss_cur = []
for epoch in range(epochs):
    
    s = env.getState() #Get a state from the environment.
    
    #Choose either a random action or one from our network.
    if np.random.rand() < e:
        pro_list = learner.forward(one_hot[s,:].unsqueeze(0))
        action = np.random.randint(env.num_actions)
    else:
        pro_list = learner.forward(one_hot[s,:].unsqueeze(0))
        action = th.argmax(pro_list)
        
    reward = env.pullArm(action) #reward for taking an action given a bandit.
    
    #Update the network.
    loss = loss_fun(pro_list, action, reward)
    loss_cur.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #Update lr
    lr_sche.step()
    
    total_reward[s, action] += reward
    if epoch % 500 == 0:
        print("Mean reward for each of the " + str(env.num_bandits) + 
              " bandits: " + str(np.mean(total_reward,axis=1)))
    

output = learner.forward(one_hot).cpu().detach().numpy()
output = np.array(output)
for a in range(env.num_bandits):
    print("The agent thinks action " + str(np.argmax(output[a,:])+1) + " for bandit " + str(a+1) + " is the most promising")
    if np.argmax(output[a,:]) == np.argmin(env.bandits[a,:]):
        print("and it was right!")
    else:
        print("and it was wrong!")

plt.figure()
plt.plot(loss_cur)
plt.show()



