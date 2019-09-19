import numpy as np
import torch as th
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import swarmtools as sts


class con_bandit():
    '''stationary bandit problem'''
    def __init__(self):
        self.state = 0
        self.bandits = np.array([[.9,.9,.6,.2],[.7,.2,1,.8],[.2,.9,.6,.8],
                                 [.5,.6,1,.2],[.4,.5,.2,.8],[.4,.2,.7,.8],
                                 [.5,.6,1,.2],[.4,.5,.2,.8],[.4,.2,.7,.8]])
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
                    nn.Linear(input_s, output_s),
                    nn.Sigmoid())
        
    def forward(self, x):
        
        return self.net(x)
    
    
def update_model(flat_param, model, model_shapes):
    idx = 0
    i = 0
    for param in model.parameters():
        delta = np.product(model_shapes[i])
        block = flat_param[idx:idx+delta]
        block = np.reshape(block, model_shapes[i])
        i += 1
        idx += delta
        block_data = th.from_numpy(block).float()

        block_data = block_data.cuda()
        param.data = block_data


'''NES training loop'''
env = con_bandit()
learner = agent(env.num_bandits, env.num_actions).cuda()
orig_params = []
model_shapes = []
for param in learner.parameters():
    p = param.data.cpu().detach().numpy()
    model_shapes.append(p.shape)
    orig_params.append(p.flatten())
orig_params_flat = np.concatenate(orig_params)
NPARAMS = len(orig_params_flat)
print("The number of NN's params =", NPARAMS)

learner.eval()
one_hot = F.one_hot(th.arange(0, env.num_bandits)).float().cuda()
eval_num = 300000  # the number of samples
lb = np.ones(NPARAMS) * -1
ub = np.ones(NPARAMS) * 1
mu = np.zeros(NPARAMS)
optimizer = sts.BasicNES(NPARAMS, lb, ub, mu, mu_lr=0.1, popsize=30, elite_rt=0.8, optim='SGD', mirror_sample=True,
                         step=10, mu_decay=0.9)

# solutions = optimizer.start(lbound, ubound)
fits = np.empty(optimizer.popsize)
evals = 0
batch_size = env.num_bandits * 15
epoch = 0
best_f = []
epochs = int(eval_num / (optimizer.popsize * batch_size))
while evals < eval_num:

    solutions = optimizer.ask()
    # compute all particles' fitness:
    for i in range(optimizer.popsize):
        update_model(solutions[i, :], learner, model_shapes)
        sum_r = 0
        for j in range(batch_size):
            s = env.getState()  # get an random state from env
            pro_list = learner.forward(one_hot[s, :].unsqueeze(0))
            action = th.argmax(pro_list)
            reward = env.pullArm(action)
            sum_r += reward
        fits[i] = -sum_r
    
    optimizer.tell(fits)
    
#    optimizer.step(epochs, epoch, end_w=0.1)
    
    # print evolution process
    best_f.append(optimizer.current_best()[0])
    print('EPOCH:', epoch, 'Fitness:', optimizer.current_best()[0])
    epoch += 1
    
    # update evals
    evals += optimizer.popsize * batch_size

best_params = optimizer.current_best()[1]
update_model(best_params, learner, model_shapes)

output = learner.forward(one_hot).cpu().detach().numpy()
for a in range(env.num_bandits):
    print("The agent thinks action " + str(np.argmax(output[a,:])+1) + " for bandit " + str(a+1) + " is the most promising")
    if np.argmax(output[a,:]) == np.argmin(env.bandits[a,:]):
        print("and it was right!")
    else:
        print("and it was wrong!")

plt.figure()
plt.plot(best_f)
plt.xlabel('Epochs')
plt.ylabel('Fitness')
#plt.savefig('res_nes.png',dpi=600)
plt.show()



