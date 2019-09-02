import numpy as np
import matplotlib.pyplot as plt


def func(x):
    # y = np.sum(x**2)
    y = 100 * np.sum((x[:-1]**2-x[1:])**2) + np.sum((x[:-1]-1)**2)
    return y


'''Optimizers Class'''
class Optimizer(object):
    def __init__(self, obj, epsilon=1e-08):
        self.obj = obj
        self.dim = obj.dim
        self.eps = epsilon
        self.t = 0

    def update(self, der):
        self.t += 1
        dir_ = self._compute_step(der)
        the = self.obj.mu
        ratio = np.linalg.norm(dir_) / (np.linalg.norm(the) + self.eps)
        self.obj.mu = the + dir_
        return ratio

    def _compute_step(self, der):
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, obj, stepsize, beta1=0.99, beta2=0.999):
        Optimizer.__init__(self, obj)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, der):
        w = self.stepsize * np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)
        self.m = self.beta1 * self.m + (1-self.beta1) * der
        self.v = self.beta2 * self.v + (1-self.beta2) * (der * der)
        dir_ = w * self.m / (np.sqrt(self.v) + self.eps)
        return dir_


class BasicSGD(Optimizer):
    def __init__(self, obj, stepsize):
        Optimizer.__init__(self, obj)
        self.stepsize = stepsize

    def _compute_step(self, der):
        dir_ = self.stepsize * der
        return dir_


class SGD(Optimizer):
    def __init__(self, obj, stepsize, momentum=0.9):
        Optimizer.__init__(self, obj)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize = stepsize
        self.m = momentum

    def _compute_step(self, der):
        self.v = self.m * self.v + self.stepsize*der
        dir_ = self.v
        return dir_
    

class StepLR:
    def __init__(self, optimizer, step=10, gamma=.1):
        self.step = step
        self.gamma = gamma
        self.optimizer = optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

    def update_lr(self):
        if self.optimizer.t % self.step == 0:
            self.optimizer.stepsize *= self.gamma


class BasicPSO:
    
    def __init__(self, num_params,
                 w = 0.9,
                 c1 = 2,
                 c2 = 2,
                 popsize = 15,
                 ada_w = False,
                 ada_c = False,
                 dire = 0,
                 bound_check = False):
        
        self.num_params = num_params
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.popsize = popsize
        self.ada_w = ada_w
        self.ada_c = ada_c
        self.dire = dire                 # 0: descent; 1: ascent
        self.bound_check = bound_check   # True or False
        self.lbounds = np.empty((self.popsize, self.num_params))
        self.ubounds = np.empty((self.popsize, self.num_params))
        self.solutions = np.empty((self.popsize, self.num_params))
        self.velocitys = np.empty((self.popsize, self.num_params))
        if self.ada_w:
            self.step_size = np.empty(1)
        if self.dire == 0:
            self.g_fit = np.inf
            self.p_fits = np.ones(self.popsize) * np.inf
        else:
            self.g_fit = -np.inf
            self.p_fits = -np.ones(self.popsize) * np.inf
        self.g_pop = np.empty(self.num_params)
        self.p_pops = np.empty((self.popsize, self.num_params))
    
    def start(self, lbound, ubound):
        '''initialize particles and velocity'''
        self.lbound = lbound             # array-like and size = num_params
        self.ubound = ubound             # same above
        self.lbounds = np.tile(self.lbound, (self.popsize, 1))
        self.ubounds = np.tile(self.ubound, (self.popsize, 1))
        self.solutions = self.lbounds + np.random.rand(self.popsize, self.num_params) * \
                        (self.ubounds - self.lbounds)
                        
        self.velocitys = (self.lbounds - self.ubounds) + np.random.rand(self.popsize, self.num_params) * \
                        (self.ubounds - self.lbounds) * 2
        
        return self.solutions
        
    def ask(self, check_type=None):
        '''update all particles based on the basic PSO rule'''
        g_pops = np.tile(self.g_pop, (self.popsize, 1))
        R1 = np.random.rand(self.popsize, self.num_params)
        R2 = np.random.rand(self.popsize, self.num_params)
        self.velocitys = self.w*self.velocitys + self.c1*R1*(self.p_pops-self.solutions) + \
                         self.c2*R2*(g_pops-self.solutions)
        self.solutions += self.velocitys
        
        # bound check
        if self.bound_check:
            # bound check
            if check_type == 'box':
                idb = np.where(self.solutions<self.lbounds)
                self.solutions[idb[0],idb[1]] = self.lbounds[idb[0],idb[1]]
                idu = np.where(self.solutions>self.ubounds)
                self.solutions[idu[0],idu[1]] = self.ubounds[idu[0],idu[1]]
            if check_type == 'restart':
                randmaxt = self.lbounds + np.random.rand(self.popsize, self.num_params) * \
                            (self.ubounds - self.lbounds)
                idb = np.where(self.solutions<self.lbounds)
                self.solutions[idb[0],idb[1]] = randmaxt[idb[0],idb[1]]
                idu = np.where(self.solutions>self.ubounds)
                self.solutions[idu[0],idu[1]] = randmaxt[idu[0],idu[1]]
            
        return self.solutions     
        
    def tell(self, fit_array):
        '''update p_best and g_best'''
        if self.dire == 0:
            idx = np.where(fit_array < self.p_fits)[0]
            self.p_fits[idx] = fit_array[idx]
            self.p_pops[idx, :] = np.copy(self.solutions[idx, :])
            idb = np.argmin(fit_array)
            if fit_array[idb] < self.g_fit:
                self.g_fit = fit_array[idb]
                self.g_pop = np.copy(self.solutions[idb, :])
        else:
            idx = np.where(fit_array > self.p_fits)[0]
            self.p_fits[idx] = fit_array[idx]
            self.p_pops[idx, :] = np.copy(self.solutions[idx, :])
            idb = np.argmax(fit_array)
            if fit_array[idb] > self.g_fit:
                self.g_fit = fit_array[idb]
                self.g_pop = np.copy(self.solutions[idb, :])
    
    def current_best(self):
        '''get best params and cost function value'''
        best_params = np.copy(self.g_pop)
        best_fit = np.copy(self.g_fit)
        best_pops = np.copy(self.p_pops)
        
        return (best_params, best_fit)
    
    def step(self, epochs, epoch, end_w):
        '''Implement decrease linearly weight'''
        assert self.ada_w,'Please set ada_w=True if you want to use adaptive weight'
        if epoch == 0:
            self.step_size = (self.w - end_w) / epochs
        self.w -= self.step_size
        if self.w <= 0:
            self.w = self.step_size


class BasicNES:
    def __init__(self, num_params,
                 lbound,
                 ubound,
                 mu,
                 mu_lr=0.5,
                 sigma_init=1.0,
                 sigma_lr=0.2,
                 sigma_decay=0.999,
                 sigma_db=0.01,
                 popsize=30,
                 elite_rt=1.0,
                 optim='SGD',
                 bound_check=False,
                 mirror_sample=True,
                 step=100,
                 mu_decay=0.1):

        self.dim = num_params
        self.popsize = popsize
        self.lbounds = np.tile(lbound, (self.popsize, 1))
        self.ubounds = np.tile(ubound, (self.popsize, 1))
        self.mu = mu
        self.mu_lr = mu_lr
        self.sigma = np.ones(self.dim) * sigma_init
        self.sigma_lr = sigma_lr
        self.sigma_decay = sigma_decay
        self.sigma_db = sigma_db
        self.elite_rt = elite_rt
        self.bound_check = bound_check
        self.solutions = np.empty((self.popsize, self.dim))
        if optim == 'Adam':
            self.optimizer = Adam(self, mu_lr, beta1=0.99, beta2=0.999)
        elif optim == 'BasicSGD':
            self.optimizer = BasicSGD(self, mu_lr)
        elif optim == 'SGD':
            self.optimizer = SGD(self, mu_lr, momentum=0.5)
        self.best = np.inf
        self.shapevec = np.linspace(0.5, -0.5, int(self.popsize*self.elite_rt))
        if step > 0 and 0 < mu_decay < 1:
            self.schlr = StepLR(self.optimizer, step=step, gamma=mu_decay)
        self.best_mu = np.zeros(self.dim)
        self.mirror_sample = mirror_sample  # If mirror_sample=True, the popsize must be even
        self.step = step
        self.mu_decay = mu_decay

    def ask(self, check_type=None):
        if self.mirror_sample:
            assert self.popsize % 2 == 0, "If mirror_sample=True, the popsize must be even"
            half_popsize = int(self.popsize / 2)
            epsilon_half = np.random.randn(half_popsize, self.dim)*self.sigma
            self.epsilon = np.concatenate([epsilon_half, -epsilon_half])
        else:
            self.epsilon = np.random.randn(self.popsize, self.dim)*self.sigma
        self.solutions = self.mu + self.epsilon

        if self.bound_check:
            # check type
            if check_type == "box":
                idb = np.where(self.solutions < self.lbounds)
                self.solutions[idb[0], idb[1]] = self.lbounds[idb[0], idb[1]]
                idu = np.where(self.solutions > self.ubounds)
                self.solutions[idu[0], idu[1]] = self.ubounds[idu[0], idu[1]]
            if check_type == "restart":
                randmaxt = self.lbounds + np.random.rand(self.popsize, self.dim) * \
                           (self.ubounds - self.lbounds)
                idb = np.where(self.solutions < self.lbounds)
                self.solutions[idb[0], idb[1]] = randmaxt[idb[0], idb[1]]
                idu = np.where(self.solutions > self.ubounds)
                self.solutions[idu[0], idu[1]] = randmaxt[idu[0], idu[1]]

        return self.solutions

    def tell(self, fit_array):
        if self.step > 0 and 0 < self.mu_decay < 1:
            self.schlr.update_lr()

        index = np.argsort(fit_array)
        eps_cut = self.epsilon[index[0:int(self.popsize*self.elite_rt)], :]
        fit_cut = self.shapevec

        # update mean
        gol_mu = np.sum(eps_cut*fit_cut.reshape(len(fit_cut), 1), axis=0)

        self.update_ratio = self.optimizer.update(gol_mu)

        # update sigma
        gol_sg = np.sum(fit_cut.reshape(len(fit_cut), 1)*(eps_cut**2-self.sigma**2)/self.sigma, axis=0) / (self.popsize*self.elite_rt)
        self.sigma += self.sigma_lr*gol_sg

        if fit_array[index[0]] < self.best:
            self.best = fit_array[index[0]]
            self.best_mu = np.copy(self.solutions[index[0], :])

    def current_best(self):

        return (self.best, self.best_mu, self.update_ratio)


if __name__=="__main__":

    dim = 30
    epochs = 10000
    lb = np.ones(dim) * -30
    ub = np.ones(dim) * 30
    mu = np.ones(dim) * -30
    NES = BasicNES(dim, lb, ub, mu, mu_lr=0.5, popsize=50, elite_rt=0.8, optim='SGD', mirror_sample=True,
                   step=int(epochs/3), mu_decay=0.9)
    fit_array = np.empty(NES.popsize)

    res_cur = []
    rat_cur = []
    sig_cur = []
    for i in range(epochs):

        solutions = NES.ask()
        for j in range(NES.popsize):
            fit_array[j] = func(solutions[j, :])

        NES.tell(fit_array)
        res = NES.current_best()

        # print('Iter:', i, ' bestv:', res[0])
        res_cur.append(res[0])
        rat_cur.append(res[2])
        sig_cur.append(np.linalg.norm(NES.sigma))
    # print(best)

    plt.figure()
    plt.plot(res_cur)
    plt.yscale('log')
    plt.show()

    plt.figure()
    plt.plot(rat_cur)
    plt.yscale('log')
    plt.show()

    plt.figure()
    plt.plot(sig_cur)
    plt.yscale('log')
    plt.show()
    
    
            
        
        
        
        