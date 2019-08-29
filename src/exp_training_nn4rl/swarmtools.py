import numpy as np


def func(x):
    y = np.sum(x**2)
    return y


'''Optimizers Class'''
class Optimizer(object):
    def __init__(self, obj, epsilon=1e-08):
        self.obj = obj
        self.dim = obj.num_params
        self.eps = epsilon
        self.t = 0

    def update(self, der):
        self.t += 1
        dir_ = self._compute_dir(der)
        the = self.obj.mu
        rat = np.linalg.norm(dir_) / (np.linalg.norm(the) + self.eps)
        self.obj.mu = the + dir_
        return rat

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
        dir_ = -w * self.m / (np.sqrt(self.v) + self.eps)
        return dir_


class BasicSGD(Optimizer):
    def __init__(self, obj, stepsize):
        Optimizer.__init__(self, obj)
        self.stepsize = stepsize

    def _compute_step(self, der):
        dir_ = -self.stepsize * der
        return dir_


class SGD(Optimizer):
    def __init__(self, obj, stepsize, momentum=0.9):
        Optimizer.__init__(self, obj)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize = stepsize
        self.m = momentum

    def _compute_step(self, der):
        self.v = self.m * self.v - self.stepsize*der
        dir_ = self.v
        return dir_


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
                 sigma_init=1,
                 sigma_lr=0.2,
                 popsize=30,
                 elite_rt=1,
                 ada_ert=False,
                 dire=0,
                 smooth_fit=True,
                 bound_check=False):

        self.dim = num_params
        self.popsize = popsize
        self.lbounds = np.tile(lbound, (self.popsize, 1))
        self.ubounds = np.tile(ubound, (self.popsize, 1))
        self.mu = mu
        self.mu_lr = mu_lr
        self.sigma = np.ones(self.dim) * sigma_init
        self.sigma_lr = sigma_lr
        self.elite_rt = elite_rt
        self.ada_ert = ada_ert
        self.dire = dire
        self.smooth_fit = smooth_fit
        self.bound_check = bound_check
        self.solutions = np.empty((self.popsize, self.dim))

    def ask(self, check_type=None):
        self.solutions = self.mu + np.random.randn(self.popsize, self.dim)*self.sigma

        if self.bound_check:
            # check type
            if check_type == "box":
                idb = np.where(self.solutions < self.lbounds)
                self.solutions[idb[0], idb[1]] = self.lbounds[idb[0], idb[1]]
                idu = np.where(self.solutions > self.ubounds)
                self.solutions[idu[0], idu[1]] = self.ubounds[idu[0], idu[1]]





    

if __name__=="__main__":
    
    dim = 30
    epochs = 10000
    lb = np.ones(dim) * -30
    ub = np.ones(dim) * 30
    PSO = BasicPSO(dim, lb, ub, popsize=30, ada_w=True, dire=0, bound_check=True)
    
    solutions = PSO.start() # initial
    fit_array = np.empty(PSO.popsize)
    
    for i in range(epochs):
        
        for j in range(PSO.popsize):
            fit_array[j] = func(solutions[j, :])
        
    
        PSO.tell(fit_array)
        solutions = PSO.ask()
        res = PSO.current_best()
        
        PSO.step(epochs, i, end_w=0.1)
        
        print('Iter:', i, ' bestv:', res[1])
        
    
    
            
        
        
        
        