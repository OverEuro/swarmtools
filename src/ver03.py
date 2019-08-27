import numpy as np
import matplotlib.pyplot as plt



def testfun(x):
#    loss = np.sum((x - np.ones(len(x))*10)**2)
    loss = 100 * np.sum((x[:-1]**2-x[1:])**2) + np.sum((x[:-1]-1)**2)
    return loss


def GetExp(pop, popf, xnes, sigma, dim):
    
    sp = np.ones((len(popf), dim))
    for i in range(len(popf)):
        sp[i,:] = popf[i]*((1/np.sqrt(2*np.pi*sigma**2))*np.exp(-(pop[i,:]-xnes)**2/(2*sigma**2)))
    
    return np.sum(sp) / (len(popf)*dim)
    

def SimpleGauss(gen, sig, num, dim):
    
    xsga = -np.ones(dim) * 30
    res = []
    best = 1e+10
    for i in range(gen):
        if (i+1)%200 == 0:
            sig /= 10
        for j in range(num):
            
            tril = xsga + np.random.randn(dim) * sig
            fit = testfun(tril)
            
            if fit < best:
                best = fit
                xsga = tril
        res.append(best)
        
    return xsga, res

def NES(gen, sig, num, dim):

    xnes = -np.ones(dim) * 30
    res = []
    exp_1 = []
    exp_2 = []
    best = 1e+10
    pop = np.ones((num, dim))
    popf = np.ones(num)
    sigma = np.ones(dim) * sig
    shapevec = np.linspace(0.5, -0.5, num)
    delta_m = np.zeros(dim)
    delta_s = np.zeros(dim)
    for i in range(gen):
        
        for j in range(num):
            
            tril = xnes + np.random.randn(dim) * sigma
            pop[j, :] = tril
            fit = testfun(tril)
            popf[j] = fit
            
            if fit < best:
                best = fit
#                xnes = tril
        res.append(best)
        
        index = np.argsort(popf)
        popf[index] = shapevec
        exp_1.append(GetExp(pop, popf, xnes, sigma, dim))
        # Update the xmean and sigam
        sum_m = 0
        sum_s = 0
        for q in range(num):
            sum_m = sum_m + popf[q] * (pop[q, :] - xnes)
            sum_s = sum_s + popf[q] * ((pop[q, :] - xnes)**2 - sigma**2) / (sigma)
        delta_m = 0.5 * delta_m + sum_m
        delta_s = 0.5 * delta_s + sum_s/num
        xnes += 0.5 * delta_m
        sigma += 0.2 * delta_s
        
#        print(sigma)
        
        exp_2.append(GetExp(pop, popf, xnes, sigma, dim))
    
    return xnes, res, exp_1, exp_2


def NESelite(gen, sig, num, dim):

    xnes = -np.ones(dim) * 30
    res = []
    exp_1 = []
    exp_2 = []
    best = 1e+10
    pop = np.ones((num, dim))
    popf = np.ones(num)
    sigma = np.ones(dim) * sig
    elites = int(np.ceil(num*0.8))
    shapevec = np.linspace(0.5, -0.5, elites)
    delta_m = np.zeros(dim)
    delta_s = np.zeros(dim)
    for i in range(gen):
        
        for j in range(num):
            
            tril = xnes + np.random.randn(dim) * sigma
            pop[j, :] = tril
            fit = testfun(tril)
            popf[j] = fit
            
            if fit < best:
                best = fit
#                xnes = tril
        res.append(best)
        
        index = np.argsort(popf)
        popn = pop[index[0:elites], :]
        popfn = popf[index[0:elites]]
        popfn = shapevec
        exp_1.append(GetExp(popn, popfn, xnes, sigma, dim))
        # Update the xmean and sigam
        sum_m = 0
        sum_s = 0
        for q in range(elites):
            sum_m = sum_m + popfn[q] * (popn[q, :] - xnes)
            sum_s = sum_s + popfn[q] * ((popn[q, :] - xnes)**2 - sigma**2) / (sigma)
        delta_m = delta_m * 0.5 + sum_m
        delta_s = delta_s * 0.5 + sum_s/num
        xnes += 0.5 * delta_m
        sigma += 0.2 * delta_s
        
#        print(sigma)
        
        exp_2.append(GetExp(popn, popfn, xnes, sigma, dim))
    
    return xnes, res, exp_1, exp_2



if __name__ == "__main__":

    xsga, sga = SimpleGauss(6000, 1, 30, 10)
    xnes, nes, exp_1, exp_2 = NES(6000, 1, 30, 10)
    xeli, eli, exp_3, exp_4 = NESelite(6000, 1, 30, 10)
    
    plt.figure()
    plt.plot(sga, label='SimpleGauss')
    plt.plot(nes, label='NES')
    plt.plot(eli, label='NESelite')
    plt.yscale('log')
    plt.legend()
    plt.grid()
#    plt.savefig('com_1.png', dpi=600)
    plt.show()
    
    plt.figure()
    plt.plot(exp_3, label='1')
    plt.plot(exp_4, label='2')
#    plt.yscale('log')
    plt.legend()
    plt.grid()
#    plt.savefig('com_2.png', dpi=600)
    plt.show()