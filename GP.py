# reference
# https://en.wikipedia.org/wiki/Gaussian_process
# https://github.com/PacktPublishing/Bayesian-Analysis-with-Python/tree/master/Chapter%208

#%matplotlib inline
import pymc3 as pm
import numpy as np
import theano.tensor as tt
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


if __name__ == "__main__":
    np.random.seed(1)
    squared_distance = lambda x, y: cdist(x.reshape(-1,1), y.reshape(-1,1)) ** 2 #SED function

    N = 20         # number of training points.
    n = 100         # number of test points.
    
    np.random.seed(1)
    f = lambda x: np.sin(x).flatten()

    x = np.random.uniform(0, 10, size=N)
    y = np.random.normal(np.sin(x), np.sqrt(0.01))

    plt.plot(x, y, 'o')
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$f(x)$', fontsize=16, rotation=0)
    
    with pm.Model() as GP:
        mu = np.zeros(N)
        eta = pm.HalfCauchy('eta', 0.1)
        rho = pm.HalfCauchy('rho', 1)
        sigma = pm.HalfCauchy('sigma', 1)
        
        D = squared_distance(x, x) #SED(x,x)
        
        K = tt.fill_diagonal(eta * pm.math.exp(-rho * D), eta + sigma) #(K(x, x) + σ I)
        
        obs = pm.MvNormal('obs', mu, cov=K, observed=y)
        
    
        test_points = np.linspace(0, 10, 100)
        D_pred = squared_distance(test_points, test_points) #SED(x*,x*)
        D_off_diag = squared_distance(x, test_points) #SED(x,x*) n * N
        
        K_oo = eta * pm.math.exp(-rho * D_pred) #K(x*,x*)
        K_o = eta * pm.math.exp(-rho * D_off_diag) #K(x,x*)

        inv_K = tt.nlinalg.matrix_inverse(K)
        
        mu_post = pm.Deterministic('mu_post', pm.math.dot(pm.math.dot(K_o.T, inv_K), y))
        SIGMA_post = pm.Deterministic('SIGMA_post', K_oo - pm.math.dot(pm.math.dot(K_o.T, inv_K), K_o))        

        step = pm.Metropolis()                
        start = pm.find_MAP()
        trace = pm.sample(1000, step = step, start=start, nchains = 1)        
        varnames = ['eta', 'rho', 'sigma']
        chain = trace[100:]
        pm.traceplot(chain, varnames)
        
        plt.figure()
        y_pred = [np.random.multivariate_normal(m, S) for m,S in zip(chain['mu_post'], chain['SIGMA_post'])]        
        for yp in y_pred:
            plt.plot(test_points, yp, 'r-', alpha=0.1)
        
        plt.plot(x, y, 'bo')
        plt.xlabel('$x$', fontsize=16)
        plt.ylabel('$f(x)$', fontsize=16, rotation=0)