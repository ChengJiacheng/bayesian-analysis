# -*- coding: utf-8 -*-
import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
if __name__ == "__main__":
    
    np.random.seed(123)
    n_experiments = 100
    theta_real = 0.35  # unkwon value in a real experiment
    alpha_prior = 1
    beta_prior = 1
    data = stats.bernoulli.rvs(p=theta_real, size=n_experiments)
    data_sum = data.sum()
    
    with pm.Model() as our_first_model:
        # a priori
        theta = pm.Beta('theta', alpha=1, beta=1)
        # likelihood
        y = pm.Bernoulli('y', p=theta, observed=data)
        #y = pm.Binomial('theta',n=n_experimentos, p=theta, observed=sum(datos))
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(2000, step=step, start=start,  nchains = 1)
        burnin = 0  # no burnin
        chain = trace[burnin:]
        pm.traceplot(chain, lines={'theta':theta_real});
        
    plt.figure()    
    x = np.linspace(.2, .6, 1000)
    func = stats.beta(a =alpha_prior+data_sum, b=beta_prior+n_experiments-data_sum)
    y = func.pdf(x)
    
    plt.plot(x, y, 'r-', lw=3, label='True distribution')
    plt.hist(chain['theta'], bins=30, normed=True, label='Estimated posterior distribution')


