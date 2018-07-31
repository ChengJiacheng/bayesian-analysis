# -*- coding: utf-8 -*-

import pymc3 as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    
    np.random.seed(1)
    N = 100
    alfa_real = 2.5
    beta_real = 0.9
    eps_real = np.random.normal(0, 0.5, size=N)
    
    x = np.random.normal(10, 1, N)
    y_real = alfa_real + beta_real * x 
    y = y_real + eps_real
    
    plt.figure(figsize=(10,5))
    plt.plot(x, y, 'b.')
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$y$', fontsize=16, rotation=0)
    plt.plot(x, y_real, 'k')
    
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=1)
        epsilon = pm.HalfCauchy('epsilon', 5)
    
        mu = pm.Deterministic('mu', alpha + beta * x)
        y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)
        
        start = pm.find_MAP() 
        step = pm.Metropolis() 
        trace = pm.sample(10000, step, start, nchains = 1)
        
        step = pm.NUTS()
        trace_n = pm.sample(2000, step=step, start=start, nchains = 1)
    pm.traceplot(trace)
    
    plt.figure()
    sns.kdeplot(trace['alpha'], trace['beta'])
    plt.xlabel(r'$\alpha$', fontsize=16)
    plt.ylabel(r'$\beta$', fontsize=16, rotation=0)
    
    pm.traceplot(trace_n)
    
    plt.figure()
    plt.plot(x, y, 'b.');

    alpha_m = trace['alpha'].mean()
    beta_m = trace['beta'].mean()
    plt.plot(x, alpha_m + beta_m * x, c='k', label='y = {:.2f} + {:.2f} * x'.format(alpha_m, beta_m))
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$y$', fontsize=16, rotation=0)
    plt.legend(loc=2, fontsize=14)
    
    plt.figure()
    plt.plot(x, y, 'b.');
    alpha_m = trace_n['alpha'].mean()
    beta_m = trace_n['beta'].mean()
    


    idx = range(0, len(trace_n['alpha']), 10)
    plt.plot(x, trace_n['alpha'][idx] + trace_n['beta'][idx] *  x[:,np.newaxis], c='gray', alpha=0.5);
    
    plt.plot(x, alpha_m + beta_m * x, c='k', label='y = {:.2f} + {:.2f} * x'.format(alpha_m, beta_m))
    
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$y$', fontsize=16, rotation=0)
    plt.legend(loc=2, fontsize=14)