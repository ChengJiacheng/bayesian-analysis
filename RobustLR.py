# -*- coding: utf-8 -*-

import pymc3 as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    ans = sns.load_dataset('anscombe')
    x_3 = ans[ans.dataset == 'III']['x'].values
    y_3 = ans[ans.dataset == 'III']['y'].values
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    beta_c, alpha_c = stats.linregress(x_3, y_3)[:2]
    plt.plot(x_3, (alpha_c + beta_c* x_3), 'k', label='y ={:.2f} + {:.2f} * x'.format(alpha_c, beta_c))
    plt.plot(x_3, y_3, 'bo')
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$y$', rotation=0, fontsize=16)
    plt.legend(loc=0, fontsize=14)
    plt.subplot(1,2,2)
    sns.kdeplot(y_3);
    plt.xlabel('$y$', fontsize=16)
    plt.tight_layout()
    
    
    with pm.Model() as model_t:
        alpha = pm.Normal('alpha', mu=0, sd=100)
        beta = pm.Normal('beta', mu=0, sd=1)
        epsilon = pm.HalfCauchy('epsilon', 5)
        nu = pm.Deterministic('nu', pm.Exponential('nu_', 1/29) + 1)
            
        y_pred = pm.StudentT('y_pred', mu=alpha + beta * x_3, sd=epsilon, nu=nu, observed=y_3)

    
        start = pm.find_MAP()
        step = pm.Metropolis() 
        trace_t = pm.sample(2000, step=step, start=start)
    pm.traceplot(trace_t);
    
    plt.figure()
    beta_c, alpha_c = stats.linregress(x_3, y_3)[:2]
    
    plt.plot(x_3, (alpha_c + beta_c * x_3), 'k', label='non-robust', alpha=0.5)
    plt.plot(x_3, y_3, 'bo')
    alpha_m = trace_t['alpha'].mean()
    beta_m = trace_t['beta'].mean()
    plt.plot(x_3, alpha_m + beta_m * x_3, c='k', label='robust')
    
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$y$', rotation=0, fontsize=16)
    plt.legend(loc=2, fontsize=12)
    plt.tight_layout()
    