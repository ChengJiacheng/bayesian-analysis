# -*- coding: utf-8 -*-

import pymc3 as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    clusters = 3

    n_cluster = [90, 50, 75]
    n_total = sum(n_cluster)
    
    means = [9, 21, 35]
    std_devs = [2, 2, 2]

    mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
    
    with pm.Model() as model_ug:
        # Each observation is assigned to a cluster/component with probability p
        p = pm.Dirichlet('p', a=np.ones(clusters))
        category = pm.Categorical('category', p=p, shape=n_total) 
        
        # We estimate the unknown gaussians means and standard deviation
        means = pm.Normal('means', mu=[10, 20, 35], sd=2, shape=clusters)
        sd = pm.HalfCauchy('sd', 5)
    
        y = pm.Normal('y', mu=means[category], sd=sd, observed=mix)
    
        step1 = pm.ElemwiseCategorical(vars=[category], values=range(clusters))
        step2 = pm.Metropolis(vars=[means, sd, p])
        trace_ug = pm.sample(10000, step=[step1, step2], nchains=1)
        
        chain_ug = trace_ug[1000:]
        pm.traceplot(chain_ug)
    
    plt.figure()
    ppc = pm.sample_ppc(chain_ug, 50, model_ug)
    for i in ppc['y']:
        sns.kdeplot(i, alpha=0.1, color='b')
    
    sns.kdeplot(np.array(mix), lw=2, color='k') # you may want to replace this with the posterior mean
    plt.xlabel('$x$', fontsize=14)