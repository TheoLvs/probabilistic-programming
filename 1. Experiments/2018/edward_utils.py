#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
EDWARD UTILS
Started on the 16/12/2017
theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats

# Probabilistic programming
import tensorflow as tf
import edward as ed
from edward.models import Bernoulli,Beta,Empirical,Normal,Poisson,Uniform,Exponential





class Distribution(object):
    def __init__(self,prior = None,name = None,**kwargs):
        self.prior = prior


    def init_posterior(self,positive = True,empirical = True,n_samples = 1000):


        if empirical:
            if positive:
                self.posterior = Empirical(params=tf.nn.softplus(tf.Variable(tf.random_normal([n_samples]))))
            else:
                self.posterior = Empirical(params=tf.Variable(tf.random_normal([n_samples,])))


        else:
            if positive:
                self.posterior = Normal(loc=tf.Variable(tf.random_normal([1])),
                                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))





    def plot(self,n_samples = 10000,show = True,bins = 20):
        sns.distplot(self.prior.sample(n_samples).eval(),bins = bins,label = "prior",hist = False,kde_kws={"shade": True})
        sns.distplot(self.posterior.sample(n_samples).eval(),bins = bins,label = "posterior",hist = False,kde_kws={"shade": True})
        plt.legend()
        if show:
            plt.show()





class Problem(object):
    def __init__(self,latent_vars = None):
        self.latent_vars = latent_vars


    def add_latent_var(self,latent_var):
        self.latent_vars.append(latent_var)

    def unwind_latent_vars(self):
        return {v.prior:v.posterior for v in self.latent_vars}


    def run(self,data,method = "klqp",**kwargs):

        if method == "klqp":
            print(">> Initializing ... ",end = "")
            inference = ed.KLqp(self.unwind_latent_vars(), data=data)
            inference.initialize(**kwargs)
            print("ok")

            # RUNNING THE INFERENCE
            sess = ed.get_session()
            init = tf.global_variables_initializer()
            init.run()
            losses = []
            for _ in tqdm(range(inference.n_iter)):
                info_dict = inference.update()
                losses.append(info_dict['loss'])
            plt.figure(figsize = (7,3))
            plt.title("Loss")
            plt.semilogy(losses)
            plt.show()


        elif method == "hmc":
            print(">> Initializing ... ",end = "")
            inference = ed.HMC(self.unwind_latent_vars(), data=data)
            inference.initialize(**kwargs)
            print("ok")

            # RUNNING THE INFERENCE
            sess = ed.get_session()
            init = tf.global_variables_initializer()
            init.run()
            acceptance_rates = []
            for _ in tqdm(range(inference.n_iter)):
                info_dict = inference.update()
                acceptance_rates.append(info_dict['accept_rate'])
            plt.figure(figsize = (7,3))
            plt.title("Acceptance Rate")
            plt.semilogy(acceptance_rates)
            plt.show()