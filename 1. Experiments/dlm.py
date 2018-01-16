#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
DLM
Started on the 2018/01/16
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
import itertools

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression




def generate_campaign(start_date,length,intensity,std_noise = 0,gamma = (2,3,7)):
    support = np.linspace(0,length,100)
    distribution = stats.gamma.pdf(support,*gamma)
    distribution = np.divide(distribution,np.max(distribution))
    lever = np.zeros(100)
    lever[start_date:start_date+length] = intensity * distribution[::int(100/length)][:length] + np.random.normal(0,std_noise,length)
    return lever





def generate_promotion(start_date,length,intensity,std_noise = 0,gamma = (2,1)):
    support = np.linspace(0,length,100)
    distribution = stats.gamma.pdf(support,*gamma)
    distribution = np.divide(distribution,np.max(distribution))
    lever = np.zeros(100)
    lever[start_date:start_date+length] = intensity * distribution[::int(100/length)][:length] + np.random.normal(0,std_noise,length)
    return lever






class Variable(object):
    def __init__(self,params = None):
        if type(params) == list: params = np.array(params)
        self.set_params(params)


    def __add__(self,other):
        params = 0.5 * (self.params + other.params)
        return Variable(params = params)


    def set_params(self,params = None):
        if params is None:
            start = np.random.randint(0,95)
            length = np.random.randint(5,100-start)
            params = np.array([start,length])

        self.params = params.astype(int)
        self.params[0] = min([95,self.params[0]])
        self.params[1] = min([100-self.params[0],self.params[1]])

        self.data = self.generate()



    def mutate(self,p = 0.5,impact = 0.1):
        noise = stats.bernoulli.rvs(size = self.params.shape,p = p).astype(float)
        noise *= stats.uniform.rvs(size = self.params.shape,loc = -impact,scale = 2*impact)
        noise *= self.params
        noise = noise.astype(int)

        params = self.params + noise
        self.set_params(params)



    def plot(self):
        plt.plot(self.data)





class Campaign(Variable):
    def __init__(self,params = None):
        super().__init__(params = params)


    def __add__(self,other):
        params = 0.5 * (self.params + other.params)
        return Campaign(params = params)


    def __repr__(self):
        return "Campaign(start={},length={})".format(*self.params)

    def generate(self):
        return generate_campaign(self.params[0],self.params[1],1)





class Promotion(Variable):
    def __init__(self,params = None):
        super().__init__(params = params)


    def __add__(self,other):
        params = 0.5 * (self.params + other.params)
        return Promotion(params = params)


    def __repr__(self):
        return "Promotion(start={},length={})".format(*self.params)

    def generate(self):
        return generate_promotion(self.params[0],self.params[1],1)





class Variables(object):
    def __init__(self,variables = None):
        if variables is None:
            variables = [Campaign(),Promotion(),Promotion()]
        self.variables = variables

    def __getitem__(self,key):
        return self.variables[key]

    def __iter__(self):
        return iter(self.variables)

    def __len__(self):
        return len(self.variables)


    def __add__(self,other):
        variables = []
        for i in range(len(self)):
            variables.append(self[i]+other[i])
        return Variables(variables = variables)


    def evaluate(self,sales):
        data = self.get_data()
        self.lr = LinearRegression()
        self.lr.fit(data,sales)
        prediction = self.lr.predict(data)
        return r2_score(prediction,sales)


    def mutate(self):
        for variable in self.variables:
            variable.mutate()


    def get_data(self):
        return np.vstack([v.data for v in self]).T



    def plot(self,sales):
        r2 = self.evaluate(sales)
        d = np.array(range(100))
        data = self.get_data()
        layers = [list(data.T[i]*self.lr.coef_[i]) for i in range(len(self.lr.coef_))]
        base = list(np.ones(100) * self.lr.intercept_)
        plt.title("R2 = {:.3g}".format(r2))
        plt.stackplot(d,[base,*layers])
        plt.plot(sales,color = "black")
        plt.show()




class Population(object):
    def __init__(self,elements = None,n = 50):

        self.elements = elements if elements is not None else [Ensemble() for i in range(n)]


    def __getitem__(self,key):
        if type(key) == tuple or type(key) == list:
            d = []
            for i in key:
                d.append(self.elements[i])
            return d
        else:
            return self.elements[key]
    
    def __iter__(self):
        return iter(self.elements)
    
    def __len__(self):
        return len(self.elements)



    def evaluate(self):
        fitnesses = [(i,element.evaluate()) for i,element in enumerate(self)]
        indices,fitnesses = zip(*sorted(fitnesses,key = lambda x : x[1],reverse = True))
        return indices,fitnesses



    def selection(self,top = 0.5):
        indices,fitnesses = self.evaluate()
        n = int(top*len(fitnesses))
        return indices[:n]



    def crossover(self,indices):
        combinations = list(itertools.combinations(indices,2))
        np.random.shuffle(combinations)
        combinations = combinations[:len(self)]
        new_population = []
        for i,j in combinations:
            new_population.append(self[i]+self[j])

        if len(new_population) < len(self):
            new_population.extend([Ensemble() for i in range(len(self)-len(new_population))])
        self.elements = new_population



    def mutate(self):
        for d in self:
            d.mutate()


    def evolve(self,top = 0.25):
        indices = self.selection(top = top)
        self.crossover(indices)
        self.mutate()
        
