#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:59:57 2024

@author: w10944rb
"""

import numpy as np
import pandas as pd
import skfda
import pickle
from sys import argv


#Manually specified variables
dataframepicklefile = argv[1]
n_basis = int(argv[2])
Plastic_start_step = int(argv[3])
Plastic_end_step = int(argv[4])
train_to_validate = float(argv[5])

#Define normaliser class
class Normaliser():
    def __init__(self, data=None):
        self.min = None
        self.max = None
        if data is not None:
            self.add_data(data)

    @property
    def ready(self):
        return self.min is not None and self.max is not None

    @property
    def scale(self):
        self.check_ready()
        return np.abs(self.max - self.min)

    @property
    def offset(self):
        self.check_ready()
        return self.min

    def check_ready(self):
        if not self.ready:
            raise ValueError('No data set')

    def add_data(self, data):
        new_min = data.min(axis=0)
        new_max = data.max(axis=0)
        if self.ready:
            mask = new_min < self.min
            self.min[mask] = new_min[mask]

            mask = new_max > self.max
            self.max[mask] = new_max[mask]
        else:
            self.min = new_min
            self.max = new_max

    def normalise(self, data, i=None):
        self.check_ready()
        if i is None:
            return (data - self.offset) / self.scale
        else:
            assert isinstance(i, int)
            
    def recover(self, data, i=None):
        self.check_ready()
        if i is None:
            return data * self.scale + self.offset
        else:
            assert isinstance(i, int)
            return data * self.scale[i] + self.offset[i]
        
#Import the data

data = pd.read_pickle(dataframepicklefile)


Temp_prof = []
PEEQ_prof = []
Barrelling_prof = []

#Turn the cells containing data tables into useable numpy arrays
for n,cell in enumerate(data['Temperature profile']):
    if not(isinstance(cell,float)):
        prof = np.zeros((len(cell)-7,2))
        for k,string in enumerate(cell[4:44]):
            nums = string.split()
            prof[k,0] = float(nums[0])
            prof[k,1] = float(nums[1])
        Temp_prof.append(prof)
    else:
        Temp_prof.append([])

for cell in data['PEEQ Results']:
    if not(isinstance(cell,float)):
        prof = np.zeros((len(cell)-8,2))
        for k,string in enumerate(cell[3:-5]):
            nums = string.split()
            prof[k,0] = float(nums[0])
            prof[k,1] = float(nums[1])
        PEEQ_prof.append(prof)
    else:
        PEEQ_prof.append([])



for cell in data['Barrelling Profile']:
    if not(isinstance(cell,float)):
        prof = np.zeros((len(cell)-7,2))
        for k,string in enumerate(cell[3:44]):
            nums = string.split()
            prof[k,0] = float(nums[0])
            prof[k,1] = float(nums[1])
        Barrelling_prof.append(prof)
    else:
        Barrelling_prof.append([])
        
Force = []
failures = []
for i in range(len(data['Force Results1'])):

    if not(isinstance(data['Force Results1'][i],float)):
        res1 = []
        res2 = []
        for j,string in enumerate(data['Force Results1'][i][3:-4]):
            nums= string.split()
            res1.append(float(nums[1]))
        for j,string in enumerate(data['Force Results2'][i][3:-4]):
            nums= string.split()
            res2.append(float(nums[1]))
        Force.append(np.array(res1) + np.array(res2))
    else:
        Force.append([0])

#Convert force-displacement into stress and strain values
true_strain = np.linspace(0,0.475,20)
sample_volume = np.pi*(5e-3**2)*10e-3
sample_area = sample_volume / (np.exp(-true_strain) /100)

#Create a boolean vector containing all of the simulations that successfully
#completed
truth_list = []
for i in Force:
    truth_list.append(len(i) == 20)

Force_filtered = np.zeros((sum(truth_list),20))

count = 0
for n, values in enumerate(Force):

    if truth_list[n]:
        Force_filtered[count,:] = values
        count += 1


true_stress_filtered = np.zeros(np.shape(Force_filtered))
for n, force in enumerate(Force_filtered):
    true_stress_filtered[n,:] = force / sample_area


basis = skfda.representation.basis.MonomialBasis(n_basis=n_basis)


#filter out failed simulations
valid_sims = np.array(data[['Friction','Conductance']][truth_list])

#Normalise the data, then conduct fPCA on the normalised data
X = Normaliser(valid_sims)
X_normalised = X.normalise(valid_sims)
fixed_data=skfda.FDataGrid(true_stress_filtered[:,Plastic_start_step:Plastic_end_step],grid_points=true_strain[Plastic_start_step:Plastic_end_step])
basis_rep = fixed_data.to_basis(basis)
Y = Normaliser(basis_rep.coefficients)
Y_normalised = Y.normalise(basis_rep.coefficients)

#Separate the training and validation data
data_randomiser = np.random.choice(np.shape(Y_normalised)[0],np.shape(Y_normalised)[0],replace=False)
Y_randomised = np.zeros(np.shape(Y_normalised))
X_randomised = np.zeros(np.shape(X_normalised))
for n,i in enumerate(data_randomiser):
    Y_randomised[n,:]=Y_normalised[i,:]
    X_randomised[n,:]=X_normalised[i,:]
cut_off = int(np.shape(Y_randomised)[0]*train_to_validate)
X_train = X_randomised[:cut_off,:]
X_validate = X_randomised[cut_off:,:]
Y_train = Y_randomised[:cut_off,:]
Y_validate = Y_randomised[cut_off:,:]

pd.DataFrame(X_train).to_pickle("X_train.pkl")
pd.DataFrame(Y_train).to_pickle("Y_train.pkl")
pd.DataFrame(X_validate).to_pickle("X_validate.pkl")
pd.DataFrame(Y_validate).to_pickle("Y_validate.pkl")

with open("Quad_plast_800C001_X.pkl","wb") as outp:
    pickle.dump(X, outp, pickle.HIGHEST_PROTOCOL)
with open("Quad_plast_800C001_Y.pkl","wb") as outp:
    pickle.dump(Y, outp, pickle.HIGHEST_PROTOCOL)