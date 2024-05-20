#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:06:59 2023

@author: w10944rb
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
import pymc as pm
import pytensor
import pytensor.tensor as pt
from datetime import datetime


#Inputs here
Model_pickle_file = "friction_conductance_power.pkl"
Force_time_data_file = "800C_1s-1_csv.csv"
x_correction = 0.009
sigma = 54
timesteps = np.linspace(0.025,0.475,19)

def input_set_up(friction,conductance,power,max_time=50,time_points=1000):
    #Inflexible function designed to take material parameters and create an array 
    #for use generating predicted outputs of the gaussian process
    X_matrix = np.zeros((time_points,4))
    X_matrix[:,0] = np.linspace(0,max_time,time_points)
    X_matrix[:,1] = friction
    X_matrix[:,2] = conductance
    X_matrix[:,3] =power
    return X_matrix

def input_set_up2(changing_var, *args):
    #flexible for parameters specified as positional arguments
    output = np.zeros((len(changing_var),len(args)+1))
    output[:,0] = changing_var
    #print('input_set_up2 args:',args[0])
    for n,i in enumerate(args):
        #print('loop:',i)
        output[:,n+1] = i
    return output

def input_set_up3(changing_var, *args):
    #Flexible for parameters specified as a list
    output = np.zeros((len(changing_var),len(args[0])+1))
    output[:,0] = changing_var
    #print('input_set_up2 args:',args[0])
    for n,i in enumerate(args[0]):
        #print('loop:',i)
        output[:,n+1] = i
    return output

def my_model(theta, x,gp_model):

    inputs = input_set_up3(x,theta)
    #print(inputs)
    predicted_output = gp_model.predict_f(inputs)
    return predicted_output[0].numpy()


def my_loglike(theta, x, data, sigma,gp_model):
    p_model = my_model(theta, x,gp_model)
    return -(0.5 / (len(data)*(sigma**2))) * np.sum((data - p_model) ** 2)

data = pd.read_pickle(Model_pickle_file)

Temp_prof = []
PEEQ_prof = []
Barrelling_prof = []


for cell in data['Temperature profile']:
    prof = np.zeros((len(cell)-7,2))
    for k,string in enumerate(cell[4:44]):
        nums = string.split()
        prof[k,0] = float(nums[0])
        prof[k,1] = float(nums[1])
    Temp_prof.append(prof)

for cell in data['PEEQ Results']:
    prof = np.zeros((len(cell)-8,2))
    for k,string in enumerate(cell[3:-5]):
        nums = string.split()
        prof[k,0] = float(nums[0])
        prof[k,1] = float(nums[1])
    PEEQ_prof.append(prof)



for cell in data['Barrelling Profile']:
    prof = np.zeros((len(cell)-7,2))
    for k,string in enumerate(cell[3:44]):
        nums = string.split()
        prof[k,0] = float(nums[0])
        prof[k,1] = float(nums[1])
    Barrelling_prof.append(prof)

#There is a bug in this section returning twice as many data points
for cell in data['Barrelling Profile']:
    prof = np.zeros((len(cell)-7,2))
    for k,string in enumerate(cell[3:44]):
        nums = string.split()
        prof[k,0] = float(nums[0])
        prof[k,1] = float(nums[1])
    Barrelling_prof.append(prof)

Force = np.zeros((len(data['Force Results1']),20))
for i in range(len(data['Force Results1'])):
    res1 = np.zeros((len(data['Force Results1'][i][3:23]),2))
    res2 = np.zeros((len(data['Force Results2'][i][3:23]),2))
    for j,string in enumerate(data['Force Results1'][i][3:23]):
        nums= string.split()
        res1[j,0] = float(nums[0])
        res1[j,1] = float(nums[1])
    for j,string in enumerate(data['Force Results2'][i][3:23]):
        nums= string.split()
        res2[j,0] = float(nums[0])
        res2[j,1] = float(nums[1])
    Force[i,:] = res1[:,1] + res2[:,1]
    
final_step_temperature_filter = [(i[:,1].max() <1000) for i in Temp_prof]

x = np.linspace(0,50,19)
X = x[:,None]
n_outputs=20
Y = np.zeros((sum(final_step_temperature_filter),19))
for n,i in enumerate(Force[final_step_temperature_filter]):
    Y[n,:] = np.array(i[1:]) #First step where time=0 and force =0 is omitted
Y_nu = np.ravel(Y)

fric= np.zeros(len(Y_nu))
cond = np.zeros(len(Y_nu))
power = np.zeros(len(Y_nu))
tim = np.zeros(len(Y_nu))
oidx = np.zeros(len(Y_nu))
for i in range(len(data['Friction'][final_step_temperature_filter])):
    n = i*19
    fric[n:n+19] = np.array(data['Friction'][final_step_temperature_filter])[i]
    cond[n:n+19] = np.array(data['Conductance'][final_step_temperature_filter])[i]
    power[n:n+19] = np.array(data['Power'][final_step_temperature_filter])[i]
    tim[n:n+19] = x
    oidx[n:n+19] = int(i)

nu_results = pd.DataFrame({'Friction': fric, 'Conductance': cond, 'Power': power, 'Time':tim, 'Force': Y_nu,  
'Output idx':oidx})

X = np.array(nu_results[['Time', 'Friction','Conductance']])
Y = np.array(nu_results['Force'])

print('Max fric:', nu_results['Friction'].max(),'Min Fric:', nu_results['Friction'].min())
print('Max cond:', nu_results['Conductance'].max(),'Min cond:', nu_results['Conductance'].min())

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
        
max_cond = 2000
cond_filter = X[:,2] < max_cond

X_normaliser = Normaliser(X[cond_filter,:])
Y_normaliser = Normaliser(Y)
X_normed = X_normaliser.normalise(X[cond_filter,:])
Y_normed = Y_normaliser.normalise(Y)

with open("output.txt","w") as ou:
   ou.write(f"X input size: {np.shape(X_normed)} Force input size: {np.shape(Y[cond_filter,None])}")
ou.close()

model = gpflow.models.GPR(
    (X_normed, Y[cond_filter,None]),
    kernel=gpflow.kernels.RBF(np.shape(X_normed)[-1], lengthscales=np.ones(np.shape(X_normed)[-1])),

)

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
with open("output.txt","a") as ou:
   ou.writelines(f"{current_time} Beginning GP model training")
ou.close()
print("Start Time =", current_time)



opt = gpflow.optimizers.Scipy()
result=opt.minimize(model.training_loss, model.trainable_variables)
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
with open("output.txt","a") as ou:
   ou.writelines(f"{current_time} GP model training complete")
   ou.writelines(f"Variance: {model.kernel.variance}")
   ou.writelines(f"Lengthscales: {str(model.kernel.lengthscales[:])}")
ou.close()

#Check if the GP qas successfully trained. If not, then try a different kernel
if not(result.success):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open("output.txt","a") as ou:
       ou.writelines(f"{current_time} GP RBF model failed, training Matern 5/2")
    ou.close()
    model = gpflow.models.GPR(
        (X_normed, Y[cond_filter,None]),
        kernel=gpflow.kernels.Materrn52(np.shape(X_normed)[-1], lengthscales=np.ones(np.shape(X_normed)[-1])),)
    opt = gpflow.optimizers.Scipy()
    result=opt.minimize(model.training_loss, model.trainable_variables)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open("output.txt","a") as ou:
       ou.writelines(f"{current_time} GP model training complete")
       ou.writelines(f"Variance: {model.kernel.variance}")
       ou.writelines(f"Lengthscales: {str(model.kernel.lengthscales[:])}")
    ou.close()

if not(result.success):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open("output.txt","a") as ou:
       ou.writelines(f"{current_time} GP Matern 5/2 model failed, training Matern 3/2")
    ou.close()
    model = gpflow.models.GPR(
        (X_normed, Y[cond_filter,None]),
        kernel=gpflow.kernels.Matern32(np.shape(X_normed)[-1], lengthscales=np.ones(np.shape(X_normed)[-1])),)
    opt = gpflow.optimizers.Scipy()
    result=opt.minimize(model.training_loss, model.trainable_variables)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open("output.txt","a") as ou:
       ou.writelines(f"{current_time} GP model training complete")
       ou.writelines(f"Variance: {model.kernel.variance}")
       ou.writelines(f"Lengthscales: {str(model.kernel.lengthscales[:])}")
    ou.close()
    
if not(result.success):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open("output.txt","a") as ou:
       ou.writelines(f"{current_time} GP Matern 3/2 model failed, training Matern 1/2")
    ou.close()
    model = gpflow.models.GPR(
        (X_normed, Y[cond_filter,None]),
        kernel=gpflow.kernels.Matern12(np.shape(X_normed)[-1], lengthscales=np.ones(np.shape(X_normed)[-1])),)
    opt = gpflow.optimizers.Scipy()
    result=opt.minimize(model.training_loss, model.trainable_variables)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open("output.txt","a") as ou:
       ou.writelines(f"{current_time} GP model training complete")
       ou.writelines(f"Variance: {model.kernel.variance}")
       ou.writelines(f"Lengthscales: {str(model.kernel.lengthscales[:])}")
    ou.close()



def data_vs_gaussian3inp(X,Y, curve_no, axes,m):
    loc1 = curve_no*19
    loc2 = loc1+19
    x_test = np.linspace(0,1,1000)
    x_example = input_set_up2(x_test,  X[loc1,1],X[loc1,2])
    axes.plot(X[loc1:loc2,0],Y[loc1:loc2],'ko', label='data')
    print(x_example)
    #axes.set_title('Curve '+str(curve_no)+' Friction='+str(X[loc1,1])+' Cond='+str(X[loc1,2])+' Power='+str(X[loc1,3]))
    mean,var = m.predict_f(x_example)

    axes.plot(x_test,mean)
    axes.fill_between(x_test,
                 mean[:,0] - 1.96 * np.sqrt(var[:,0]),
                 mean[:,0] + 1.96 * np.sqrt(var[:,0]),
                 color='C0', alpha=0.2)
    plot_no = str(curve_no)

validation_curves=[1,5,67,88,95,10,16,109,91,53]
rows=int(np.ceil((len(validation_curves)/5)))
cols=5

fig,ax=plt.subplots(rows,cols, figsize=(20,10*int(np.ceil(rows/cols))))
#ax(0).plot(np.linspace(0,10,10))
for i in range(rows):
    for j in range(cols):
        if (i*cols)+j <= len(validation_curves):
            data_vs_gaussian3inp(X_normed,Y,validation_curves[(i*cols)+(j)],ax[i,j],model)
            
fig.savefig("check_fit.png")

now = datetime.now()
current_time = now.strftime("%H:%M:%S")

with open("output.txt","a") as ou:
   ou.writelines(f"{current_time} Plotting complete, preparing MCMC")
ou.close()

def normal_gradients(theta, x, data, sigma, GPmodel):
    """
    Calculate the partial derivatives of a function at a set of values. The
    derivatives are calculated using the central difference, using an iterative
    method to check that the values converge as step size decreases.

    Parameters
    ----------
    theta: array_like
        A set of values, that are passed to a function, at which to calculate
        the gradient of that function
    x, data, sigma:
        Observed variables as we have been using so far


    Returns
    -------
    grads: array_like
        An array of gradients for each non-fixed value.
    """

    grads = np.empty(2)
    
    aux_vect = (data[:,None] - my_model(theta, x, GPmodel))#/0.01 #/(2*sigma**2)
    grads[0] = np.sum(aux_vect * x)
    grads[1] = np.sum(aux_vect)

    return grads

class LogLikeWithGrad(pt.Op):
    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, x, sigma,GPmodel):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.x = x
        self.sigma = sigma
        self.GPmodel = GPmodel

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.data, self.x, self.sigma, self.GPmodel)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.x, self.data, self.sigma, self.GPmodel)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        return [g[0] * self.logpgrad(theta)]


class LogLikeGrad(pt.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, data, x, sigma, GPmodel):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.data = data
        self.x = x
        self.sigma = sigma
        self.GPmodel = GPmodel

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        # calculate gradients
        grads = normal_gradients(theta, self.x, self.data, self.sigma, self.GPmodel)

        outputs[0][0] = grads

dat = np.loadtxt(Force_time_data_file, delimiter=",")


Force_at_800C_1s = np.zeros(len(timesteps))
for n,step in enumerate(timesteps):
    Force_at_800C_1s[n] = dat[:,1][abs((dat[:,0]+x_correction)-step)==min(abs((dat[:,0]+x_correction)-step))]

now = datetime.now()
current_time = now.strftime("%H:%M:%S")

with open("output.txt","a") as ou:
   ou.writelines(f"{current_time} Force values: {Force_at_800C_1s}")
ou.close()


x_norm = np.linspace(0,1,19)
logl = LogLikeWithGrad(my_loglike, Force_at_800C_1s, x_norm,sigma,  model.posterior())
with pm.Model() as pymodel:
    # uniform priors on f and c
    f = pm.Uniform("Friction", lower=0, upper=1)
    c = pm.Uniform("Conductance", lower=0, upper=1)

    #sigma = pm.Uniform("sigma", lower=0,upper=1)

    # convert m and c to a tensor vector
    theta = pt.as_tensor_variable([f,c])

    # use a Normal distribu ntion
    pm.Potential("likelihood", logl(theta))
    #step=pm.NUTS(step_scale=0.005, adapt_step_size=False)
    step=pm.Metropolis(step_scale=0.01, adapt_step_size=False)

    idata = pm.sample(tune=10000, draws=20000, step=step,cores=1, chains=5)

idata.to_netcdf("800C_1s-1_friction_conductance.nc")
