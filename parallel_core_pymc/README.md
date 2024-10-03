The following scripts are for training the Gaussian process and then
conducting MCMC:

- Data_set_up_step.sh
- PYMC_GPflow_CSF3sampler_step1.sh
- PYMC_GPflow_CSF3sampler_step2.sh
- PYMC_GPflow_CSF3sampler_step3.sh

## Instructions for `Data_set_up_step.sh`

Take the assembled results, and use the scikit-fda python module (may
need to be installed) to reduce the amount of output data to
coefficients of basis functions, and separate training and validation
data.

## Instructions for `PYMC_GPflow_CSF3sampler_step1.sh`:

This script collects the relevant data together into a file, and calls a
python script for training a Gaussian process regression model with a
number of manually specified variables. The data gathered includes a set
of training data in the form of input (x) data and the corresponding
output (y) training data. The Gaussian process is intended to be trained
on the coefficients of basis functions from functional principal
component analysis.

This script begins with a set of variables that need to be specified
before running.

OUTPUTFILE: the intended name of the file created in the scratch folder
where the data is collected, and outputs are stored.

MAINSCRIPT:the name of the python script used which is
`PYMC_GPflow_SF3sampler_step1.py`

XNORMALISER, YNORMALISER: the data was normalised between 0 and 1 using
an instance of the normaliser class found in the script. It contains the
data required to recover the normalised inputs. This is not necessary
for this step, but is useful in step 2. XNORMALISER refers to the input
data normaliser, and YNORMALISER is for the output data (the basis
function coefficients)

XDATA, YDATA: the training data for the inputs and outputs of the GP
model

XVALIDATE, YVALIDATE Validation data

MODEL_NAME: the name of the saved GP model for both step 2 and for any
later work that uses the model

MODELFOLDER: rds folder containing the data used

SUBFOLDER: the folder containing the script

VARDAT: used in step 2. Experimental data curves used to generate a
covariance matrix of the basis functions for use in the Bayesian
statistics

KERNEL: which kernel to use. Can choose from "RBF", "Matern52",
"Matern32", "Matern12" and "Auto". The kernel specifies the behaviour of
the GP, and the best one is likely to be one of these 4 kernels combined
with a linear kernel. If not known, specify "Auto" and all 4 of these
kernels are trained on the data. The Gaussian process that produces the
lowest chi squared score for the mean values for predict_f when
predicting the validation data is the one that is chosen.

The model is then saved both as a tensor flow file (for transferring to
other computers if additional checks are needed) and as a pickle file so
that the model posterior can be used to save time by preventing
redundant calculations.

Additionally, a plot of the normalised validation basis function data vs
the data predicted by the Gaussian process are produced to visually
check the the model.

## Instructions for `PYMC_GPflow_CSF3sampler_step2.sh`:

This produces a job array for Pymc Makov chains. Large quantities of
memory may be necessary for the use of the posterior of the model as the
model saves previous predictions from the GP to prevent additional
calculations.

Select the number of Markov chains in the shell script by modifying this
line (currently set to 25 chains):
#\$ -t 1-25

Modify the following variables appropriately:

OUTPUTFILE: set the the same as in step 1
MAINSCRIPT: `PYMC_GPflow_CSF3sampler_v2pkl.py` the python script used
in this step

XNORMALISER, YNORMALISER: the same normaliser objects saved as pickle
files as step 1

MODEL: This is the model pickle created in step 1 saved as a pickle
file

SUBFOLDER: the location of the python script

VARDAT: experimental data used during the MCMC

BASIS_FUNCS: number of basis functions used to create the output
training and validation data. This is used to find the function
coefficients for the data, and the covariance matrix is then used in the
MCMC.

PLASTIC_START, PLASTIC_END: some of the Abaqus time steps are ignored
when fitting the basis functions. This determines the range of strain
values examined from the experimental data for the basis function
covariance matrix. These is determined when setting up the output
training data.

BURN_IN: number of samples ignored at the start of the Markov chain to
prevent biasing the result

SAMPLE: length of each Markov chain

COEFFS: basis coefficients based on the mean data used to perform the
Bayesian inference

The result will be a series of netcdf files for each Markov chain titled
`Idata_chain_*.nc` depending on the chain number

## Instructions for `PYMC_GPflow_CF3sampler_step3.sh`:

Simply change the output file name. This simply concatenates all of the
data in the netcdf files in the chosen output file
