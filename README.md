# Constructing a Bayesian inference workflow in Matflow

## Aim of the project:

To construct a workflow capable of performing Bayesian inference using
Monte Carlo Markov chain (MCMC) sampling of the inputs into an Abaqus
FEM model. As the MCMC sampling requires large numbers of evaluations of
the model, the only practical way to do this is to use a
machine-learning method to construct a statistical model and sample the
statistical model instead of the Abaqus model.

## The basic workflow is as follows:

1. select a bunch of parameter samples
2. Use the parameter samples to edit an Abaqus input file, and run each
   of the simulations
3. Collect the outputs and put the relevant data into some easily
   computable format
4. Take the x-y type data and convert each possible graph into a handful
   of coefficients using functional Principal Component Analysis (fPCA)
5. Map the inputs to outputs by constructing a statistical model.
   Gaussian Process regression has so far been the favoured method
6. Perform MCMC on the statistical model

## Previous work stored in this repository:

In the repository are the scripts, and input files that directly
implement most of the above steps, and the workflow runs as following:

1. A bash script can be found in `surrogate_Abaqus_3_800C_1s-1.txt`. Its
   function is to run Abaqus a number of times while varying interface
   friction and thermal conductance between the sample and the machine.
   To do this it gathers together:
   
   - the python script that calls abaqus (surrogate_Abaqus_3_800C_1s-1.py),
   
   - a python script that is run within Abaqus CAE
     
     (`read_Force_PEEQ_NT11_barrelling_forcemac.py`) to extract data from
       output database (.odb) files,
   
   - the basic Abaqus input file `800C_1s-1_step_19.inp`
   
   - Some output databases from the folder Abaqus_5_min_heatup, for which
     the relevant one for a given sample is imported into Abaqus, and is
     selected by modifying the relevant line in the `800C_1s-1_step_19.inp`
     file. This ensures the sample begins the compression test with the
     correct temperature distribution profile for the given thermal
     conductance value.

2. Once `surrogate_Abaqus_3_800C_1s-1.txt` is run, it puts the model
   information and scripts into a folder and runs `surrogate_Abaqus_3_800C_1s-1.py`
   
   `surrogate_Abaqus_3_800C_1s-1.py` contains functions for reading in the
   text in 800C_1s-1_step_19.inp, changing the values of friction and
   thermal conductance, then writing them in to a new `.inp` file. Samples
   are chosen very simply by using nested for loops near the bottom of the
   script. The thermal conductance term changes both the gap conductance in
   the `.inp` file, and the imported temperature profile from the relevant
   `.odb` file.
   
   A pandas data frame is created to store the values of the friction and
   thermal conductance, and the contents of the `.rpt` files containing the
   relevant model outputs.
   
   The function `call_abaqus_with_new_params` takes as input the varying
   model parameters to modify the input file, the name of the input file,
   the directory of the output file, and the count/ number of the
   simulation so that the new output file may have a unique(ish) name and
   to find the relevant line in the pandas data frame.
   
   `call_abaqus_with_new_params` then modifies the input file to put the
   relevant input parameters in the right place, then Abaqus is called using
   said input file. Once complete, an abaqus output database (.odb) is
   generated, and the script `read_Force_PEEQ_NT11_barrelling_forcemac.py` is
   run within an Abaqus kernel to extract the most interesting data and put
   them into a text file with a `.rpt` suffix.
   
   Once the data is extracted as `.rpt` files, they are simply read into a
   python variable and dumped into the pandas data frame.
   
   The pandas Dataframe is then saved as a pickle (`.pkl`) file and used in
   the next script.

3. As with step 1, `MCMC_800_1s-1.txt` is a bash script designed to put
   files in the same place and then run the `MCMC_800C_1s-1.py` script. It
   gathers together the pickle file from the previous step, some physical
   test data (required for the bayesian inference), and the script.

4. `MCMC_800C_1s-1.py` is where the real magic happens. First it imports
   the data from the pickle file, and then puts them into nice useable
   numpy arrays. Then it filters some data out by NT11 values (the material
   data used in the model is based on a lookup table with temperatures
   between 700-1000 degrees, and straying outside of this causes output
   discontinuities that really mess with the gaussian processes)
   
   There is a normaliser object coded here is designed to normalise the
   input values between 0 and 1, and keep track of what the original values
   were (If the input values are numerically different orders of magnitude,
   that also messes with the gaussian process). The inputs are then
   normalised.
   
   Once the gaussian process is complete, a handful of the data is plotted
   against the gaussian process values for the given input parameters.
   (Although what really needs to happen is that the data is split to a
   training and validation set, and we plot all the validation data)
   
   There are various classes and likelihood functions that are wrappers for
   the inputs into the bayesian inference model. Don't worry too much about
   these at this point, we will be using Metropolis sampling to avoid using
   the more efficient/vastly more annoying gradient based methods.
   
   Pymc uses a context manager to build models. Once the sampling is
   complete , the data can be moved to a netcdf file for analysis in
   Jupyter.

## How to Use Previous work:

Open `surrogate_Abaqus_3_800C_1s-1.txt`.

Modify the following data:

```
OUTPUTFILE="SFCP_800C_1s-1_imprtedodbs_cond0-1500" # Desired output file location
MAINSCRIPT="surrogate_Abaqus_3_800C_1s-1.py" # Script to run
IMPORTTEMPFOLER="Abaqus_5_min_heatup/" # Location of heat up .odbs
TEMPERATUREODB="800" #Temperature leave as 800
TEMPERATUREODBITERATION='22' # Number of time steps in the heat up model. Leave as 22
MODEL="800C_1s-1_setup.inp" # Model name
PLASTICITYDATA="Patryk_mat_data.txt" # Possibly redundant code, ignore
ODBREADER="read_Force_PEEQ_NT11_barrelling_forcemac.py" # Script to read Abaqus outputs
SUBFOLDER="Abaqus/Friction_conductance/" # Where the script and .inp file is kept
```

Open `MCMC_800_1s-1.txt`

Modify the following data:

```
OUTPUTFILE="MCMC_GPsurrgt_800C_1s-1_cond0-1500_20000_chain/" #Desired output file location
MAINSCRIPT="MCMC_800C_1s-1.py" # MCMC script, leave
MODELDATA="friction_conductance_power.pkl" # Name of the pickle file after running the previous script
MODELDATAFOLDER="scratch/SFCP_800C_1s-1_importedodbs_cond0-1500/" # Where the above pickle file is kept
SUBFOLDER="GP_surrogate_pymc/" # Where the MCMC script is kept
TESTDATAFOLDER= "Patryk_Force_time_data/" # Where the physical data Force-time data is kept
TESTDATA="800C_1s-1_csv.csv" # Physical data file
```
