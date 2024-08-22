#!/bin/bash --login

# SGE options (whose lines must begin with #$)

#$ -cwd               # Run the job in the current directory 
source activate PYMC_GPFlow



#!/bin/sh
OUTPUTFILE="MCMC_test"
MAINSCRIPT="PData_set_up.py"
DATA="friction_conductance_power.pkl"
SUBFOLDER="~"
BASIS_FUNCS=2
PLASTIC_START=4
PLASTIC_END=-1
TRAIN_VALIDATE_RATIO="0.8"



mkdir ~/scratch/"${OUTPUTFILE}"

cp "${SUBFOLDER}${MAINSCRIPT}" ~/scratch/"${OUTPUTFILE}"
cd ~/scratch/"${OUTPUTFILE}"
python "${SUBFOLDER}${MAINSCRIPT}" "${BASIS_FUNCS}" "${PLASTIC_START}" "${PLASTIC_END}" "${TRAIN_VALIDATE_RATIO}"
