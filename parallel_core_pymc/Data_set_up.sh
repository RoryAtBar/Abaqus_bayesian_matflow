#!/bin/bash --login

# SGE options (whose lines must begin with #$)

#$ -cwd               # Run the job in the current directory 
source ../test_venv/bin/activate



#!/bin/sh
OUTPUTDIR="MCMC_test"
MAINSCRIPT="Data_set_up.py"
DATA="friction_conductance_power.pkl"
SUBFOLDER=/mnt/iusers01/support/mbexegc2/scratch/Abaqus_bayesian_matflow/parallel_core_pymc
BASIS_FUNCS=2
PLASTIC_START=4
PLASTIC_END=-1
TRAIN_VALIDATE_RATIO="0.8"



mkdir ~/scratch/"${OUTPUTDIR}"

cp "${SUBFOLDER}/${MAINSCRIPT}" ~/scratch/"${OUTPUTDIR}"
cd ~/scratch/"${OUTPUTDIR}"
python "${SUBFOLDER}/${MAINSCRIPT}" "${BASIS_FUNCS}" "${PLASTIC_START}" "${PLASTIC_END}" "${TRAIN_VALIDATE_RATIO}"
