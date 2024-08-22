#!/bin/bash --login

# SGE options (whose lines must begin with #$)

#$ -cwd               # Run the job in the current directory 
source activate PYMC_GPFlow
#$ -l mem2000
#$ -hold_jid PYMC_GPflow_CSF3sampler_step1.sh 
#$ -t 1-25


#!/bin/sh
OUTPUTFILE="MCMC_test_single_out_autokernel"
MAINSCRIPT="PYMC_GPflow_CSF3sampler_v2pkl.py"
XNORMALISER="Quad_plast_800C001_X.pkl"
YNORMALISER="Quad_plast_800C001_Y.pkl"

MODEL="Quadplast_timesteps_5-15.pkl"
SUBFOLDER="/mnt/iusers01/jf01/w10944rb/GP_surrogate_pymc/parallel_core_pymc/"
VARDAT="Ti64_Flowcurves/"
BASIS_FUNCS=2
PLASTIC_START=4
PLASTIC_END=15
BURN_IN=100000
SAMPLE=200000
COEFFS="46973769.8 322884.150"


cd ~/scratch/"${OUTPUTFILE}"
python "${SUBFOLDER}${MAINSCRIPT}" "${SGE_TASK_ID}" "${MODEL}" "${XNORMALISER}" "${YNORMALISER}" "${VARDAT}" "${BASIS_FUNCS}" "${PLASTIC_START}" "${PLASTIC_END}" "${BURN_IN}" "${SAMPLE}" "${COEFFS}"
