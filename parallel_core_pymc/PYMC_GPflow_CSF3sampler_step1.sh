#!/bin/bash --login

# SGE options (whose lines must begin with #$)

#$ -cwd               # Run the job in the current directory 
source activate PYMC_GPFlow



#!/bin/sh
OUTPUTFILE="MCMC_test_single_out_autokernel"
MAINSCRIPT="PYMC_GPflow_CSF3sampler_step1.py"
XNORMALISER="Quad_plast_800C001_X.pkl"
YNORMALISER="Quad_plast_800C001_Y.pkl"
XDATA="X_train.pkl"
YDATA="Y_train.pkl"
XVALIDATE="X_validate.pkl"
YVALIDATE="Y_validate.pkl"
MODEL_NAME="Quadplast_timesteps_5-15"
MODELFOLDER="/mnt/iusers01/jf01/w10944rb/saved_models/Chapt3/Quad_plast/800C_001s-1/"
SUBFOLDER="/mnt/iusers01/jf01/w10944rb/GP_surrogate_pymc/parallel_core_pymc/"
VARDAT="Ti64_Flowcurves/"
KERNEL="Auto"
BASIS_FUNCS=2
PLASTIC_START=4
PLASTIC_END=15



mkdir ~/scratch/"${OUTPUTFILE}"
cd "${MODELFOLDER}"
cp "${MODELFOLDER}${XDATA}" ~/scratch/"${OUTPUTFILE}"
cp "${MODELFOLDER}${YDATA}" ~/scratch/"${OUTPUTFILE}"
cp "${MODELFOLDER}${XVALIDATE}" ~/scratch/"${OUTPUTFILE}"
cp "${MODELFOLDER}${YVALIDATE}" ~/scratch/"${OUTPUTFILE}"
cp "${MODELFOLDER}${XNORMALISER}" ~/scratch/"${OUTPUTFILE}"
cp "${MODELFOLDER}${YNORMALISER}" ~/scratch/"${OUTPUTFILE}"
cp -r "${VARDAT}" ~/scratch/"${OUTPUTFILE}"
cp "${SUBFOLDER}${MAINSCRIPT}" ~/scratch/"${OUTPUTFILE}"
cd ~/scratch/"${OUTPUTFILE}"
python "${SUBFOLDER}${MAINSCRIPT}" "${XDATA}" "${YDATA}" "${XVALIDATE}" "${YVALIDATE}" "${XNORMALISER}" "${YNORMALISER}" "${BASIS_FUNCS}" "${KERNEL}" "${MODEL_NAME}"
