#!/bin/bash --login

# SGE options (whose lines must begin with #$)

#$ -cwd               # Run the job in the current directory 
source ../test_venv/bin/activate


#!/bin/sh
OUTPUTDIR=~/scratch/MCMC_test_single_out_autokernel
MAINSCRIPT="PYMC_GPflow_CSF3sampler_step1.py"
XNORMALISER="Quad_plast_800C001_X.pkl"
YNORMALISER="Quad_plast_800C001_Y.pkl"
XDATA="X_train.pkl"
YDATA="Y_train.pkl"
XVALIDATE="X_validate.pkl"
YVALIDATE="Y_validate.pkl"
MODEL_NAME="Quadplast_timesteps_5-15"
MODELFOLDER=/mnt/iusers01/support/mbexegc2/scratch/MCMC_test
SUBFOLDER=/mnt/iusers01/support/mbexegc2/scratch/Abaqus_bayesian_matflow/parallel_core_pymc
VARDAT=/mnt/iusers01/support/mbexegc2/scratch/Abaqus_bayesian_matflow/parallel_core_pymc/Ti64_Flowcurves
KERNEL="Auto"
BASIS_FUNCS=2
PLASTIC_START=4
PLASTIC_END=15



mkdir "${OUTPUTDIR}"
cd "${MODELFOLDER}"
cp "${MODELFOLDER}/${XDATA}" "${OUTPUTDIR}"
cp "${MODELFOLDER}/${YDATA}" "${OUTPUTDIR}"
cp "${MODELFOLDER}/${XVALIDATE}" "${OUTPUTDIR}"
cp "${MODELFOLDER}/${YVALIDATE}" "${OUTPUTDIR}"
cp "${MODELFOLDER}/${XNORMALISER}" "${OUTPUTDIR}"
cp "${MODELFOLDER}/${YNORMALISER}" "${OUTPUTDIR}"
cp -r "${VARDAT}/" "${OUTPUTDIR}"
cp "${SUBFOLDER}/${MAINSCRIPT}" "${OUTPUTDIR}"
cd "${OUTPUTDIR}"
python "${SUBFOLDER}/${MAINSCRIPT}" "${XDATA}" "${YDATA}" "${XVALIDATE}" "${YVALIDATE}" "${XNORMALISER}" "${YNORMALISER}" "${BASIS_FUNCS}" "${KERNEL}" "${MODEL_NAME}"
