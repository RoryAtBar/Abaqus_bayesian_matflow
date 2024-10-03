#!/bin/bash --login

# SGE options (whose lines must begin with #$)

#$ -cwd               # Run the job in the current directory 
source ../test_venv/bin/activate
#$ -hold_jid PYMC_GPflow_CSF3sampler_step2.sh 



#!/bin/sh
OUTPUTDIR=~/scratch/MCMC_test_single_out_autokernel
MAINSCRIPT="PYMC_GPflow_CSF3sampler_step3.py"
SUBFOLDER=/mnt/iusers01/support/mbexegc2/scratch/Abaqus_bayesian_matflow/parallel_core_pymc
COMBINEDTRACEFILENAME="Quad_plast_25_chain"


cd ${OUTPUTDIR}
python ${SUBFOLDER}/${MAINSCRIPT} "${COMBINEDTRACEFILENAME}"
