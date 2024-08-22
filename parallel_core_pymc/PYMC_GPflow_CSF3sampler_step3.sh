#!/bin/bash --login

# SGE options (whose lines must begin with #$)

#$ -cwd               # Run the job in the current directory 
source activate PYMC_GPFlow
#$ -hold_jid PYMC_GPflow_CSF3sampler_step2.sh 



#!/bin/sh
OUTPUTFILE="MCMC_test_single_out_autokernel"
MAINSCRIPT="PYMC_GPflow_CSF3sampler_step3.py"
SUBFOLDER="/mnt/iusers01/jf01/w10944rb/GP_surrogate_pymc/parallel_core_pymc/"
COMBINEDTRACEFILENAME="Quad_plast_25_chain"


cd ~/scratch/"${OUTPUTFILE}"
python "${SUBFOLDER}${MAINSCRIPT}" "${COMBINEDTRACEFILENAME}" 
