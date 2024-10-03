#!/bin/bash --login

# SGE options (whose lines must begin with #$)

#$ -cwd               # Run the job in the current directory
#$ -l mem512

source .venv/bin/activate

#!/bin/sh
SCRATCH=${HOME}/scratch
OUTPUTDIR=${SCRATCH}/MCMC_GPsurrgt_800C_1s-1_cond0-1500_20000_chain
MAINSCRIPT=${PWD}/MCMC_800C_1s-1.py
MODELDATA="friction_conductance_power.pkl"
MODELDATADIR=${SCRATCH}/SFCP_800C_1s-1_importedodbs_cond0-1500
TESTDATADIR=${PWD}
TESTDATA="800C_1s-1_csv.csv"

mkdir ${OUTPUTDIR}
cp ${MODELDATADIR}/${MODELDATA} ${OUTPUTDIR}
cp ${TESTDATADIR}/${TESTDATA} ${OUTPUTDIR}
cd $OUTPUTDIR
python ${MAINSCRIPT}
