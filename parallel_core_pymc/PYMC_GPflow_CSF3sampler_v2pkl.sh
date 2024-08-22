#!/bin/bash --login

# SGE options (whose lines must begin with #$)

#$ -cwd               # Run the job in the current directory 
source activate gpflow_compatibilityAug24
#$ -t 1-30


#!/bin/sh
OUTPUTFILE="MCMCv2_30_chain_100_000"
MAINSCRIPT="PYMC_GPflow_CSF3sampler_v2pkl.py"
XNORMALISER="Quad_plast_800C001_X.pkl"
YNORMALISER="Quad_plast_800C001_Y.pkl"
IMPORTTEMPFILE="Abaqus_5_min_heatup/"
IMPORTTEMPDB="800150C_heatup22.odb"
MODEL="Quad_plast_pickle_test.pkl"
MODELFOLDER="/mnt/iusers01/jf01/w10944rb/saved_models/Chapt3/Quad_plast/800C_001s-1/"
SUBFOLDER="/mnt/iusers01/jf01/w10944rb/GP_surrogate_pymc/parallel_core_pymc/"
VARDAT="Ti64_Flowcurves/"
BASIS_FUNCS=2
PLASTIC_START=4
PLASTIC_END=15
BURN_IN=100000
SAMPLE=100000
COEFFS="46973769.8 322884.150"


mkdir ~/scratch/"${OUTPUTFILE}"
cd "${MODELFOLDER}"

cp "${MODEL}" /mnt/iusers01/jf01/w10944rb/scratch/"${OUTPUTFILE}"
cp "${XNORMALISER}" /mnt/iusers01/jf01/w10944rb/scratch/"${OUTPUTFILE}"
cp "${YNORMALISER}" /mnt/iusers01/jf01/w10944rb/scratch/"${OUTPUTFILE}"
cp -r "${VARDAT}" /mnt/iusers01/jf01/w10944rb/scratch/"${OUTPUTFILE}"
cd ~/scratch/"${OUTPUTFILE}"
python "${SUBFOLDER}${MAINSCRIPT}" "${SGE_TASK_ID}" "${MODEL}" "${XNORMALISER}" "${YNORMALISER}" "${VARDAT}" "${BASIS_FUNCS}" "${PLASTIC_START}" "${PLASTIC_END}" "${BURN_IN}" "${SAMPLE}" "${COEFFS}"
