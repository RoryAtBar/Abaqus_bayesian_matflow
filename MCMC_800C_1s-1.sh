#!/bin/bash --login

# SGE options (whose lines must begin with #$)

#$ -cwd               # Run the job in the current directory 
#$ -l mem256 
source activate PYMC_GPFlow


#!/bin/sh
OUTPUTFILE="MCMC_GPsurrgt_800C_1s-1_cond0-1500_20000_chain/"
MAINSCRIPT="MCMC_800C_1s-1.py"
MODELDATA="friction_conductance_power.pkl"
MODELDATAFOLDER="scratch/SFCP_800C_1s-1_importedodbs_cond0-1500/"
SUBFOLDER="GP_surrogate_pymc/"
TESTDATAFOLDER= "Patryk_Force_time_data/"
TESTDATA="800C_1s-1_csv.csv"
mkdir "/mnt/iusers01/jf01/w10944rb/scratch/${OUTPUTFILE}"
cp "/mnt/iusers01/jf01/w10944rb/${MODELDATAFOLDER}${MODELDATA}" "/mnt/iusers01/jf01/w10944rb/scratch/${OUTPUTFILE}"
cp "/mnt/iusers01/jf01/w10944rb/${TESTDATAFOLDER}${TESTDATA}" "/mnt/iusers01/jf01/w10944rb/scratch/${OUTPUTFILE}"
cd $OUTPUTFILE
python "/mnt/iusers01/jf01/w10944rb/${SUBFOLDER}${MAINSCRIPT}"  
