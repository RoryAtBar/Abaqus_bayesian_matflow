#!/bin/bash --login

# SGE options (whose lines must begin with #$)

#$ -cwd               # Run the job in the current directory 
source .venv/bin/activate
module load apps/binapps/abaqus/2022

#!/bin/sh
SCRATCH=${HOME}/scratch
DATADIR=${SCRATCH}/Abaqus_bayesian_matflow
OUTPUTFILE="SFCP_800C_1s-1_importedodbs_cond0-1500"
MAINSCRIPT="surrogate_Abaqus_3_800C_1s-1.py"
IMPORTTEMPFOLER="Abaqus_5_min_heatup"
TEMPERATUREODB="800"
TEMPERATUREODBITERATION='22'
MODEL="800C_1s-1_setup.inp"
PLASTICITYDATA="Patryk_mat_data.txt"
ODBREADER="read_Force_PEEQ_NT11_barrelling_forcemac.py"
mkdir "${HOME}/scratch/${OUTPUTFILE}"
for i in "nocond" 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200
do
    IMPORTNAME="${TEMPERATUREODB}${i}C_heatup${TEMPERATUREODBITERATION}.odb"
    cp "${DATADIR}/${IMPORTTEMPFOLER}/${IMPORTNAME}" "${SCRATCH}/${OUTPUTFILE}"
done
cp "${DATADIR}/${ODBREADER}" "${SCRATCH}/${OUTPUTFILE}"
cp "${DATADIR}/${PLASTICITYDATA}"  "${SCRATCH}/${OUTPUFILE}"
cp "${DATADIR}/${MODEL}"  "${SCRATCH}/${OUTPUTFILE}/friction_conductance.inp"
cd ${SCRATCH}/${OUTPUTFILE}
python "${DATADIR}/${MAINSCRIPT}"  
