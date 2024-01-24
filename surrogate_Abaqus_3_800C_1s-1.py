import numpy as np
import pandas as pd
import subprocess
import os
import time

subprocess.run(['pwd'])
#Version 2 uses a modified version of read_Force_PEEQ_NT11_barrelling based on a macro
#Version 3 imports a different odb for each value of platen conductance
def call_abaqus_with_new_params(list_of_material_coefficients, original_inp_file, output_directory,count):
    #This function calls the 'generate_input_file' function to create a model with randomised parameters
    #It's purpose is is call abaqus with the compression test model, then call abaqus cae to interpret the
    #output data base. The sub process generates a text file with force values vs time step called 'force_output.txt'
    #which is then read and returned as the output of this function
    #The list_of_material_coefficients is a numpy array of randomised multiples of material data, original_inp_file is a 
    #string locating the inp file, and output directory is where the .odb file is to be placed
    #print('abaqus function called')
    output_filename='Doesitwork'
    input_file = generate_input_file(list_of_material_coefficients, original_inp_file)
    Run_Abaqus = subprocess.run(['abq2022','job=sub_script_check', 'input='+original_inp_file, 'interactive'])
    read_odb_into_text_file = subprocess.run(['abq2022','cae', 'noGUI=read_Force_PEEQ_NT11_barrelling_forcemac.py'])
    subprocess.run(['ls','-l'])
    with open('Force_sample_set1.rpt','r') as f:
        force_vals1=f.read().split('\n')[:-1]
    f.close()
    with open('Force_sample_set2.rpt','r') as f:
        force_vals2=f.read().split('\n')[:-1]
    f.close()
    #compression_force = np.zeros(len(force_vals))
    with open('PEEQ_output.rpt','r') as f:
        PEEQ_vals=f.read().split('\n')[:-1]
    f.close()
    with open('outer_sample_xcoords.rpt','r') as f:
        barrelling_profile=f.read().split('\n')[:-1]
    f.close()
    with open('NT11.rpt','r') as f:
        NT11=f.read().split('\n')[:-1]
    f.close()    
    #for i, force in enumerate(force_vals):
    #    compression_force[i] = float(force)
    #print('abaqus function completed')
    file_count = str(count)
    results_df.at[count,'Force Results1'] = force_vals1
    results_df.at[count,'Force Results2'] = force_vals2
    results_df.at[count,'Barrelling Profile'] = barrelling_profile
    results_df.at[count,'PEEQ Results'] = PEEQ_vals
    results_df.at[count,'Temperature profile'] = NT11
    subprocess.run(['mv','PEEQ_output.rpt',f'PEEQ_output{file_count}.rpt'])
    subprocess.run(['mv','outer_sample_xcoords.rpt',f'outer_sample_xcoords{file_count}.rpt'])
    subprocess.run(['mv','NT11.rpt',f'NT11_{file_count}.rpt'])
    subprocess.run(['mv','Force_sample_set1.rpt',f'Force_sample_set1{file_count}.rpt'])
    subprocess.run(['mv','Force_sample_set2.rpt',f'Force_sample_set2{file_count}.rpt'])
    if np.random.rand() > 0.9:
        subprocess.run(['mv','sub_script_check.odb',f'{file_count}.odb'])
        subprocess.run(['cp','friction_conductance.inp',f'{file_count}.inp'])
    subprocess.run(['rm','sub_script_check*'])
    #return compression_force

def modify_friction(inp_text, coefficient_of_friction):
    new_text = inp_text
    replacement_line = ' '+str(coefficient_of_friction)+','
    for n,line in enumerate(inp_text):
        if line == '*Surface Interaction, name=FRICTION':
            new_text[n+3] = replacement_line
    return new_text

def modify_platen_conductance(inp_text, platen_conductance):
    new_text = inp_text
    replacement_line = str(platen_conductance[0])+',    0.'
    next_line = (len(replacement_line)- len('0., 0.001'))*' '
    for n,line in enumerate(inp_text):
        if line == '*Surface Interaction, name=SAMPLE_PLATEN_CONDUCTANCE':
            new_text[n+3] = replacement_line
            new_text[n+4] = next_line + '0., 0.001'
    curworkdir = os.getcwd()        
    for n,line in enumerate(new_text):
        if line == "** Name: Predefined Field-2   Type: Temperature":
            new_text[n+1] = f"*Initial Conditions, type=TEMPERATURE, file={curworkdir}/{platen_conductance[1]}, step=1, inc=0, interpolate"
    return new_text

def modify_power(inp_text, p):
    new_text = inp_text
    for n,line in enumerate(inp_text):
        f = 175102289.
        s = 1276614982.14433
        if line == '*Amplitude, name=POWER':
            new_text[n+1] = f'             0., {p},             0.5, {p},             2.5, {p},              3., {p}'
            new_text[n+2] = f'             5., {p},             5.5, {p},             7.5, {p},              8., {p}'
            new_text[n+3] = f'            10., {p},            10.5, {p},            12.5, {p},             13., {p}'
            new_text[n+4] = f'            15., {p},            15.5, {p},            17.5, {p},             18., {p}'
            new_text[n+5] = f'            20., {p},            20.5, {p},            22.5, {p},             23., {p}'
            new_text[n+6] = f'            25., {p},            25.5, {p},            27.5, {p},             28., {p}'
            new_text[n+7] = f'            30., {p},            30.5, {p},            32.5, {p},             33., {p}'
            new_text[n+8] = f'            35., {p},            35.5, {p},            37.5, {p},             38., {p}'
            new_text[n+9] = f'            40., {p},            40.5, {p},            42.5, {p},             43., {p}'
            new_text[n+10] = f'            45., {p},            45.5, {p},            47.5, {p},             48., {p}'
            new_text[n+11] = f'            50., {p}'
    return new_text            

#Main function for generating inp files. This function organises the above functions.
#It takes the location of the inp file, and a seperate file with a plasticity data lookup table in the 
#same format as the inp file, reads them and feeds them through all of the above functions in order.
def generate_input_file(parameters, inp_file):
    inp_data = open(inp_file).read().split('\n')
    inp_data = modify_friction(inp_data, parameters[0])
    inp_data = modify_platen_conductance(inp_data, parameters[1])
    inp_data = modify_power(inp_data, parameters[2])
    #print(new_plasticity_data==plasticity_data_table)
    #print(list_of_material_coefficients)
    new_inp = ''
    for line in inp_data:
        new_inp += line + '\n'
    with open(inp_file,'w') as f:
        f.write(new_inp)
    f.close()
    return new_inp

def model_sensitivity_lib_format(Friction_Coefficient,Sample_Platen_Thermal_Conductivity):
    #This function is to recycle existing code into the sensitivity library
    parameters = [Friction_Coefficient,Sample_Platen_Thermal_Conductivity]
    inp_file = '900C_001s-1_step_19.inp'
    output_directory = ''
    comp_force = call_abaqus_with_new_params(parameters, inp_file, output_directory)
    return comp_force


starting_time = time.time()
friction = [0.01, 0.99] #[min, max]
conductivity = [1000, 2000] #[min, max
power = [-2e7,1.6e7]

samples_per_param = 15
no_samples = samples_per_param**2
#results = {'power input': np.zeros(no_samples) 'coefficient of friction':np.zeros(no_samples), 'platen sample interface conductance':np.zeros(no_samples),'Force Results':np.zeros(no_samples)}
results = {'Friction':np.zeros(no_samples), 'Conductance':np.zeros(no_samples), 'Power':np.zeros(no_samples), 'Force Results1':np.zeros(no_samples), 'Force Results2':np.zeros(no_samples),'PEEQ Results':np.zeros(no_samples), 'Barrelling Profile':np.zeros(no_samples),'Temperature profile':np.zeros(no_samples)}
results_df = pd.DataFrame(results)
results_df['Force Results1'] = results_df['Force Results1'].astype(object)
results_df['Force Results2'] = results_df['Force Results2'].astype(object)
results_df['Barrelling Profile'] = results_df['Barrelling Profile'].astype(object)
results_df['PEEQ Results'] = results_df['PEEQ Results'].astype(object)
results_df['Temperature profile'] = results_df['Temperature profile'].astype(object)


subprocess.run(['rm','sub_script_check*'])
power_vals = np.linspace(power[0],power[1],samples_per_param)
conductivity_vals = [[0,"800nocondC_heatup22.odb"],
                     [100,"80010C_heatup22.odb"],
                     [200,"80020C_heatup22.odb"],
                     [300,"80030C_heatup22.odb"],
                     [400,"80040C_heatup22.odb"],
                     [500,"80050C_heatup22.odb"],
                     [600,"80060C_heatup22.odb"],
                     [700,"80070C_heatup22.odb"],
                     [800,"80080C_heatup22.odb"],
                     [900,"80090C_heatup22.odb"],
                     [1000,"800100C_heatup22.odb"],
                     [1100,"800110C_heatup22.odb"],
                     [1200,"800120C_heatup22.odb"],
                     [1300,"800130C_heatup22.odb"],
                     [1400,"800140C_heatup22.odb"],
                     [1500,"800150C_heatup22.odb"]]
                     #[1600,"800160C_heatup22.odb"]
                     #[1700,"800170C_heatup22.odb"],
                     #[1800,"800180C_heatup22.odb"],
                     #[1900,"800190C_heatup22.odb"],
                     #[2000,"800200C_heatup22.odb"]]
friction_vals = np.linspace(friction[0],friction[1],samples_per_param)
count = 0
output_directory = ''
original_inp_file = 'friction_conductance.inp'
for c in conductivity_vals:
    for F in friction_vals:
        p = 0
        list_of_material_coefficients = [F,c,p]
            #force_results = call_abaqus_with_new_params(list_of_material_coefficients, original_inp_file, output_directory,count)
        results_df.loc[count,'Power'] = p
        results_df.loc[count,'Friction'] = F
        results_df.loc[count,'Conductance'] = c[0]
        call_abaqus_with_new_params(list_of_material_coefficients, original_inp_file, output_directory,count)
        count += 1
        perc = 100* count/no_samples
        remaining = (100 - perc)/100
        time_elapsed = time.time() - starting_time
        tot_est_time = time_elapsed*100/perc 
        print(f'Sample {count} of {no_samples} complete. {perc} percent complete')
        print('estimated time remaining:', (tot_est_time*remaining)/3600, 'hours')
#for sample in range(no_samples):
#    friction_coeff = np.random.normal(friction_vals[0],friction_vals[1])
#    conductivity = np.random.normal(conductivity_vals[0],conductivity_vals[1])
#    force_results = model_sensitivity_lib_format(friction_coeff,conductivity)
#    results_df.loc[sample,'coefficient of friction'] = friction_coeff
#    results_df.loc[sample,'platen sample interface conductance'] = conductivity
#    results_df['Force Results'][sample] = force_results
#    subprocess.run(['mv','sub_script_check.odb', str(sample)+'.odb'])

results_df.to_pickle('friction_conductance_power.pkl')
