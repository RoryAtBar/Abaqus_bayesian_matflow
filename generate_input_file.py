import os

def generate_input_file(path: str, friction: float, conductance: list, power: float, inp_file: str):
    """Create abaqus input file using material coefficients to modify existing inp_file"""
    with open(inp_file, 'r') as f:
        inp_data = open(inp_file).read().split('\n')
    inp_data = modify_friction(inp_data, friction)
    inp_data = modify_platen_conductance(inp_data, conductance)
    inp_data = modify_power(inp_data, power)
    new_inp = ''
    for line in inp_data:
        new_inp += line + '\n'
    with open(path,'w') as f:
        f.write(new_inp)
    f.close()


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
            new_text[n+1] = f"*Initial Conditions, type=TEMPERATURE, file={platen_conductance[1]}, step=1, inc=0, interpolate"
    return new_text
