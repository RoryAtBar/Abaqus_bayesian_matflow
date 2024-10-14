import json
import numpy as np

with open('odb_results_table.json', 'r') as file:
    data = json.load(file)

def clean_data(raw_data:list[str], first_index: int, last_index: int):
    """Remove junk data points and convert json string data into numpy array"""
    # There are some lines of junk/empty data, which varies depending on the data variables
    # The lines of interest are specified by first_index and last_index
    clean_data = []
    if last_index < 0:
        last_index = len(raw_data[0]) + last_index
    raw_data_last_index = len(raw_data[0]) - 1
    n_junk_rows = raw_data_last_index - last_index + first_index
    for n, row in enumerate(raw_data):
        if not (isinstance(row, float)):
            prof = np.zeros((len(row) - n_junk_rows, 2))
            for k, string in enumerate(row[first_index:last_index + 1]):
                nums = string.split()
                prof[k, 0] = float(nums[0])
                prof[k, 1] = float(nums[1])
            clean_data.append(prof)
        else:
            clean_data.append([])
    return clean_data


temp_profile = clean_data(data["nt11"], 4, 44)
peeq_profile = clean_data(data["peeq_vals"], 3, -5)
barelling_profile = clean_data(data["barrelling_profile"], 3, -5)
force_profile1 = clean_data(data["force_vals1"], 3, -5)
force_profile2 = clean_data(data["force_vals2"], 3, -5)
force_profile = [fp1 + fp2 for fp1, fp2 in zip(force_profile1, force_profile2)]