import json
from pathlib import Path

def gather_odb_outputs(force_sample_set_1: Path, force_sample_set_2: Path,
                       nt11: Path, outer_sample_xcoords: Path, peeq_output: Path):
    """Combine results files from processing odb file"""
    with open(force_sample_set_1,'r') as f:
        force_vals1 = f.read().split('\n')[:-1]
    with open(force_sample_set_2,'r') as f:
        force_vals2 = f.read().split('\n')[:-1]
    with open(peeq_output,'r') as f:
        peeq_vals = f.read().split('\n')[:-1]
    with open(outer_sample_xcoords,'r') as f:
        barrelling_profile = f.read().split('\n')[:-1]
    with open(nt11,'r') as f:
        nt11 = f.read().split('\n')[:-1]

    data = {"force_vals1": force_vals1,
            "force_vals2": force_vals2,
            "barrelling_profile": barrelling_profile,
            "peeq_vals": peeq_vals,
            "nt11": nt11}
    with open("odb_results.json", "w") as f:
        json.dump(data, f, indent=2)

    return data
