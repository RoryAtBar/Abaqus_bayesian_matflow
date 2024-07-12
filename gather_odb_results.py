import json
from pathlib import Path

def gather_odb_results(combined_odb_results: list):
    """Gather results from each combination of parameters from a sequence"""
    force_vals1 = []
    force_vals2 = []
    barrelling_profile = []
    peeq_vals = []
    nt11 = []
    for result in combined_odb_results:
      force_vals1.append(result["force_vals1"])
      force_vals2.append(result["force_vals2"])
      barrelling_profile.append(result["barrelling_profile"])
      peeq_vals.append(result["peeq_vals"])
      nt11.append(result["nt11"])
    gathered_odb_results = {"force_vals1": force_vals1,
                            "force_vals2": force_vals2,
                            "barrelling_profile": barrelling_profile,
                            "peeq_vals": peeq_vals,
                            "nt11": nt11}
    with open("odb_results_table.json", "w") as f:
        json.dump(gathered_odb_results, f, indent=2)

    return {"odb_results_table": gathered_odb_results}
