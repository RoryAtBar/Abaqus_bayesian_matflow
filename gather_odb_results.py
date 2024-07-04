import json
from pathlib import Path

def gather_odb_results(combined_odb_results: list):
    """Gather results from each combination of parameters from a sequence"""
#    results = []
#    for result in odb_results:
#      results.append(result)
#      # combine list of dictionaries into list of dictionary data
#      # eventually want a dict of lists
    with open("odb_results_table.json", "w") as f:
        json.dump(combined_odb_results, f, indent=2)
    summary_table = combined_odb_results 
    return summary_table
