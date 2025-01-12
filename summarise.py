import arviz as az
from pathlib import Path


def summarise(dependency_hack_iteration: list):
    print(f"Iterations: {dependency_hack_iteration}")
    netcdf_files = (
        Path.cwd().parents[3].glob("artifacts/tasks/task_4_sample_model/e_0/r_*/Idata_chain.nc")
    )

    traces = []
    for file in netcdf_files:
        trace = az.from_netcdf(file)
        traces.append(trace)

    concat_traces = az.concat(traces, dim="chain")
    az.to_netcdf(concat_traces, "combined_trace.nc")

    ax = az.plot_trace(concat_traces)
    fig = ax.ravel()[0].figure
    fig.savefig("Trace_plot.png")
    ax = az.plot_posterior(concat_traces)
    fig = ax.ravel()[0].figure
    fig.savefig("Posterior.png")

    summary_table = az.summary(concat_traces)
    print(summary_table)
