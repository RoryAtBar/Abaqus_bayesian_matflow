#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:29:47 2024

@author: w10944rb
"""

import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sys import argv
from glob import glob

data_filename = argv[1]

trace_files = glob("Idata_chain_*")
traces = []

for i in trace_files:
    trace = az.from_netcdf(i)
    traces.append(trace)

concat_traces = az.concat(traces, dim='chain')
az.to_netcdf(concat_traces, f"{data_filename}.nc")

ax = az.plot_trace(concat_traces)
fig = ax.ravel()[0].figure
fig.savefig("Trace_plot.png")
ax = az.plot_posterior(concat_traces)
fig = ax.ravel()[0].figure
fig.savefig("Posterior.png")

summary_table = az.summary(concat_traces)
print(summary_table)
