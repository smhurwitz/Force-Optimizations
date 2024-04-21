# This script is run after "stage_2_scan.py" has generated some optimized coils.
# This script reads the results.json files in the subdirectories, plots the
# distribution of results, filters out unacceptable runs, and prints out runs that
# are Pareto-optimal.

import glob
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from paretoset import paretoset

# Initialize an empty DataFrame
df = pd.DataFrame()

results = glob.glob("./output/initial/*/results.json")
for results_file in results:
    with open(results_file, "r") as f:
        data = json.load(f)

    # Wrap lists in another list
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = [value]

    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

#########################################################
# Here you can define criteria to filter out the most interesting runs.
#########################################################

margin_up   = 1.05
margin_low  = 0.95
allowed_length  = 5.00 * margin_up
allowed_κ       = 12.00 * margin_up
allowed_MSC     = 6.00 * margin_up
allowed_CC_dist = 0.083 * margin_low
allowed_CS_dist = 0.166 * margin_low
allowed_arclen_var = 1e-2

succeeded = df["linking_number"] < 0.1
succeeded = np.logical_and(succeeded, df["max_length"] < allowed_length)
succeeded = np.logical_and(succeeded, df["max_max_κ"] < allowed_κ)
succeeded = np.logical_and(succeeded, df["max_MSC"] < allowed_MSC)
succeeded = np.logical_and(succeeded, df["coil_coil_distance"] > allowed_CC_dist)
succeeded = np.logical_and(succeeded, df["coil_surface_distance"] > allowed_CS_dist)
succeeded = np.logical_and(succeeded, df["max_arclength_variance"] < allowed_arclen_var)

#########################################################
# End of filtering criteria
#########################################################

df_filtered = df[succeeded]

pareto_mask = paretoset(df_filtered[["BdotN", "max_max_force"]], sense=[min, min])
df_pareto = df_filtered[pareto_mask]

print("Best Pareto-optimal results:")
print(
    (df_pareto[
        [
            "UUID",
            "BdotN",
            "max_max_force",
            "max_max_κ",
            "max_MSC",
            "max_length",
            "coil_coil_distance",
            "coil_surface_distance"
        ]
    ]).to_markdown()
)

# print("Directory names only:")
# for dirname in df_pareto["directory"]:
#     print(dirname)

#########################################################
# Plotting
#########################################################

plt.figure(0, figsize=(14.5, 8))
plt.rc("font", size=8)
nrows = 1
ncols = 2
markersize = 5

subplot_index = 1
plt.subplot(nrows, ncols, subplot_index)
subplot_index += 1
plt.scatter(df["BdotN"], df["max_max_force"], c=df["max_length"], s=1)
plt.scatter(
    df_filtered["BdotN"],
    df_filtered["max_max_force"],
    c=df_filtered["max_length"],
    s=markersize,
)
plt.scatter(
    df_pareto["BdotN"], df_pareto["max_max_force"], c=df_pareto["max_length"], marker="+"
)
plt.xlabel("Bnormal")
plt.ylabel("Max Force")
plt.xscale("log")
plt.colorbar(label="max_length")

plt.subplot(nrows, ncols, subplot_index)
subplot_index += 1
plt.scatter(df["BdotN"], df["max_max_force"], c=df["order"], s=1)
plt.scatter(
    df_filtered["BdotN"],
    df_filtered["max_max_force"],
    c=df_filtered["order"],
    s=markersize,
)
plt.scatter(
    df_pareto["BdotN"], df_pareto["max_max_force"], c=df_pareto["order"], marker="+"
)
plt.xlabel("Bnormal")
plt.ylabel("Max Force")
plt.xscale("log")
plt.colorbar(label="order")


plt.figure(2, figsize=(14.5, 8))
plt.rc("font", size=8)
nrows = 1
ncols = 2
markersize = 5

subplot_index = 1
plt.subplot(nrows, ncols, subplot_index)
subplot_index += 1
plt.scatter(
    df_filtered["order"],
    df_filtered["eval_time"],
    s=markersize,
)
plt.xlabel("Fourier order")
plt.ylabel("Evaluation time")


plt.figure(1, figsize=(14.5, 8))
nrows = 4
ncols = 5

subplot_index = 1
def plot_2d_hist(field, log=False):
    global subplot_index
    plt.subplot(nrows, ncols, subplot_index)
    subplot_index += 1
    nbins = 20
    if log:
        data = df[field]
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), nbins)
    else:
        bins = nbins
    n,bins,patchs = plt.hist(df[field], bins=bins, label="before filtering")
    plt.hist(df_filtered[field], bins=bins, alpha=1, label="after filtering")
    plt.xlabel(field)
    plt.legend(loc=0, fontsize=6)
    if log:
        plt.xscale("log")


# 2nd entry of each tuple is True if the field should be plotted on a log x-scale.
fields = (
    ("R1", False),
    ("order", False),
    ("max_max_force", False),
    ("max_length", False),
    ("max_max_κ", False),
    ("max_MSC", False),
    ("coil_coil_distance", False),
    ("coil_surface_distance", False),
    ("length_target", False),
    # ("force_threshold", False),
    ("max_κ_threshold", False),
    ("msc_threshold", False),
    ("cc_threshold", False),
    ("cs_threshold", False),
    ("length_weight", True),
    ("max_κ_weight", True),
    ("msc_weight", True),
    ("cc_weight", True),
    ("cs_weight", True),
    ("force_weight", True),
    ("linking_number", False)
)

for field, log in fields:
    plot_2d_hist(field, log)

plt.figtext(0.5, 0.995, os.path.abspath(__file__), ha="center", va="top", fontsize=6)
plt.tight_layout()
plt.show()