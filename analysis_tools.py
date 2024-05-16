#!/usr/bin/env python

import glob
import json
import matplotlib.pyplot as plt
from numbers import Number
import numpy as np
import os
import pandas as pd
import plotly.express as px
import scipy
import simsopt
from paretoset import paretoset
from simsopt.field import (InterpolatedField, 
                           SurfaceClassifier, 
                           particles_to_vtk,
                           compute_fieldlines, 
                           LevelsetStoppingCriterion, 
                           plot_poincare_data)
from simsopt.geo import SurfaceRZFourier
from simsopt.util import proc0_print, comm_world


def get_dfs(inputs="./output/1/data/*/results.json", OUTPUT_DIR='1'):
    """Returns DataFrames for the raw, filtered, and Pareto data."""
    ### STEP 1: Import raw data
    df = pd.DataFrame()
    results = glob.glob(inputs)
    for results_file in results:
        with open(results_file, "r") as f:
            data = json.load(f)
        # Wrap lists in another list
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = [value]
        df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)  
    df = df.query('ncoils==5')

    ### STEP 2: Filter the data
    margin_up   = 1.05
    margin_low  = 0.95

    df_filtered = df.query(
        # ENGINEERING CONSTRAINTS:
        f"max_length < {5 * margin_up}"
        f"and max_max_κ < {12.00 * margin_up}"
        f"and max_MSC < {6.00 * margin_up}"
        f"and coil_coil_distance > {0.083 * margin_low}"
        f"and coil_surface_distance > {0.166 * margin_low}"
        f"and mean_AbsB > 0.22" #prevent coils from becoming detached from LCFS
        # FILTERING OUT BAD/UNNECESSARY DATA:
        f"and max_arclength_variance < 1e-2"
        f"and coil_surface_distance < 0.375"
        f"and coil_coil_distance < 0.14"
        f"and max_length > 4.0"
        f"and normalized_BdotN < {0.5 * 1e-2}"
        f"and max_max_force<30000"
    )   

    ### STEP 3: Generate Pareto front and export UUIDs as .txt
    pareto_mask = paretoset(df_filtered[["normalized_BdotN", "max_max_force"]], sense=[min, min])
    df_pareto = df_filtered[pareto_mask]

    if OUTPUT_DIR is not None:
        np.savetxt(f"./output/{OUTPUT_DIR}/pareto.txt", df_pareto['UUID'].values, fmt='%s')

    ### Return statement
    return df, df_filtered, df_pareto


def parameter_correlations(df, sort_by='normalized_BdotN'):
    df_sorted = df.sort_values(by=[sort_by])
    columns_to_drop = ['BdotN', 'gradient_norm', 'Jf', 'JF', 'mean_AbsB',
                       'max_arclength_variance', 'iterations', 'linking_number',
                       'function_evaluations', 'success', 'arclength_weight',
                       'R0', 'ntheta', 'nphi', 'ncoils', 'nfp','MSCs',
                       'max_forces', 'arclength_variances', 'max_κ',
                       'UUID_init', 'message', 'coil_currents', 'UUID',
                       'lengths', 'eval_time', 'order']
    df_sorted = df_sorted.drop(columns=columns_to_drop)

    df_correlation = pd.DataFrame({'Parameter': [], 'R': [], 'P': []})
    for i in range(len(df_sorted.columns)):
        series_name = df_sorted.columns[i]
        series = np.array(df_sorted)[:, i]
        bdotn_series = np.array(df_sorted[sort_by])
        if isinstance(series[0], Number):
            series = series.astype(float)
            result = scipy.stats.linregress(bdotn_series, series)
            r = result.rvalue
            p = result.pvalue
            df_row = {'Parameter': series_name, 'R': r, 'P': p}
            df_correlation = df_correlation._append(df_row, ignore_index = True)
        else: 
            df_row = {'Parameter': series_name, 'R': -np.inf, 'P': -np.inf}
            df_correlation = df_correlation._append(df_row, ignore_index = True)

    return df_correlation.sort_values(by=['R'], ascending=False)


def pareto_plt(df, df_filtered, df_pareto):
    """Creates a color-map plot of the Pareto front and filtered points. """
    markersize = 5
    n_pareto = df_pareto.shape[0]
    n_filtered = df_filtered.shape[0] - n_pareto
    
    fig = plt.figure(figsize=(6.5, 8))
    plt.rc("font", size=13)
    norm = plt.Normalize(min(df_filtered['coil_surface_distance']), max(df_filtered['coil_surface_distance']))
    plt.scatter(
        df_filtered["normalized_BdotN"],
        df_filtered["max_max_force"],
        c=df_filtered["coil_surface_distance"],
        s=markersize,
        label=f'all optimizations, N={n_filtered}',
        norm=norm
    )
    plt.scatter(
        df_pareto["normalized_BdotN"], 
        df_pareto["max_max_force"], 
        c=df_pareto["coil_surface_distance"], 
        marker="+",
        label=f'Pareto front, N={n_pareto}',
        norm=norm,
    )
    plt.xlabel(r'$\langle|\mathbf{B}\cdot\mathbf{n}|\rangle/\langle B \rangle$ [unitless]')
    plt.ylabel("max force [N/m]")
    plt.xlim(0.7 * min(df_filtered["normalized_BdotN"]), max(df_filtered["normalized_BdotN"]))
    plt.ylim(8500, 25000)
    plt.xscale("log")
    plt.colorbar(label="coil-surface distance [m]")
    plt.legend(loc='upper right', fontsize='11')
    plt.title('Pareto Front')
    return fig


def pareto_interactive_plt(df, color='coil_surface_distance'):
    """Creates an interactive plot of the Pareto front."""
    fig = px.scatter(
        df, 
        x="normalized_BdotN", 
        y="max_max_force", 
        color=color,
        log_x=True,
        width=500, 
        height=400,
        title="Initial Force Optimizations",
        hover_data={
            'UUID':True,
            'max_max_force':':.2e',
            'coil_surface_distance':':.2f',
            'coil_coil_distance':':.3f',
            'length_target':':.2f',
            'force_threshold':':.2e',
            'cc_threshold':':.2e',
            'cs_threshold':':.2e',
            'length_weight':':.2e',
            'msc_weight':':.2e',
            'cc_weight':':.2e',
            'cs_weight':':.2e',
            'force_weight':':.2e',
            'normalized_BdotN':':.2e',
            'max_MSC':':.2e',
            'max_max_κ':':.2e'
            }
        )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white'
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        tickformat='.1e',
        dtick=0.25
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        tickformat = "000"
    )
    return fig


def poincare(UUID, OUT_DIR='1/poincare', nfieldlines=10, tmax_fl=20000, degree=4, 
            R0_min=1.2125346, R0_max=1.295, interpolate=True, debug=False):
    """Compute Poincare plots."""
 
    # Directory for output
    OUT_DIR = f'./output/{OUT_DIR}/{UUID}/'
    print(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load in the boundary surface:
    filename = 'inputs/input.LandremanPaul2021_QA'
    surf = SurfaceRZFourier.from_vmec_input(filename, nphi=200, ntheta=30, range="full torus")
    nfp = surf.nfp

    # Load in the optimized coils from stage_two_optimization.py:
    coils_filename = glob.glob(f"./**/{UUID}/biot_savart.json", recursive=True)[0] 
    bs = simsopt.load(coils_filename)

    sc_fieldline = SurfaceClassifier(surf, h=0.03, p=2)
    if debug: 
        surf.to_vtk(OUT_DIR + 'surface')
        sc_fieldline.to_vtk(OUT_DIR + 'levelset', h=0.02)

    def trace_fieldlines(bfield, label):
        R0 = np.linspace(R0_min, R0_max, nfieldlines)
        Z0 = np.zeros(nfieldlines)
        phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=tmax_fl, tol=1e-16, comm=comm_world,
            phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        if debug: particles_to_vtk(fieldlines_tys, OUT_DIR + f'fieldlines_{label}')
        plot_poincare_data(fieldlines_phi_hits, phis, OUT_DIR + f'poincare_fieldline_{label}.png', dpi=150)
        image = plt.imread(OUT_DIR + f'poincare_fieldline_{label}.png')
        return image


    # Bounds for the interpolated magnetic field chosen so that the surface is
    # entirely contained in it
    n = 20
    rs = np.linalg.norm(surf.gamma()[:, :, 0:2], axis=2)
    zs = surf.gamma()[:, :, 2]
    rrange = (np.min(rs), np.max(rs), n)
    phirange = (0, 2*np.pi/nfp, n*2)
    zrange = (0, np.max(zs), n//2)

    def skip(rs, phis, zs):
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc_fieldline.evaluate_rphiz(rphiz)
        skip = list((dists < -0.05).flatten())
        return skip

    bsh = InterpolatedField(
        bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True, skip=skip
    )
    bsh.set_points(surf.gamma().reshape((-1, 3)))
    bs.set_points(surf.gamma().reshape((-1, 3)))
    image = trace_fieldlines(bsh, 'bsh') if(interpolate) else trace_fieldlines(bs, 'bs')
    return image


def success_plt(df, df_filtered):
    fig = plt.figure(1, figsize=(14.5, 11))
    nrows = 5
    ncols = 5

    def plot_2d_hist(field, log=False, subplot_index=0):
        plt.subplot(nrows, ncols, subplot_index)
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
        ("force_threshold", False),
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
        ("linking_number", False),
        ('ncoils', False)
    )

    i=1
    for field, log in fields:
        plot_2d_hist(field, log, i)
        i += 1

    plt.tight_layout()
    return fig
