#!/usr/bin/env python

import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import scipy
import shutil
import simsopt
from matplotlib import colors
from mayavi import mlab
from numbers import Number
from paretoset import paretoset
from simsopt._core.optimizable import load
from simsopt.field import (InterpolatedField, 
                           SurfaceClassifier, 
                           particles_to_vtk,
                           compute_fieldlines, 
                           LevelsetStoppingCriterion, 
                           plot_poincare_data,
                           BiotSavart)
from simsopt.field.force import coil_force, LpCurveForce, self_force
from simsopt.field.selffield import regularization_circ
from simsopt.geo import SurfaceRZFourier, plot
from simsopt.util import comm_world


### DATA ANALYSIS #############################################################
def get_dfs(INPUT_DIR='./output/QA/1/optimizations/', OUTPUT_DIR=None):
    """Returns DataFrames for the raw, filtered, and Pareto data."""
    ### STEP 1: Import raw data
    inputs=f"{INPUT_DIR}*/results.json"
    results = glob.glob(inputs)
    dfs = []
    for results_file in results:
        with open(results_file, "r") as f:
            data = json.load(f)
        # Wrap lists in another list
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = [value]
        dfs.append(pd.DataFrame(data))
    df = pd.concat(dfs, ignore_index=True) 
    
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

    # Copy pareto fronts to a separate folder
    if OUTPUT_DIR is not None:
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for UUID in df_pareto['UUID']:
            SOURCE_DIR = glob.glob(f"./**/{UUID}/", recursive=True)[0] 
            DEST_DIR = f"{OUTPUT_DIR}{UUID}/"
            shutil.copytree(SOURCE_DIR, DEST_DIR)

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


### PHYSICS STUFF #############################################################
def poincare(UUID, OUT_DIR='./output/QA/1/poincare/', INPUT_FILE="./inputs/input.LandremanPaul2021_QA",
             nfieldlines=10, tmax_fl=20000, degree=4, R0_min=1.2125346, 
             R0_max=1.295, interpolate=True, debug=False):
    """Compute Poincare plots."""
 
    # Directory for output
    OUT_DIR = OUT_DIR + UUID + "/"

    # Load in the boundary surface:
    surf = SurfaceRZFourier.from_vmec_input(INPUT_FILE, nphi=200, ntheta=30, range="full torus")
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


def qfm(UUID):
    """Generated quadratic flux minimizing surfaces, largely copied from
    https://github.com/hiddenSymmetries/simsopt/blob/master/examples/1_Simple/qfm.py"""

    path = glob.glob("./**/" + UUID + "/biot_savart.json", recursive=True)[0]
    with open(path, "r") as f:
        data = json.load(f)
        # Wrap lists in another list
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = [value]
        df = pd.DataFrame(data)
    nfp = df["nfp"]
    
    path = glob.glob("./**/" + UUID + "/biot_savart.json", recursive=True)[0]
    bs = load(path)

    mpol = 5 # Maximum poloidal mode number included.
    ntor = 5 # Maximum toroidal mode number included, divided by nfp.
    stellsym = True
    constraint_weight = 1e0  

    phis = np.linspace(0, 1/nfp, 25, endpoint=False)
    thetas = np.linspace(0, 1, 25, endpoint=False)
    s = SurfaceRZFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis,
        quadpoints_theta=thetas)
    s.fit_to_curve(ma, 0.2, flip_theta=True)


### PLOTTING ##################################################################
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


def plot_coils(UUID, surf_file=None, surf_color="white", coil_color="force", 
               arrow=None, select_coil=None):    
    fig = mlab.figure(bgcolor=(1,1,1), fgcolor=(0,0,0), size=(1200,1200))
    
    path = glob.glob("./**/" + UUID + "/biot_savart.json", recursive=True)[0]
    bs = load(path)
    coils = bs.coils
    
    i=0
    for c in coils:
        def close(data): return np.concatenate((data, [data[0]]))
        gamma = c.curve.gamma()
        x = close(gamma[:, 0])
        y = close(gamma[:, 1])
        z = close(gamma[:, 2])

        tube_radius = 0.015
        if coil_color == "force":
            force = np.linalg.norm(coil_force(c, coils, regularization_circ(0.05)), axis=1)
            force = np.append(force, force[0])
            mlab.plot3d(x, y, z, force, tube_radius=tube_radius)
        else:
            mlab.plot3d(x, y, z, tube_radius=tube_radius)  
        
        
        if arrow is not None and (select_coil is not None and select_coil==i):
            def skip(data, npts=50): return data[::round(len(data) / npts)]
        
            if arrow == "force":
                f = coil_force(c, coils, regularization_circ(0.05))
            elif arrow == "mutual":
                gammadash = c.curve.gammadash()
                gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
                tangent = gammadash / gammadash_norm
                mutual_coils = [coil for coil in coils if coil is not c]
                mutual_field = BiotSavart(mutual_coils).set_points(c.curve.gamma()).B()
                mutualforce = np.cross(c.current.get_value() * tangent, mutual_field)
                f = mutualforce
            elif arrow == "self":
                f = self_force(c, regularization_circ(0.05))
            else:
                f=None

            u = skip(f[:,0])
            v = skip(f[:,1])
            w = skip(f[:,2])
            x = skip(gamma[:, 0])
            y = skip(gamma[:, 1])
            z = skip(gamma[:, 2])
            mlab.quiver3d(x,y,z,u,v,w, line_width=2.0, scale_mode="none",scale_factor=0.125)  
        
        i += 1  


    if surf_file is not None:
        s = SurfaceRZFourier.from_vmec_input(surf_file, range="full torus", nphi=32, ntheta=32)
        gamma = s.gamma()
        gamma = np.concatenate((gamma, gamma[:, :1, :]), axis=1)
        gamma = np.concatenate((gamma, gamma[:1, :, :]), axis=0)

        rgb = colors.to_rgba(surf_color)[0:3]
        mlab.mesh(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], color=rgb)


    mlab.colorbar(orientation="vertical", title="Force [N/m]")
    return fig

    
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