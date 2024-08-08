import desc
import glob
import imageio
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import scipy
import shutil
import simsopt
import subprocess
from desc.grid import LinearGrid
from desc.geometry import FourierRZToroidalSurface
from desc.equilibrium import Equilibrium
from desc.plotting import plot_boozer_surface
from desc.vmec_utils import ptolemy_identity_fwd
from matplotlib import colors
from mayavi import mlab
from numbers import Number
from paretoset import paretoset
from simsopt._core.optimizable import load
from simsopt.field import (BiotSavart,
                           InterpolatedField, 
                           SurfaceClassifier, 
                           particles_to_vtk,
                           compute_fieldlines, 
                           LevelsetStoppingCriterion)
from simsopt.field.force import coil_force, self_force
from simsopt.field.selffield import regularization_circ
from simsopt.geo import QfmResidual, QfmSurface, SurfaceRZFourier, Volume
from simsopt.util import comm_world



###############################################################################
# I) DATA ANALYSIS
###############################################################################

def get_dfs(INPUT_DIR='./output/QA/with-force-penalty/1/optimizations/', OUTPUT_DIR=None):
    """Returns DataFrames for the raw, filtered, and Pareto data."""
    ### STEP 1: Import raw data
    inputs=f"{INPUT_DIR}**/results.json"
    results = glob.glob(inputs, recursive=True)
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
            SOURCE_DIR = glob.glob(f"/**/{UUID}/", recursive=True)[0] 
            DEST_DIR = f"{OUTPUT_DIR}{UUID}/"
            shutil.copytree(SOURCE_DIR, DEST_DIR)

    ### Return statement
    return df, df_filtered, df_pareto


def parameter_correlations(df, sort_by='normalized_BdotN', matrix=False):
    df_sorted = df.sort_values(by=[sort_by])
    columns_to_drop = ['BdotN', 'gradient_norm', 'Jf', 'JF', 'mean_AbsB',
                       'max_arclength_variance', 'iterations', 'linking_number',
                       'function_evaluations', 'success', 'arclength_weight',
                       'R0', 'ntheta', 'nphi', 'ncoils', 'nfp','MSCs',
                       'max_forces', 'arclength_variances', 'max_κ',
                       'UUID_init', 'message', 'coil_currents', 'UUID',
                       'lengths', 'eval_time', 'order', 'dx', 'RMS_forces', 'min_forces']
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

    if(matrix):
        matrix = np.abs(df_sorted.corr())
        plt.figure(figsize=(9,9))
        plt.imshow(matrix, cmap='Blues')
        plt.title("Corrrelation Matrix")
        colorbar = plt.colorbar()
        colorbar.set_label("|Pearson's R|")
        plt.clim(0, 1)
        variables = []
        for i in matrix.columns:
            variables.append(i)

        # Adding labels to the matrix
        plt.xticks(range(len(matrix)), variables, rotation=45, ha='right')
        plt.yticks(range(len(matrix)), variables)

        # Display the plot
        plt.show()
        
    return df_correlation.sort_values(by=['R'], ascending=False)



###############################################################################
# II) PLOTTING
###############################################################################

def create_movies(images, OUT_NAME, OUT_PATH):
    writer = imageio.get_writer(OUT_PATH + OUT_NAME + ".mp4")
    for image in images:
        writer.append_data(image)
    writer.close()
    imageio.mimsave(OUT_PATH + OUT_NAME + ".gif", images, loop=0)

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


def plot_coils(BS_PATH, fig=None, surf_file=None, surf_color="white", coil_color="force", 
               arrow=None, select_coil=None, size=(1200,1200), crange=(0,15000)):   
    if fig is None: 
        fig = mlab.figure(bgcolor=(1,1,1), fgcolor=(0,0,0), size=size)
        mlab.options.offscreen = True
    fig.scene.disable_render = True
    bs = load(BS_PATH) #path to biotsavart.json
    coils = bs.coils
    vmin = crange[0]
    vmax = crange[1]

    def force(c, mode="total"):
        if mode == "total" or mode == "force": 
            return coil_force(c, coils, regularization_circ(0.05))
        elif mode == "mutual":
            gammadash = c.curve.gammadash()
            gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
            tangent = gammadash / gammadash_norm
            mutual_coils = [coil for coil in coils if coil is not c]
            mutual_field = BiotSavart(mutual_coils).set_points(c.curve.gamma()).B()
            mutualforce = np.cross(c.current.get_value() * tangent, mutual_field)
            return mutualforce
        elif mode == "self":
            return self_force(c, regularization_circ(0.05))
        else:
            return None

    for c in coils:
        def close(data): return np.concatenate((data, [data[0]]))
        gamma = c.curve.gamma()
        x = close(gamma[:, 0])
        y = close(gamma[:, 1])
        z = close(gamma[:, 2])

        tube_radius = 0.015
        if coil_color == "force":
            f = np.linalg.norm(force(c), axis=1)
            f = np.append(f, f[0])
            mlab.plot3d(x, y, z, f, tube_radius=tube_radius, vmin=vmin, vmax=vmax)
        else:
            mlab.plot3d(x, y, z, tube_radius=tube_radius, vmin=vmin, vmax=vmax)  


        i=0
        gamma = c.curve.gamma()
        if arrow is not None and (select_coil is None or i==select_coil):
            def skip(data, npts=50): return data[::round(len(data) / npts)]
            f = force(c, arrow)
            u = skip(f[:,0])
            v = skip(f[:,1])
            w = skip(f[:,2])
            x = skip(gamma[:, 0])
            y = skip(gamma[:, 1])
            z = skip(gamma[:, 2])
            scalars = skip(np.linalg.norm(f, axis=1))
            obj = mlab.quiver3d(x, y, z, u, v, w, line_width=2.0, 
                                scale_mode="none", scale_factor=0.125, 
                                vmin=vmin, vmax=vmax, scalars=scalars)  
            obj.glyph.color_mode = 'color_by_scalar'
        i += 1          

    if surf_file is not None:
        s = SurfaceRZFourier.from_vmec_input(surf_file, range="full torus", nphi=32, ntheta=32)
        gamma = s.gamma()
        gamma = np.concatenate((gamma, gamma[:, :1, :]), axis=1)
        gamma = np.concatenate((gamma, gamma[:1, :, :]), axis=0)

        rgb = colors.to_rgba(surf_color)[0:3]
        mlab.mesh(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], color=rgb)

    mlab.colorbar(orientation="vertical", title="Force [N/m]")
    mlab.axes(x_axis_visibility=False, y_axis_visibility=False, z_axis_visibility=False)
    fig.scene.camera.zoom(1.65)
    fig.scene.disable_render = False
    return mlab.screenshot()


def plot_losses(INPUT_PATH, s=0.3):
    # INPUT_PATH points to "confined_fraction.dat"

    with open(INPUT_PATH, "r") as f: ar = np.loadtxt(f)

    time             = ar[:, 0]
    confined_passing = ar[:, 1]
    confined_trapped = ar[:, 2]
    number           = ar[:, 3]
    confined_total   = confined_passing + confined_trapped
    loss_frac        = 1 - confined_total

    fig = plt.figure()
    plt.plot(time, loss_frac)
    plt.title(f"Collisionless particle losses, s={s}, N={int(number[0])}")
    plt.xscale('log')
    plt.xlabel("time [s]")
    plt.ylabel("loss fraction")
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



###############################################################################
# III) PHYSICS STUFF
###############################################################################

def poincare(UUID, OUT_DIR='./output/QA/with-force-penalty/1/poincare/', 
             INPUT_FILE="./inputs/input.LandremanPaul2021_QA", phis=[0.0],
             nfieldlines=10, tmax_fl=20000, degree=4, R0_min=1.2125346, 
             R0_max=1.295, interpolate=True, debug=False):
    """Compute Poincare plots."""
 
    # Directory for output
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load in the boundary surface:
    surf = SurfaceRZFourier.from_vmec_input(INPUT_FILE, nphi=200, ntheta=30, range="full torus")
    nfp = surf.nfp

    # Load in the optimized coils from stage_two_optimization.py:
    coils_filename = glob.glob(f"/**/{UUID}/biot_savart.json", recursive=True)[0] 
    bs = simsopt.load(coils_filename)

    sc_fieldline = SurfaceClassifier(surf, h=0.03, p=2)
    if debug: 
        surf.to_vtk(OUT_DIR + 'surface')
        sc_fieldline.to_vtk(OUT_DIR + 'levelset', h=0.02)

    def trace_fieldlines(bfield, label):
        R0 = np.linspace(R0_min, R0_max, nfieldlines)
        Z0 = np.zeros(nfieldlines)
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=tmax_fl, tol=1e-16, comm=comm_world,
            phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        if debug: particles_to_vtk(fieldlines_tys, OUT_DIR + f'fieldlines_{label}')

        if len(phis) > 1:
            raise NotImplementedError("Not yet implemented!")
            # plot_poincare_data(fieldlines_phi_hits, phis, OUT_DIR + f'poincare_fieldline_{label}.png', dpi=150)

        fig = plt.figure()
        for j in range(len(fieldlines_phi_hits)):
            data_this_phi = fieldlines_phi_hits[j][np.where(fieldlines_phi_hits[j][:, 1] == 0)[0], :]
            if data_this_phi.size == 0:
                continue
            r = np.sqrt(data_this_phi[:, 2]**2+data_this_phi[:, 3]**2)
            plt.scatter(r, data_this_phi[:, 4], marker='o', s=2, linewidths=0)

        return fig

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


def qfm(UUID, INPUT_FILE="./inputs/input.LandremanPaul2021_QA", vol_frac=1.00):
    """Generated quadratic flux minimizing surfaces, adapted from
    https://github.com/hiddenSymmetries/simsopt/blob/master/examples/1_Simple/qfm.py"""

    # We start with an initial guess that is just the stage I LCFS
    s = SurfaceRZFourier.from_vmec_input(INPUT_FILE, range="full torus", nphi=32, ntheta=32)
    
    # Optimize at fixed volume:
    path = glob.glob("/**/" + UUID + "/biot_savart.json", recursive=True)[0]
    bs = load(path)
    qfm = QfmResidual(s, bs)
    qfm.J()
    vol = Volume(s)
    vol_target = vol_frac * vol.J()
    qfm_surface = QfmSurface(bs, s, vol, vol_target)

    constraint_weight = 1e0  
    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=1000,
                                                            constraint_weight=constraint_weight)
    vol_err = abs((s.volume()-vol_target)/vol_target)
    residual = np.linalg.norm(qfm.J())
    # print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    return qfm_surface.surface, vol_err, residual


def run_SIMPLE(UUID, trace_time=1e-1, s=0.3, n_test_part=1024, vmec_name="eq_scaled.nc", 
               BUILD_DIR="/Users/sienahurwitz/Documents/Physics/Codes/SIMPLE/build/",
               RUN_DIR="/Users/sienahurwitz/Documents/Physics/Codes/SIMPLE/build/",
               suppress_output=False):

    # STEP 1: generate the input files and save to run dir
    with open(RUN_DIR + "simple.in", "w") as f: 
        f.write(f"&config\n")
        f.write(f"trace_time = {trace_time}d0\n")
        f.write(f"sbeg = {s}d0\n")
        f.write(f"ntestpart = {n_test_part}\n")
        f.write(f"netcdffile = '{vmec_name}'\n")
        f.write(f"/\n")

    # STEP 2: move the vmec equil to the run dir
    SOURCE = glob.glob(f"/**/{UUID}/eq_scaled.nc", recursive=True)[0]
    DEST = RUN_DIR + "eq_scaled.nc"
    shutil.copy(SOURCE, DEST)

    # STEP 3: run SIMPLE   
    # command = f"cd {BUILD_DIR} && ./simple.x"
    command = BUILD_DIR + "simple.x"
    if suppress_output:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL) 
    else: 
        subprocess.run(command, shell=True, check=True)

    # STEP 4: move inputs and outputs to UUID's folder
    DEST = glob.glob(f"/**/{UUID}/", recursive=True)[0]
    files = ["simple.in", "times_lost.dat", vmec_name]
    for file in files:
        os.remove(RUN_DIR + file) 
    shutil.move(RUN_DIR + "confined_fraction.dat", 
                DEST +f"confined_fraction_s={s:.1E}.dat")  


def surf_to_desc(simsopt_surf, LMN=8):
    """Returns a DESC equilibrium from a simsopt surface, adapted from code 
    written by Matt Landreman. Note that LMN is a DESC resolution parameter. """
    nfp = simsopt_surf.nfp
    ndofs = len(simsopt_surf.x)
    index = int((ndofs + 1) / 2)
    xm = simsopt_surf.m[:index]
    xn = simsopt_surf.n[:index]
    rmnc = simsopt_surf.x[:index]
    zmns = np.concatenate(([0], simsopt_surf.x[index:]))

    rmns = np.zeros_like(rmnc)
    zmnc = np.zeros_like(zmns)

    # Adapted from desc's VMECIO.load around line 126:
    m, n, Rb_lmn = ptolemy_identity_fwd(xm, xn, s=rmns, c=rmnc)
    m, n, Zb_lmn = ptolemy_identity_fwd(xm, xn, s=zmns, c=zmnc)
    surface = np.vstack((np.zeros_like(m), m, n, Rb_lmn, Zb_lmn)).T
    desc_surface = FourierRZToroidalSurface(
        surface[:, 3],
        surface[:, 4],
        surface[:, 1:3].astype(int),
        surface[:, 1:3].astype(int),
        nfp,
        simsopt_surf.stellsym,
        check_orientation=False,
    )

    # To avoid warning message, flip the orientation manually:
    if desc_surface._compute_orientation() == -1:
        desc_surface._flip_orientation()
        assert desc_surface._compute_orientation() == 1

    eq = Equilibrium(
        surface=desc_surface,
        L=LMN,
        M=LMN,
        N=LMN,
        L_grid=2 * LMN,
        M_grid=2 * LMN,
        N_grid=2 * LMN,
        sym=True,
        NFP=nfp,
        Psi=np.pi * simsopt_surf.minor_radius()**2,
        ensure_nested=False,
    )

    # Check that the desc surface matches the input surface.
    # Grid resolution for testing the surfaces match:
    ntheta = len(simsopt_surf.quadpoints_theta)
    nphi = len(simsopt_surf.quadpoints_phi)
    simsopt_surf2 = SurfaceRZFourier(
        mpol=simsopt_surf.mpol,
        ntor=simsopt_surf.ntor,
        nfp=simsopt_surf.nfp,
        stellsym=simsopt_surf.stellsym,
        dofs=simsopt_surf.dofs,
        quadpoints_phi=np.linspace(0, 1 / simsopt_surf.nfp, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False),
    )
    gamma = simsopt_surf2.gamma()

    grid = LinearGrid(
        rho=1,
        theta=np.linspace(0, 2 * np.pi, ntheta, endpoint=False),
        zeta=np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False),
        NFP=nfp,
    )
    data = eq.compute(["X", "Y", "Z"], grid=grid)

    def compare_simsopt_desc(simsopt_data, desc_data):
        desc_arr = desc_data.reshape((ntheta, nphi), order="F")
        desc_arr = np.vstack((desc_arr[:1, :], np.flipud(desc_arr[1:, :])))
        np.testing.assert_allclose(simsopt_data, desc_arr, atol=1e-14)

    compare_simsopt_desc(gamma[:, :, 0].T, data["X"])
    compare_simsopt_desc(gamma[:, :, 1].T, data["Y"])
    compare_simsopt_desc(gamma[:, :, 2].T, data["Z"])

    # If tests are passed:
    # eq.solve()
    desc.continuation.solve_continuation_automatic(eq, verbose=0)
    return eq


###############################################################################
# IV) SANITY CHECKS
###############################################################################

def check_poincare_and_qfm(UUID="6266c8d4bb25499b899d86e9e3dd2ee2", phi=0.1, 
              INPUT_FILE="./inputs/input.LandremanPaul2021_QA"):
    """
    Checks that the target LCFS, QFM surface, and Poincaré plots all agree.
    """

    def get_crosssection(s):
        cs = s.cross_section(phi)
        r = np.sqrt(cs[:, 0] ** 2 + cs[:, 1] ** 2)
        r = np.concatenate((r, [r[0]]))
        z = cs[:, 2]
        z = np.concatenate((z, [z[0]]))
        return r, z

    # show poincaré plots
    poincare(UUID, INPUT_FILE=INPUT_FILE, nfieldlines=10, tmax_fl=20000, 
        degree=4, phis=[phi])
    
    # show target LCFS
    LCFS = SurfaceRZFourier.from_vmec_input(INPUT_FILE, range="full torus", nphi=32, ntheta=32)
    r, z = get_crosssection(LCFS)
    plt.plot(r, z, linewidth=2, c='r', label="target LCFS")

    # show QFMs at various radial distances
    N=2
    for i in range(N):
        vol_frac = 1.0 - 0.02 * i / (N - 1 + 1e-100)
        qfm_surf, _, _ = qfm(UUID, INPUT_FILE=INPUT_FILE, vol_frac=vol_frac)
        r, z = get_crosssection(qfm_surf)
        if i != N - 1:
            plt.plot(r, z, '--', linewidth=2, c='k')
        else:
            plt.plot(r, z, '--', linewidth=2, c='k', label="QFMs")

    plt.legend()
    plt.show()