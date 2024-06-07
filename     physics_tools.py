import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simsopt
from simsopt._core.optimizable import load
from simsopt.field import (InterpolatedField, 
                           SurfaceClassifier, 
                           particles_to_vtk,
                           compute_fieldlines, 
                           LevelsetStoppingCriterion, 
                           plot_poincare_data)
from simsopt.geo import SurfaceRZFourier
from simsopt.util import comm_world


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


def qfm(UUID, mpol=5, ntor=5):
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
    
    stellsym = True
    constraint_weight = 1e0  

    # We start with an initial guess that is 
    phis = np.linspace(0, 1/nfp, 25, endpoint=False)
    thetas = np.linspace(0, 1, 25, endpoint=False)
    s = SurfaceRZFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis,
        quadpoints_theta=thetas)
    s.fit_to_curve(ma, 0.2, flip_theta=True)
