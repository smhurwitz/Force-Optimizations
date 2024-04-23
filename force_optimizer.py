#!/usr/bin/env python

"""
Example script for the force metric in a stage-two coil optimization
"""
import os
import json
import numpy as np
import uuid
import time
from pathlib import Path
from scipy.optimize import minimize
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.geo import (
    CurveLength, 
    CurveCurveDistance, 
    CurveSurfaceDistance, 
    MeanSquaredCurvature, 
    LpCurveCurvature, 
    ArclengthVariation,
    curves_to_vtk,
    create_equally_spaced_curves,
    SurfaceRZFourier, 
    LinkingNumber)
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from simsopt.field.force import coil_force, LpCurveForce
from simsopt.field.selffield import regularization_circ
from simsopt.util import in_github_actions


###############################################################################
# FIXED PARAMETERS
###############################################################################

# File for the desired boundary magnetic surface:
filename = 'input.LandremanPaul2021_QA'

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 14000

# Initialize the boundary magnetic surface:
nphi = 32
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
nfp = s.nfp
R0 = s.get_rc(0, 0)

# Create a copy of the surface that is closed in theta and phi, and covers the
# full torus toroidally. This is nice for visualization.
nphi_big = nphi * 2 * nfp + 1
ntheta_big = ntheta + 1
quadpoints_theta = np.linspace(0, 1, ntheta_big)
quadpoints_phi = np.linspace(0, 1, nphi_big)
surf_big = SurfaceRZFourier(
    dofs=s.dofs,
    nfp=nfp,
    mpol=s.mpol,
    ntor=s.ntor,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta,
)


###############################################################################
# GENERATE COILS
###############################################################################


def initial_base_curves(R0, R1, order, ncoils):
    return create_equally_spaced_curves(
        ncoils,
        nfp,
        stellsym=True,
        R0=R0,
        R1=R1,
        order=order,
    )


###############################################################################
# OPTIMIZATION METHOD
###############################################################################


def optimization(
        OUTPUT_DIR="initial",
        R1 = 0.5,
        order = 5,
        ncoils = 4,
        UUID_init_from=None,
        LENGTH_TARGET=5.00, 
        LENGTH_WEIGHT=1e-03,
        CURVATURE_THRESHOLD=12.0, 
        CURVATURE_WEIGHT=1e-08,
        MSC_THRESHOLD=5.00, 
        MSC_WEIGHT=1e-04,
        CC_THRESHOLD=0.083,
        CC_WEIGHT=1e+03,
        CS_THRESHOLD=0.166,
        CS_WEIGHT=1e+03,
        FORCE_THRESHOLD=2e+04,
        FORCE_WEIGHT=1e-10,
        ARCLENGTH_WEIGHT=1e-2):
    
    start_time = time.perf_counter()

    if UUID_init_from is None: base_curves = initial_base_curves(R0, R1, order, ncoils)
    else: base_curves = None  # TODO: for initializing from previous optimization
    base_currents = [Current(1.0) * (1e5) for i in range(ncoils)]
    base_currents[0].fix_all()  # avoid minimizer setting all currents to zero
    coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
    base_coils = coils[:ncoils]
    curves = [c.curve for c in coils]
    
    bs = BiotSavart(coils)
    bs.set_points(s.gamma().reshape((-1, 3)))

    ###########################################################################
    ## FORM THE OBJECTIVE FUNCTION ############################################

    # Define the individual terms objective function:
    Jf = SquaredFlux(s, bs)
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    Jforce = [LpCurveForce(c, coils, regularization_circ(0.05), p=2, threshold=FORCE_THRESHOLD) for c in base_coils]
    Jals = [ArclengthVariation(c) for c in base_curves]

    # Form the total objective function.
    JF = Jf \
        + LENGTH_WEIGHT * sum(QuadraticPenalty(Jl, LENGTH_TARGET, "max") for Jl in Jls) \
        + CC_WEIGHT * Jccdist \
        + CS_WEIGHT * Jcsdist \
        + CURVATURE_WEIGHT * sum(Jcs) \
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
        + FORCE_WEIGHT * sum(Jforce) \
        + ARCLENGTH_WEIGHT * sum(Jals)
    
    ###########################################################################
    ## PERFORM OPTIMIZATION ###################################################
    
    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        return J, grad
    
    res = minimize(fun, JF.x, jac=True, method='L-BFGS-B', 
                   options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
    JF.x = res.x

    ###########################################################################
    ## EXPORT OPTIMIZATION DATA ###############################################

    # MAKE DIRECTORY FOR EXPORT

    UUID = uuid.uuid4().hex  # unique id for each optimization
    OUT_DIR = f"./output/{OUTPUT_DIR}/{UUID}/"  # Directory for output
    os.makedirs(OUT_DIR, exist_ok=True)


    # EXPORT VTKS

    forces = []
    for c in coils:
        force = np.linalg.norm(coil_force(c, coils, regularization_circ(0.05)), axis=1)
        force = np.append(force, force[0])
        forces = np.concatenate([forces, force])
    pointData_forces = {"F": forces}
    curves_to_vtk(curves, OUT_DIR + "curves_opt", close=True, extra_data=pointData_forces)

    bs_big = BiotSavart(coils)
    bs_big.set_points(surf_big.gamma().reshape((-1, 3)))
    pointData = {
        "B_N": np.sum(
            bs_big.B().reshape((nphi_big, ntheta_big, 3)) * surf_big.unitnormal(),
            axis=2,
        )[:, :, None]
    }
    surf_big.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData)


    # SAVE DATA TO JSON

    BdotN       = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    mean_AbsB   = np.mean(bs.AbsB())
    force       = [np.max(np.linalg.norm(coil_force(c, coils, regularization_circ(0.05)), axis=1)) for c in base_coils]

    results = {
        "nfp":                      nfp,
        "ncoils":                   ncoils,
        "order":                    order,
        "nphi":                     nphi,
        "ntheta":                   ntheta,
        "R0":                       R0,  # if initialized from circles, else None
        "R1":                       R1,  # if initialized from circles, else None
        "UUID_init":                UUID_init_from,  # if initialized from optimization, else None
        "length_target":            LENGTH_TARGET,
        "length_weight":            LENGTH_WEIGHT,
        "max_κ_threshold":          CURVATURE_THRESHOLD,
        "max_κ_weight":             CURVATURE_WEIGHT,
        "msc_threshold":            MSC_THRESHOLD,
        "msc_weight":               MSC_WEIGHT,
        "cc_threshold":             CC_THRESHOLD,
        "cc_weight":                CC_WEIGHT,
        "cs_threshold":             CS_THRESHOLD,
        "cs_weight":                CS_WEIGHT,
        "force_threshold":          FORCE_THRESHOLD,
        "force_weight":             FORCE_WEIGHT,
        "arclength_weight":         ARCLENGTH_WEIGHT,
        "JF":                       float(JF.J()),
        "Jf":                       float(Jf.J()),
        "gradient_norm":            np.linalg.norm(JF.dJ()),
        "linking_number":           LinkingNumber(curves).J(),
        "lengths":                  [float(J.J()) for J in Jls],
        "max_length":               max(float(J.J()) for J in Jls),
        "max_κ":                    [np.max(c.kappa()) for c in base_curves],
        "max_max_κ":                max(np.max(c.kappa()) for c in base_curves),
        "MSCs":                     [float(J.J()) for J in Jmscs],
        "max_MSC":                  max(float(J.J()) for J in Jmscs),
        "max_forces":               [float(f) for f in force],
        "max_max_force":            max(float(f) for f in force),
        "arclength_variances":      [float(J.J()) for J in Jals],
        "max_arclength_variance":   max(float(J.J()) for J in Jals),
        "BdotN":                    BdotN,
        "mean_AbsB":                mean_AbsB,
        "normalized_BdotN":         BdotN/mean_AbsB,
        "coil_coil_distance":       Jccdist.shortest_distance(),
        "coil_surface_distance":    Jcsdist.shortest_distance(),
        "message":                  res.message,
        "success":                  res.success,
        "iterations":               res.nit,
        "function_evaluations":     res.nfev,
        "coil_currents":            [c.get_value() for c in base_currents],
        "UUID":                     UUID,
        "eval_time":                time.perf_counter() - start_time
    }

    with open(OUT_DIR + "results.json", "w") as outfile:
        json.dump(results, outfile, indent=2)
    bs.save(OUT_DIR + f"biot_savart.json")  # save the optimized coil shapes and currents

    return res, results, base_coils
