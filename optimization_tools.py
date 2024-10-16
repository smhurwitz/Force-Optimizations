import glob
import json
import numpy as np
import os
import pandas as pd
import time
import uuid
from scipy.optimize import minimize
from simsopt._core.optimizable import load
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


def continuation(N=10000, dx=0.05, 
                 INPUT_DIR="./output/QA/with-force-penalty/1/pareto/", 
                 OUTPUT_DIR="./output/QA/with-force-penalty/2/optimizations/",
                 INPUT_FILE="./inputs/input.LandremanPaul2021_QA",
                 MAXITER=14000):
    """Performs a continuation method on a set of previous optimizations."""
    # Read in input optimizations
    results = glob.glob(f"{INPUT_DIR}*/results.json")
    df = pd.DataFrame()
    for results_file in results:
        with open(results_file, "r") as f:
            data = json.load(f)
        # Wrap lists in another list
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = [value]
        df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)  

    def perturb(init, parameter):
        "Perturbs a parameter from an initial optimization."
        return rand(1-dx, 1+dx) * init[parameter].iloc[0]

    for i in range(N):
        init = df.sample()
        
        # FIXED PARAMETERS
        ARCLENGTH_WEIGHT        = 0.01
        UUID_init_from          = init['UUID'].iloc[0]
        ncoils                  = init['ncoils'].iloc[0]
        order                   = init['order'].iloc[0]
        R1                      = init['R1'].iloc[0]

        # RANDOM PARAMETERS
        CURVATURE_THRESHOLD     = perturb(init, 'max_κ_threshold')
        MSC_THRESHOLD           = perturb(init, 'msc_threshold')
        CS_THRESHOLD            = perturb(init, 'cs_threshold')
        CC_THRESHOLD            = perturb(init, 'cc_threshold')
        FORCE_THRESHOLD         = perturb(init, 'force_threshold')
        LENGTH_TARGET           = perturb(init, 'length_target')

        LENGTH_WEIGHT           = perturb(init, 'length_weight')
        CURVATURE_WEIGHT        = perturb(init, 'max_κ_weight')
        MSC_WEIGHT              = perturb(init, 'msc_weight')
        CS_WEIGHT               = perturb(init, 'cs_weight')
        CC_WEIGHT               = perturb(init, 'cc_weight')
        FORCE_WEIGHT            = perturb(init, 'force_weight')

        # RUNNING THE JOBS
        res, results, coils = optimization(
            OUTPUT_DIR,
            INPUT_FILE,
            R1,
            order,
            ncoils,
            UUID_init_from,
            LENGTH_TARGET, 
            LENGTH_WEIGHT,
            CURVATURE_THRESHOLD, 
            CURVATURE_WEIGHT,
            MSC_THRESHOLD, 
            MSC_WEIGHT,
            CC_THRESHOLD,
            CC_WEIGHT,
            CS_THRESHOLD,
            CS_WEIGHT,
            FORCE_THRESHOLD,
            FORCE_WEIGHT,
            ARCLENGTH_WEIGHT,
            dx=dx,
            MAXITER=MAXITER)
        
        print(f"Job {i+1} completed with UUID={results['UUID']}")
            

def initial_optimizations(N=10000, with_force=True, MAXITER=14000,
                         OUTPUT_DIR="./output/QA/with-force-penalty/1/optimizations/",
                         INPUT_FILE="./inputs/input.LandremanPaul2021_QA", 
                         ncoils=5):
    
    """Performs a set of initial optimizations by scanning over parameters."""
    for i in range(N):
        # FIXED PARAMETERS
        ARCLENGTH_WEIGHT        = 0.01
        UUID_init_from          = None  # not starting from prev. optimization
        order                   = 16

        # RANDOM PARAMETERS
        R1                      = rand(0.35, 0.75)
        CURVATURE_THRESHOLD     = rand(5, 12)
        MSC_THRESHOLD           = rand(4,6)
        CS_THRESHOLD            = rand(0.166, 0.300)
        CC_THRESHOLD            = rand(0.083, 0.120)
        FORCE_THRESHOLD         = rand(0, 5e+04)
        LENGTH_TARGET           = rand(4.9,5.0)

        LENGTH_WEIGHT           = 10.0 ** rand(-4, -2)
        CURVATURE_WEIGHT        = 10.0 ** rand(-9, -5)
        MSC_WEIGHT              = 10.0 ** rand(-7, -3)
        CS_WEIGHT               = 10.0 ** rand(-1, 4)
        CC_WEIGHT               = 10.0 ** rand(2, 5)

        if with_force:
            FORCE_WEIGHT        = 10.0 ** rand(-14, -9)
        else:
            FORCE_WEIGHT        = 0

        # RUNNING THE JOBS
        res, results, coils = optimization(
            OUTPUT_DIR,
            INPUT_FILE,
            R1,
            order,
            ncoils,
            UUID_init_from,
            LENGTH_TARGET, 
            LENGTH_WEIGHT,
            CURVATURE_THRESHOLD, 
            CURVATURE_WEIGHT,
            MSC_THRESHOLD, 
            MSC_WEIGHT,
            CC_THRESHOLD,
            CC_WEIGHT,
            CS_THRESHOLD,
            CS_WEIGHT,
            FORCE_THRESHOLD,
            FORCE_WEIGHT,
            ARCLENGTH_WEIGHT,
            MAXITER=MAXITER)
        
        print(f"Job {i+1} completed with UUID={results['UUID']}")


def initial_optimizations_QH(N=10000, with_force=True, MAXITER=14000,
                         OUTPUT_DIR="./output/QA/with-force-penalty/1/optimizations/",
                         INPUT_FILE="./inputs/input.LandremanPaul2021_QH_magwell_R0=1", 
                         ncoils=3):
    
    """Performs a set of initial optimizations by scanning over parameters."""
    for i in range(N):
        # FIXED PARAMETERS
        ARCLENGTH_WEIGHT        = 0.01
        UUID_init_from          = None  # not starting from prev. optimization
        order                   = 16

        # RANDOM PARAMETERS
        R1                      = rand(0.35, 0.6)
        CURVATURE_THRESHOLD     = rand(5, 12)
        MSC_THRESHOLD           = rand(4,6)
        CS_THRESHOLD            = rand(0.166, 0.300)
        CC_THRESHOLD            = rand(0.083, 0.120)
        FORCE_THRESHOLD         = rand(0, 5e+04)
        LENGTH_TARGET           = rand(4.9,5.0)

        LENGTH_WEIGHT           = 10.0 ** rand(-7, -5)
        CURVATURE_WEIGHT        = 10.0 ** rand(-15, -11)
        MSC_WEIGHT              = 10.0 ** rand(-8, -5)
        CS_WEIGHT               = 10.0 ** rand(-5, 0)
        CC_WEIGHT               = 10.0 ** rand(-3, 0)

        if with_force:
            FORCE_WEIGHT        = 10.0 ** rand(-19, -12)
        else:
            FORCE_WEIGHT        = 0

        # RUNNING THE JOBS
        res, results, coils = optimization(
            OUTPUT_DIR,
            INPUT_FILE,
            R1,
            order,
            ncoils,
            UUID_init_from,
            LENGTH_TARGET, 
            LENGTH_WEIGHT,
            CURVATURE_THRESHOLD, 
            CURVATURE_WEIGHT,
            MSC_THRESHOLD, 
            MSC_WEIGHT,
            CC_THRESHOLD,
            CC_WEIGHT,
            CS_THRESHOLD,
            CS_WEIGHT,
            FORCE_THRESHOLD,
            FORCE_WEIGHT,
            ARCLENGTH_WEIGHT,
            MAXITER=MAXITER)
        
        print(f"Job {i+1} completed with UUID={results['UUID']}")


def optimization(
        OUTPUT_DIR="./output/QA/with-force-penalty/1/optimizations/",
        INPUT_FILE="./inputs/input.LandremanPaul2021_QA",
        R1 = 0.5,
        order = 5,
        ncoils = 5,
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
        ARCLENGTH_WEIGHT=1e-2,
        dx=None,
        MAXITER=14000):
    """Performs a stage II force optimization based on specified criteria. """
    start_time = time.perf_counter()

    # Initialize the boundary magnetic surface:
    nphi = 32
    ntheta = 32
    s = SurfaceRZFourier.from_vmec_input(INPUT_FILE, range="half period", nphi=nphi, ntheta=ntheta)
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

    def initial_base_curves(R0, R1, order, ncoils):
        return create_equally_spaced_curves(
            ncoils,
            nfp,
            stellsym=True,
            R0=R0,
            R1=R1,
            order=order,
    )

    if UUID_init_from is None: 
        base_curves = initial_base_curves(R0, R1, order, ncoils)
        total_current = 3e5
        # Since we know the total sum of currents, we only optimize for ncoils-1
        # currents, and then pick the last one so that they all add up to the correct
        # value.
        base_currents = [Current(total_current / ncoils * 1e-5) * 1e5 for _ in range(ncoils-1)]
        # Above, the factors of 1e-5 and 1e5 are included so the current
        # degrees of freedom are O(1) rather than ~ MA.  The optimization
        # algorithm may not perform well if the dofs are scaled badly.
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]

        coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
        base_coils = coils[:ncoils]
        curves = [c.curve for c in coils]
        
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
    else: 
        path = glob.glob(f"../**/{UUID_init_from}/biot_savart.json", recursive=True)[0]
        print("2")
        bs = load(path)
        coils = bs.coils
        curves = [c.curve for c in coils]
        base_coils = coils[:ncoils]
        base_curves = [base_coils[i].curve for i in range(ncoils)]
        base_currents = [base_coils[i].current for i in range(ncoils)]
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
    OUTPUT_DIR = OUTPUT_DIR + UUID + "/"  # Directory for output
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    # EXPORT VTKS

    forces = []
    for c in coils:
        force = np.linalg.norm(coil_force(c, coils, regularization_circ(0.05)), axis=1)
        force = np.append(force, force[0])
        forces = np.concatenate([forces, force])
    pointData_forces = {"F": forces}
    curves_to_vtk(curves, OUTPUT_DIR + "curves_opt", close=True, extra_data=pointData_forces)

    bs_big = BiotSavart(coils)
    bs_big.set_points(surf_big.gamma().reshape((-1, 3)))
    pointData = {
        "B_N": np.sum(
            bs_big.B().reshape((nphi_big, ntheta_big, 3)) * surf_big.unitnormal(),
            axis=2,
        )[:, :, None]
    }
    surf_big.to_vtk(OUTPUT_DIR + "surf_opt", extra_data=pointData)


    # SAVE DATA TO JSON

    BdotN      = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    mean_AbsB  = np.mean(bs.AbsB())
    max_forces = [np.max(np.linalg.norm(coil_force(c, coils, regularization_circ(0.05)), axis=1)) for c in base_coils]
    min_forces = [np.min(np.linalg.norm(coil_force(c, coils, regularization_circ(0.05)), axis=1)) for c in base_coils]
    RMS_forces = [np.sqrt(np.mean(np.square(np.linalg.norm(coil_force(c, coils, regularization_circ(0.05)), axis=1)))) for c in base_coils]
    results = {
        "nfp":                      nfp,
        "ncoils":                   int(ncoils),
        "order":                    int(order),
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
        # "linking_number":           LinkingNumber(curves).J(), #TODO: UNCOMMENT!!!!!!!
        "lengths":                  [float(J.J()) for J in Jls],
        "max_length":               max(float(J.J()) for J in Jls),
        "max_κ":                    [np.max(c.kappa()) for c in base_curves],
        "max_max_κ":                max(np.max(c.kappa()) for c in base_curves),
        "MSCs":                     [float(J.J()) for J in Jmscs],
        "max_MSC":                  max(float(J.J()) for J in Jmscs),
        "max_forces":               [float(f) for f in max_forces],
        "max_max_force":            max(float(f) for f in max_forces),
        "min_forces":               [float(f) for f in min_forces],
        "min_min_force":            min(float(f) for f in min_forces),
        "RMS_forces":               [float(f) for f in RMS_forces],
        "mean_RMS_force":            float(np.mean([f for f in RMS_forces])),
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
        "eval_time":                time.perf_counter() - start_time,
        "dx":                       dx
    }

    with open(OUTPUT_DIR + "results.json", "w") as outfile:
        json.dump(results , outfile, indent=2)
    bs.save(OUTPUT_DIR + f"biot_savart.json")  # save the optimized coil shapes and currents

    return res, results, base_coils
    

def rand(min, max):
    """Generate a random float between min and max."""
    return np.random.rand() * (max - min) + min