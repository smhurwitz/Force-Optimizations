import glob
import matplotlib.pyplot as plt
import numpy as np
import simsopt
from desc.grid import LinearGrid
from desc.geometry import FourierRZToroidalSurface
from desc.equilibrium import Equilibrium
from desc.profiles import PowerSeriesProfile
from desc.vmec_utils import ptolemy_identity_fwd
from simsopt._core.optimizable import load
from simsopt.field import (InterpolatedField, 
                           SurfaceClassifier, 
                           particles_to_vtk,
                           compute_fieldlines, 
                           LevelsetStoppingCriterion, 
                           plot_poincare_data)
from simsopt.geo import QfmResidual, QfmSurface, SurfaceRZFourier, Volume
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


def qfm(UUID, INPUT_FILE="./inputs/input.LandremanPaul2021_QA", mpol=5, ntor=5):
    """Generated quadratic flux minimizing surfaces, adapted from
    https://github.com/hiddenSymmetries/simsopt/blob/master/examples/1_Simple/qfm.py"""

    # We start with an initial guess that is just the stage I LCFS
    s = SurfaceRZFourier.from_vmec_input(INPUT_FILE, range="full torus", nphi=32, ntheta=32)
    
    # Optimize at fixed volume:
    path = glob.glob("./**/" + UUID + "/biot_savart.json", recursive=True)[0]
    bs = load(path)
    qfm = QfmResidual(s, bs)
    qfm.J()
    vol = Volume(s)
    vol_target = vol.J()
    qfm_surface = QfmSurface(bs, s, vol, vol_target)

    constraint_weight = 1e0  
    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=1000,
                                                            constraint_weight=constraint_weight)
    vol_err = abs((s.volume()-vol_target)/vol_target)
    residual = np.linalg.norm(qfm.J())
    return qfm_surface.surface, vol_err, residual


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
        current=PowerSeriesProfile(),
        pressure=PowerSeriesProfile(),
    )

    # Check that the desc surface matches the input surface.
    # Grid resolution for testing the surfaces match:
    ntheta = 50
    nphi = 39
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
        # Flip direction of theta:
        desc_arr = np.vstack((desc_arr[:1, :], np.flipud(desc_arr[1:, :])))
        np.testing.assert_allclose(simsopt_data, desc_arr, atol=1e-14)

    compare_simsopt_desc(gamma[:, :, 0].T, data["X"])
    compare_simsopt_desc(gamma[:, :, 1].T, data["Y"])
    compare_simsopt_desc(gamma[:, :, 2].T, data["Z"])

    # If tests are passed:
    return eq