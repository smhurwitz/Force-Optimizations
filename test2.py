from physics_tools import *
UUID = "6266c8d4bb25499b899d86e9e3dd2ee2"
qfm_surf, _, _ = qfm(UUID)
desc_eq = surf_to_desc(qfm_surf)
print("tests passed!")