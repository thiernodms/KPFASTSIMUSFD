from src.kp.kp import KikPiotrowski
from src.fastsim.fastsim import FastSim
from src.usfd.usfd import USFDWearModel

# Créer les modèles individuels
kp_model = KikPiotrowski(wheel_radius=0.46, rail_radius=0.3)
fastsim_model = FastSim(friction_coefficient=0.3)
usfd_model = USFDWearModel(wheel_material="R8T", rail_material="UIC60 900A")

# Résoudre le problème de contact normal
normal_solution = kp_model.solve_contact_problem(normal_force=100000.0)

# Résoudre le problème de contact tangentiel
tangential_solution = fastsim_model.solve_tangential_problem(
    contact_patch=normal_solution['contact_patch'],
    creepages={'longitudinal': 0.001, 'lateral': 0.0, 'spin': 0.0},
    normal_force=normal_solution['normal_force']
)

# Calculer l'usure
tgamma = usfd_model.calculate_tgamma(
    tangential_force=tangential_solution['tangential_forces'],
    creepage=(0.001, 0.0),
    contact_area=normal_solution['contact_patch']['area'] * 1e6
)
wear_rate = usfd_model.calculate_wear_rate(tgamma)
wear_depth = usfd_model.calculate_wear_depth(tgamma, sliding_distance=1000.0)
