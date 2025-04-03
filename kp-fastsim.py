from src.utils.uic_profile import UICProfileLoader
from src.integration.kp_fastsim import KPFastSim

# Charger les profils
loader = UICProfileLoader()
wheel_profile = loader.load_wheel_profile("profiles/S1002.wheel")
rail_profile = loader.load_rail_profile("profiles/uic60i00.rail")

# Calculer la géométrie de contact
contact_geometry = loader.calculate_contact_geometry(
    wheel_profile['name'], 
    rail_profile['name'], 
    lateral_displacement=0.0
)

# Extraire les rayons de courbure
wheel_radius = 1.0 / contact_geometry['wheel_curvature'] if contact_geometry['wheel_curvature'] > 0 else 1000.0
rail_radius = 1.0 / contact_geometry['rail_curvature'] if contact_geometry['rail_curvature'] > 0 else 1000.0

# Créer une instance de l'intégration KP-FastSim
kp_fastsim = KPFastSim(
    wheel_radius=wheel_radius,
    rail_radius=rail_radius,
    wheel_youngs_modulus=210e9,
    rail_youngs_modulus=210e9,
    wheel_poisson_ratio=0.3,
    rail_poisson_ratio=0.3,
    shear_modulus=8.0e10,
    friction_coefficient=0.3,
    discretization=50
)

# Résoudre le problème de contact complet
contact_solution = kp_fastsim.solve_contact_problem(
    normal_force=100000.0,
    creepages={
        'longitudinal': 0.001,
        'lateral': 0.0,
        'spin': 0.0
    }
)

# Afficher les résultats
print(f"Force normale: {contact_solution['total_forces']['normal']:.2f} N")
print(f"Force tangentielle longitudinale: {contact_solution['total_forces']['tangential_x']:.2f} N")
print(f"Force tangentielle latérale: {contact_solution['total_forces']['tangential_y']:.2f} N")
