from src.utils.uic_profile import UICProfileLoader
from src.kp.kp import KikPiotrowski
import numpy as np

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

# Créer une instance du modèle KP avec les rayons calculés
kp_model = KikPiotrowski(
    wheel_radius=wheel_radius,
    rail_radius=rail_radius,
    wheel_youngs_modulus=210e9,
    rail_youngs_modulus=210e9,
    wheel_poisson_ratio=0.3,
    rail_poisson_ratio=0.3
)

# Résoudre le problème de contact normal
normal_solution = kp_model.solve_contact_problem(normal_force=100000.0)

# Afficher les résultats
print(f"Force normale: {normal_solution['normal_force']:.2f} N")
print(f"Pénétration: {normal_solution['penetration'] * 1000:.4f} mm")
print(f"Demi-axe a: {normal_solution['contact_patch']['a'] * 1000:.4f} mm")
print(f"Demi-axe b: {normal_solution['contact_patch']['b'] * 1000:.4f} mm")
