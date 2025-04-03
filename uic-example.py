from src.integration.uic_simulator import UICProfileSimulator

# Créer une instance du simulateur
simulator = UICProfileSimulator() 

# Charger les profils
wheel_profile, rail_profile = simulator.load_profiles("profiles/S1002.wheel", "profiles/uic60i00.rail")

# Créer le modèle
model = simulator.create_model(wheel_profile['name'], rail_profile['name'], lateral_displacement=0.0)

# Simuler le contact et l'usure
simulation_result = simulator.simulate_contact_and_wear(
    model_name=f"{wheel_profile['name']}_{rail_profile['name']}_0.0mm",
    normal_force=100000.0,
    longitudinal_creepage=-2,
    lateral_creepage=2,
    spin_creepage=3,
    sliding_distance=1000.0
)

# Afficher les résultats
print(f"Force normale: {simulation_result['contact_solution']['total_forces']['normal']:.2f} N")
print(f"Force tangentielle: {simulation_result['contact_solution']['total_forces']['tangential_x']:.2f} N")
print(f"Taux d'usure: {simulation_result['wear_solution']['wear_rate']:.2f} μg/m/mm²")
print(f"Profondeur d'usure: {simulation_result['wear_solution']['wear_depth'] * 1000:.3f} μm")
