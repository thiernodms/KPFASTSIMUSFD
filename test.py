"""
Script de test pour l'implémentation des modèles KP-FastSim-USFD.

Ce script teste les différentes fonctionnalités de l'implémentation
pour s'assurer que tout fonctionne correctement.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.kp.kp import KikPiotrowski
from src.kp.mkp import ModifiedKikPiotrowski
from src.fastsim.fastsim import FastSim
from src.fastsim.fastrip import FaStrip
from src.usfd.usfd import USFDWearModel
from src.integration.kp_fastsim import KPFastSim
from src.integration.mkp_fastsim import MKPFastSim
from src.integration.mkp_fastrip import MKPFaStrip
from src.integration.full_model import KPFastSimUSFD
from src.utils.uic_profile import UICProfileLoader
from src.integration.uic_simulator import UICProfileSimulator

def test_kp_model():
    """
    Teste le modèle KP (Kik-Piotrowski).
    """
    print("Test du modèle KP (Kik-Piotrowski)...")
    
    # Créer une instance du modèle KP
    kp_model = KikPiotrowski(
        wheel_radius=0.46,
        rail_radius=0.3,
        wheel_youngs_modulus=210e9,
        rail_youngs_modulus=210e9,
        wheel_poisson_ratio=0.3,
        rail_poisson_ratio=0.3
    )
    
    # Résoudre le problème de contact normal
    normal_solution = kp_model.solve_contact_problem(normal_force=100000.0)
    
    # Vérifier les résultats
    print(f"  Force normale: {normal_solution['normal_force']:.2f} N")
    print(f"  Pénétration: {normal_solution['penetration'] * 1000:.4f} mm")
    print(f"  Demi-axe a: {normal_solution['contact_patch']['a'] * 1000:.4f} mm")
    print(f"  Demi-axe b: {normal_solution['contact_patch']['b'] * 1000:.4f} mm")
    print(f"  Surface de contact: {normal_solution['contact_patch']['area'] * 1e6:.2f} mm²")
    print(f"  Pression maximale: {normal_solution['contact_patch']['max_pressure'] / 1e6:.2f} MPa")
    
    # Vérifier que les résultats sont cohérents
    assert normal_solution['normal_force'] > 0, "La force normale doit être positive"
    assert normal_solution['penetration'] > 0, "La pénétration doit être positive"
    assert normal_solution['contact_patch']['a'] > 0, "Le demi-axe a doit être positif"
    assert normal_solution['contact_patch']['b'] > 0, "Le demi-axe b doit être positif"
    assert normal_solution['contact_patch']['area'] > 0, "La surface de contact doit être positive"
    assert normal_solution['contact_patch']['max_pressure'] > 0, "La pression maximale doit être positive"
    
    print("Test du modèle KP réussi!")

def test_mkp_model():
    """
    Teste le modèle MKP (Modified Kik-Piotrowski).
    """
    print("Test du modèle MKP (Modified Kik-Piotrowski)...")
    
    # Créer une instance du modèle MKP
    mkp_model = ModifiedKikPiotrowski(
        wheel_radius=0.46,
        rail_radius=0.3,
        wheel_youngs_modulus=210e9,
        rail_youngs_modulus=210e9,
        wheel_poisson_ratio=0.3,
        rail_poisson_ratio=0.3,
        semi_axes_ratio_limit=5.0
    )
    
    # Résoudre le problème de contact normal
    normal_solution = mkp_model.solve_contact_problem(normal_force=100000.0)
    
    # Vérifier les résultats
    print(f"  Force normale: {normal_solution['normal_force']:.2f} N")
    print(f"  Pénétration: {normal_solution['penetration'] * 1000:.4f} mm")
    print(f"  Demi-axe a: {normal_solution['contact_patch']['a'] * 1000:.4f} mm")
    print(f"  Demi-axe b: {normal_solution['contact_patch']['b'] * 1000:.4f} mm")
    print(f"  Surface de contact: {normal_solution['contact_patch']['area'] * 1e6:.2f} mm²")
    print(f"  Pression maximale: {normal_solution['contact_patch']['max_pressure'] / 1e6:.2f} MPa")
    
    # Vérifier que les résultats sont cohérents
    assert normal_solution['normal_force'] > 0, "La force normale doit être positive"
    assert normal_solution['penetration'] > 0, "La pénétration doit être positive"
    assert normal_solution['contact_patch']['a'] > 0, "Le demi-axe a doit être positif"
    assert normal_solution['contact_patch']['b'] > 0, "Le demi-axe b doit être positif"
    assert normal_solution['contact_patch']['area'] > 0, "La surface de contact doit être positive"
    assert normal_solution['contact_patch']['max_pressure'] > 0, "La pression maximale doit être positive"
    
    print("Test du modèle MKP réussi!")

def test_fastsim_algorithm():
    """
    Teste l'algorithme FastSim.
    """
    print("Test de l'algorithme FastSim...")
    
    # Créer une instance de l'algorithme FastSim
    fastsim_model = FastSim(
        poisson_ratio=0.3,
        shear_modulus=8.0e10,
        friction_coefficient=0.3,
        discretization=50
    )
    
    # Créer une zone de contact fictive
    a = 0.006  # 6 mm
    b = 0.004  # 4 mm
    max_pressure = 1000e6  # 1000 MPa
    
    # Créer une grille pour la zone de contact
    x = np.linspace(-a, a, 50)
    y = np.linspace(-b, b, 50)
    X, Y = np.meshgrid(x, y)
    
    # Calculer la distribution de pression (semi-ellipsoïdale)
    Z = max_pressure * np.sqrt(1 - (X/a)**2 - (Y/b)**2)
    Z[Z < 0] = 0  # Éliminer les valeurs négatives
    
    # Créer le dictionnaire de la zone de contact
    contact_patch = {
        'a': a,
        'b': b,
        'area': np.pi * a * b,
        'max_pressure': max_pressure,
        'mean_pressure': (2/3) * max_pressure,
        'pressure_distribution': Z.flatten()
    }
    
    # Créer le dictionnaire des glissements
    creepages = {
        'longitudinal': 0.001,
        'lateral': 0.0,
        'spin': 0.0
    }
    
    # Résoudre le problème de contact tangentiel
    tangential_solution = fastsim_model.solve_tangential_problem(
        contact_patch=contact_patch,
        creepages=creepages,
        normal_force=100000.0
    )
    
    # Vérifier les résultats
    print(f"  Force tangentielle longitudinale: {tangential_solution['tangential_forces'][0]:.2f} N")
    print(f"  Force tangentielle latérale: {tangential_solution['tangential_forces'][1]:.2f} N")
    print(f"  Moment de spin: {tangential_solution['moment']:.2f} N.m")
    print(f"  Proportion de la zone d'adhérence: {tangential_solution['adhesion_area']:.4f}")
    
    # Vérifier que les résultats sont cohérents
    assert tangential_solution['tangential_forces'][0] != 0, "La force tangentielle longitudinale ne doit pas être nulle"
    assert tangential_solution['adhesion_area'] >= 0 and tangential_solution['adhesion_area'] <= 1, "La proportion de la zone d'adhérence doit être entre 0 et 1"
    
    print("Test de l'algorithme FastSim réussi!")

def test_fastrip_algorithm():
    """
    Teste l'algorithme FaStrip.
    """
    print("Test de l'algorithme FaStrip...")
    
    # Créer une instance de l'algorithme FaStrip
    fastrip_model = FaStrip(
        poisson_ratio=0.3,
        shear_modulus=8.0e10,
        friction_coefficient=0.3,
        discretization=50,
        num_strips=20
    )
    
    # Créer une zone de contact fictive
    a = 0.006  # 6 mm
    b = 0.004  # 4 mm
    max_pressure = 1000e6  # 1000 MPa
    
    # Créer une grille pour la zone de contact
    x = np.linspace(-a, a, 50)
    y = np.linspace(-b, b, 50)
    X, Y = np.meshgrid(x, y)
    
    # Calculer la distribution de pression (semi-ellipsoïdale)
    Z = max_pressure * np.sqrt(1 - (X/a)**2 - (Y/b)**2)
    Z[Z < 0] = 0  # Éliminer les valeurs négatives
    
    # Créer le dictionnaire de la zone de contact
    contact_patch = {
        'a': a,
        'b': b,
        'area': np.pi * a * b,
        'max_pressure': max_pressure,
        'mean_pressure': (2/3) * max_pressure,
        'pressure_distribution': Z.flatten()
    }
    
    # Créer le dictionnaire des glissements
    creepages = {
        'longitudinal': 0.001,
        'lateral': 0.0,
        'spin': 0.0
    }
    
    # Résoudre le problème de contact tangentiel
    tangential_solution = fastrip_model.solve_tangential_problem(
        contact_patch=contact_patch,
        creepages=creepages,
        normal_force=100000.0
    )
    
    # Vérifier les résultats
    print(f"  Force tangentielle longitudinale: {tangential_solution['tangential_forces'][0]:.2f} N")
    print(f"  Force tangentielle latérale: {tangential_solution['tangential_forces'][1]:.2f} N")
    print(f"  Moment de spin: {tangential_solution['moment']:.2f} N.m")
    print(f"  Proportion de la zone d'adhérence: {tangential_solution['adhesion_area']:.4f}")
    
    # Vérifier que les résultats sont cohérents
    assert tangential_solution['tangential_forces'][0] != 0, "La force tangentielle longitudinale ne doit pas être nulle"
    assert tangential_solution['adhesion_area'] >= 0 and tangential_solution['adhesion_area'] <= 1, "La proportion de la zone d'adhérence doit être entre 0 et 1"
    
    print("Test de l'algorithme FaStrip réussi!")

def test_usfd_model():
    """
    Teste le modèle d'usure USFD.
    """
    print("Test du modèle d'usure USFD...")
    
    # Créer une instance du modèle USFD
    usfd_model = USFDWearModel(
        wheel_material="R8T",
        rail_material="UIC60 900A"
    )
    
    # Tester le calcul du taux d'usure
    tgamma_values = [5.0, 15.0, 100.0]  # N/mm²
    
    for tgamma in tgamma_values:
        wear_rate = usfd_model.calculate_wear_rate(tgamma)
        print(f"  Taux d'usure pour T-gamma = {tgamma:.1f} N/mm²: {wear_rate:.2f} μg/m/mm²")
    
    # Tester le calcul de la profondeur d'usure
    sliding_distance = 1000.0  # m
    wear_depth = usfd_model.calculate_wear_depth(tgamma_values[0], sliding_distance)
    print(f"  Profondeur d'usure pour T-gamma = {tgamma_values[0]:.1f} N/mm² et distance = {sliding_distance:.1f} m: {wear_depth * 1000:.3f} μm")
    
    # Tester le calcul du volume d'usure
    contact_area = 100.0  # mm²
    wear_volume = usfd_model.calculate_wear_volume(tgamma_values[0], sliding_distance, contact_area)
    print(f"  Volume d'usure pour T-gamma = {tgamma_values[0]:.1f} N/mm², distance = {sliding_distance:.1f} m et surface = {contact_area:.1f} mm²: {wear_volume:.2f} mm³")
    
    # Vérifier que les résultats sont cohérents
    assert wear_rate > 0, "Le taux d'usure doit être positif"
    assert wear_depth > 0, "La profondeur d'usure doit être positive"
    assert wear_volume > 0, "Le volume d'usure doit être positif"
    
    print("Test du modèle d'usure USFD réussi!")

def test_kp_fastsim_integration():
    """
    Teste l'intégration du modèle KP avec l'algorithme FastSim.
    """
    print("Test de l'intégration KP-FastSim...")
    
    # Créer une instance de l'intégration KP-FastSim
    kp_fastsim = KPFastSim(
        wheel_radius=0.46,
        rail_radius=0.3,
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
    
    # Vérifier les résultats
    print(f"  Force normale: {contact_solution['total_forces']['normal']:.2f} N")
    print(f"  Force tangentielle longitudinale: {contact_solution['total_forces']['tangential_x']:.2f} N")
    print(f"  Force tangentielle latérale: {contact_solution['total_forces']['tangential_y']:.2f} N")
    print(f"  Moment de spin: {contact_solution['total_forces']['moment']:.2f} N.m")
    
    # Vérifier que les résultats sont cohérents
    assert contact_solution['total_forces']['normal'] > 0, "La force normale doit être positive"
    assert contact_solution['total_forces']['tangential_x'] != 0, "La force tangentielle longitudinale ne doit pas être nulle"
    
    print("Test de l'intégration KP-FastSim réussi!")

def test_full_model_integration():
    """
    Teste l'intégration complète des modèles KP, FastSim et USFD.
    """
    print("Test de l'intégration complète KP-FastSim-USFD...")
    
    # Créer une instance de l'intégration complète
    full_model = KPFastSimUSFD(
        wheel_radius=0.46,
        rail_radius=0.3,
        wheel_youngs_modulus=210e9,
        rail_youngs_modulus=210e9,
        wheel_poisson_ratio=0.3,
        rail_poisson_ratio=0.3,
        shear_modulus=8.0e10,
        friction_coefficient=0.3,
        discretization=50,
        wheel_material="R8T",
        rail_material="UIC60 900A"
    )
    
    # Simuler le contact et l'usure
    simulation_result = full_model.simulate_contact_and_wear(
        normal_force=100000.0,
        creepages={
            'longitudinal': 0.001,
            'lateral': 0.0,
            'spin': 0.0
        },
        sliding_distance=1000.0
    )
    
    # Vérifier les résultats
    print(f"  Force normale: {simulation_result['contact_solution']['total_forces']['normal']:.2f} N")
    print(f"  Force tangentielle longitudinale: {simulation_result['contact_solution']['total_forces']['tangential_x']:.2f} N")
    print(f"  Force tangentielle latérale: {simulation_result['contact_solution']['total_forces']['tangential_y']:.2f} N")
    print(f"  Taux d'usure moyen: {simulation_result['wear_solution']['wear_rate']:.2f} μg/m/mm²")
    print(f"  Profondeur d'usure moyenne: {simulation_result['wear_solution']['wear_depth'] * 1000:.3f} μm")
    print(f"  Volume d'usure total: {simulation_result['wear_solution']['wear_volume']:.2f} mm³")
    
    # Vérifier que les résultats sont cohérents
    assert simulation_result['contact_solution']['total_forces']['normal'] > 0, "La force normale doit être positive"
    assert simulation_result['contact_solution']['total_forces']['tangential_x'] != 0, "La force tangentielle longitudinale ne doit pas être nulle"
    assert simulation_result['wear_solution']['wear_rate'] > 0, "Le taux d'usure doit être positif"
    assert simulation_result['wear_solution']['wear_depth'] > 0, "La profondeur d'usure doit être positive"
    assert simulation_result['wear_solution']['wear_volume'] > 0, "Le volume d'usure doit être positif"
    
    print("Test de l'intégration complète KP-FastSim-USFD réussi!")

def test_uic_profile_loader():
    """
    Teste le chargeur de profils UIC.
    """
    print("Test du chargeur de profils UIC...")
    
    # Vérifier que les fichiers de profil existent
    wheel_file = "profiles/S1002.wheel"
    rail_file = "profiles/uic60i00.rail"
    
    assert os.path.isfile(wheel_file), f"Le fichier {wheel_file} n'existe pas"
    assert os.path.isfile(rail_file), f"Le fichier {rail_file} n'existe pas"
    
    # Créer une instance du chargeur de profils UIC
    profile_loader = UICProfileLoader()
    
    # Charger les profils
    wheel_profile = profile_loader.load_wheel_profile(wheel_file)
    rail_profile = profile_loader.load_rail_profile(rail_file)
    
    # Vérifier les résultats
    print(f"  Profil de roue: {wheel_profile['name']}")
    print(f"    Nombre de points: {len(wheel_profile['x'])}")
    print(f"    Rayon de roulement: {wheel_profile['rolling_radius']:.2f} mm")
    print(f"    Hauteur du boudin: {wheel_profile['flange_height']:.2f} mm")
    print(f"    Épaisseur du boudin: {wheel_profile['flange_thickness']:.2f} mm")
    
    print(f"  Profil de rail: {rail_profile['name']}")
    print(f"    Nombre de points: {len(rail_profile['x'])}")
    print(f"    Largeur de la tête: {rail_profile['head_width']:.2f} mm")
    print(f"    Rayon de la tête: {rail_profile['head_radius']:.2f} mm")
    print(f"    Hauteur du rail: {rail_profile['rail_height']:.2f} mm")
    
    # Calculer la géométrie de contact
    contact_geometry = profile_loader.calculate_contact_geometry(
        wheel_profile['name'], rail_profile['name'], lateral_displacement=0.0
    )
    
    # Vérifier les résultats
    print(f"  Géométrie de contact:")
    print(f"    Point de contact sur la roue: ({contact_geometry['contact_point_wheel'][0]:.2f}, {contact_geometry['contact_point_wheel'][1]:.2f}) mm")
    print(f"    Point de contact sur le rail: ({contact_geometry['contact_point_rail'][0]:.2f}, {contact_geometry['contact_point_rail'][1]:.2f}) mm")
    print(f"    Angle de contact: {contact_geometry['contact_angle'] * 180 / np.pi:.2f} degrés")
    print(f"    Rayon de roulement effectif: {contact_geometry['rolling_radius']:.2f} mm")
    
    # Vérifier que les résultats sont cohérents
    assert len(wheel_profile['x']) > 0, "Le profil de roue doit contenir des points"
    assert len(rail_profile['x']) > 0, "Le profil de rail doit contenir des points"
    assert wheel_profile['rolling_radius'] > 0, "Le rayon de roulement doit être positif"
    assert rail_profile['head_width'] > 0, "La largeur de la tête du rail doit être positive"
    
    print("Test du chargeur de profils UIC réussi!")

def test_uic_simulator():
    """
    Teste le simulateur avec profils UIC.
    """
    print("Test du simulateur avec profils UIC...")
    
    # Vérifier que les fichiers de profil existent
    wheel_file = "profiles/S1002.wheel"
    rail_file = "profiles/uic60i00.rail"
    
    assert os.path.isfile(wheel_file), f"Le fichier {wheel_file} n'existe pas"
    assert os.path.isfile(rail_file), f"Le fichier {rail_file} n'existe pas"
    
    # Créer une instance du simulateur
    simulator = UICProfileSimulator()
    
    # Charger les profils
    wheel_profile, rail_profile = simulator.load_profiles(wheel_file, rail_file)
    
    # Créer le modèle
    model = simulator.create_model(wheel_profile['name'], rail_profile['name'], lateral_displacement=0.0)
    
    # Simuler le contact et l'usure
    simulation_result = simulator.simulate_contact_and_wear(
        model_name=f"{wheel_profile['name']}_{rail_profile['name']}_0.0mm",
        normal_force=100000.0,
        longitudinal_creepage=0.001,
        lateral_creepage=0.0,
        spin_creepage=0.0,
        sliding_distance=1000.0
    )
    
    # Vérifier les résultats
    print(f"  Force normale: {simulation_result['contact_solution']['total_forces']['normal']:.2f} N")
    print(f"  Force tangentielle longitudinale: {simulation_result['contact_solution']['total_forces']['tangential_x']:.2f} N")
    print(f"  Force tangentielle latérale: {simulation_result['contact_solution']['total_forces']['tangential_y']:.2f} N")
    print(f"  Taux d'usure moyen: {simulation_result['wear_solution']['wear_rate']:.2f} μg/m/mm²")
    print(f"  Profondeur d'usure moyenne: {simulation_result['wear_solution']['wear_depth'] * 1000:.3f} μm")
    print(f"  Volume d'usure total: {simulation_result['wear_solution']['wear_volume']:.2f} mm³")
    
    # Calculer la conicité équivalente
    equivalent_conicity = simulator.calculate_equivalent_conicity(
        wheel_profile['name'], rail_profile['name'], lateral_displacement_range=(-10.0, 10.0), steps=20
    )
    
    print(f"  Conicité équivalente: {equivalent_conicity:.6f}")
    
    # Vérifier que les résultats sont cohérents
    assert simulation_result['contact_solution']['total_forces']['normal'] > 0, "La force normale doit être positive"
    assert simulation_result['contact_solution']['total_forces']['tangential_x'] != 0, "La force tangentielle longitudinale ne doit pas être nulle"
    assert simulation_result['wear_solution']['wear_rate'] > 0, "Le taux d'usure doit être positif"
    assert simulation_result['wear_solution']['wear_depth'] > 0, "La profondeur d'usure doit être positive"
    assert simulation_result['wear_solution']['wear_volume'] > 0, "Le volume d'usure doit être positif"
    
    print("Test du simulateur avec profils UIC réussi!")

def run_all_tests():
    """
    Exécute tous les tests.
    """
    print("Exécution de tous les tests...")
    print("=" * 80)
    
    # Tester les modèles individuels
    test_kp_model()
    print("-" * 80)
    test_mkp_model()
    print("-" * 80)
    test_fastsim_algorithm()
    print("-" * 80)
    test_fastrip_algorithm()
    print("-" * 80)
    test_usfd_model()
    print("-" * 80)
    
    # Tester les intégrations
    test_kp_fastsim_integration()
    print("-" * 80)
    test_full_model_integration()
    print("-" * 80)
    
    # Tester le support des profils UIC
    test_uic_profile_loader()
    print("-" * 80)
    test_uic_simulator()
    
    print("=" * 80)
    print("Tous les tests ont réussi!")

if __name__ == "__main__":
    run_all_tests()
