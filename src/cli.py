"""
Interface en ligne de commande pour les modèles KP-FastSim-USFD.

Ce module fournit une interface utilisateur simple pour utiliser les modèles
KP-FastSim-USFD avec des profils UIC de roue et rail.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from src.integration.uic_simulator import UICProfileSimulator

def plot_profiles(wheel_profile, rail_profile, lateral_displacement=0.0, output_file=None):
    """
    Affiche les profils de roue et rail.
    
    Args:
        wheel_profile (dict): Dictionnaire contenant les données du profil de roue.
        rail_profile (dict): Dictionnaire contenant les données du profil de rail.
        lateral_displacement (float): Déplacement latéral de la roue par rapport au rail en mm.
        output_file (str, optional): Chemin du fichier de sortie pour enregistrer le graphique.
    """
    # Créer une nouvelle figure
    plt.figure(figsize=(10, 6))
    
    # Extraire les coordonnées
    wheel_x = wheel_profile['x'] + lateral_displacement
    wheel_y = wheel_profile['y']
    rail_x = rail_profile['x']
    rail_y = rail_profile['y']
    
    # Tracer les profils
    plt.plot(wheel_x, wheel_y, 'b-', label='Roue')
    plt.plot(rail_x, rail_y, 'r-', label='Rail')
    
    # Ajouter les légendes et les titres
    plt.title(f'Profils de roue ({wheel_profile["name"]}) et rail ({rail_profile["name"]})')
    plt.xlabel('Position latérale (mm)')
    plt.ylabel('Position verticale (mm)')
    plt.legend()
    plt.grid(True)
    
    # Ajuster les limites des axes
    x_min = min(np.min(wheel_x), np.min(rail_x))
    x_max = max(np.max(wheel_x), np.max(rail_x))
    y_min = min(np.min(wheel_y), np.min(rail_y))
    y_max = max(np.max(wheel_y), np.max(rail_y))
    plt.xlim(x_min - 10, x_max + 10)
    plt.ylim(y_min - 10, y_max + 10)
    
    # Enregistrer le graphique si un fichier de sortie est spécifié
    if output_file:
        plt.savefig(output_file)
    
    # Afficher le graphique
    plt.show()

def plot_contact_patch(contact_solution, output_file=None):
    """
    Affiche la zone de contact et la distribution de pression.
    
    Args:
        contact_solution (dict): Solution du problème de contact.
        output_file (str, optional): Chemin du fichier de sortie pour enregistrer le graphique.
    """
    # Extraire les informations de la zone de contact
    contact_patch = contact_solution['normal_solution']['contact_patch']
    a = contact_patch['a'] * 1000  # Convertir en mm
    b = contact_patch['b'] * 1000  # Convertir en mm
    max_pressure = contact_patch['max_pressure'] / 1e6  # Convertir en MPa
    
    # Créer une grille pour la zone de contact
    x = np.linspace(-a, a, 100)
    y = np.linspace(-b, b, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculer la distribution de pression (semi-ellipsoïdale)
    ellipse_term = 1 - (X/a)**2 - (Y/b)**2
    ellipse_term[ellipse_term < 0] = 0  # Éviter les valeurs négatives
    Z = max_pressure * np.sqrt(ellipse_term)
    
    # Créer une nouvelle figure
    plt.figure(figsize=(10, 8))
    
    # Tracer la distribution de pression
    plt.subplot(2, 1, 1)
    contour = plt.contourf(X, Y, Z, 20, cmap='viridis')
    plt.colorbar(contour, label='Pression (MPa)')
    plt.title(f'Distribution de pression dans la zone de contact (Pmax = {max_pressure:.2f} MPa)')
    plt.xlabel('Direction longitudinale (mm)')
    plt.ylabel('Direction latérale (mm)')
    plt.axis('equal')
    
    # Tracer la zone de contact
    plt.subplot(2, 1, 2)
    circle = plt.Circle((0, 0), 1, fill=False, color='r', linestyle='--')
    plt.gca().add_patch(circle)
    ellipse = plt.matplotlib.patches.Ellipse((0, 0), 2*a, 2*b, fill=False, color='b')
    plt.gca().add_patch(ellipse)
    plt.xlim(-max(a, b) * 1.2, max(a, b) * 1.2)
    plt.ylim(-max(a, b) * 1.2, max(a, b) * 1.2)
    plt.title(f'Zone de contact (a = {a:.2f} mm, b = {b:.2f} mm)')
    plt.xlabel('Direction longitudinale (mm)')
    plt.ylabel('Direction latérale (mm)')
    plt.grid(True)
    plt.axis('equal')
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Enregistrer le graphique si un fichier de sortie est spécifié
    if output_file:
        plt.savefig(output_file)
    
    # Afficher le graphique
    plt.show()

def plot_wear_distribution(simulation_result, output_file=None):
    """
    Affiche la distribution d'usure.
    
    Args:
        simulation_result (dict): Résultat de la simulation de contact et d'usure.
        output_file (str, optional): Chemin du fichier de sortie pour enregistrer le graphique.
    """
    # Vérifier que la solution d'usure existe
    if simulation_result['wear_solution'] is None:
        print("Aucune solution d'usure disponible.")
        return
    
    # Extraire les informations de la zone de contact
    contact_patch = simulation_result['contact_solution']['normal_solution']['contact_patch']
    a = contact_patch['a'] * 1000  # Convertir en mm
    b = contact_patch['b'] * 1000  # Convertir en mm
    
    # Extraire les taux d'usure locaux
    local_wear_rate = simulation_result['wear_solution']['local_wear_rate']
    local_wear_depth = simulation_result['wear_solution']['local_wear_depth']
    
    # Créer une grille pour la zone de contact
    x = np.linspace(-a, a, 100)
    y = np.linspace(-b, b, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculer la distribution d'usure (simplifiée)
    ellipse_term = 1 - (X/a)**2 - (Y/b)**2
    ellipse_term[ellipse_term < 0] = 0  # Éviter les valeurs négatives
    mask = ellipse_term > 0
    
    # Créer une nouvelle figure
    plt.figure(figsize=(10, 8))
    
    # Tracer le taux d'usure
    plt.subplot(2, 1, 1)
    wear_rate = simulation_result['wear_solution']['wear_rate'] * ellipse_term
    contour = plt.contourf(X, Y, wear_rate, 20, cmap='viridis')
    plt.colorbar(contour, label='Taux d\'usure (μg/m/mm²)')
    plt.title(f'Distribution du taux d\'usure (moyenne = {simulation_result["wear_solution"]["wear_rate"]:.2f} μg/m/mm²)')
    plt.xlabel('Direction longitudinale (mm)')
    plt.ylabel('Direction latérale (mm)')
    plt.axis('equal')
    
    # Tracer la profondeur d'usure
    plt.subplot(2, 1, 2)
    wear_depth = simulation_result['wear_solution']['wear_depth'] * 1000 * ellipse_term  # Convertir en μm
    contour = plt.contourf(X, Y, wear_depth, 20, cmap='viridis')
    plt.colorbar(contour, label='Profondeur d\'usure (μm)')
    plt.title(f'Distribution de la profondeur d\'usure (moyenne = {simulation_result["wear_solution"]["wear_depth"] * 1000:.3f} μm)')
    plt.xlabel('Direction longitudinale (mm)')
    plt.ylabel('Direction latérale (mm)')
    plt.axis('equal')
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Enregistrer le graphique si un fichier de sortie est spécifié
    if output_file:
        plt.savefig(output_file)
    
    # Afficher le graphique
    plt.show()

def main():
    """
    Fonction principale pour l'interface en ligne de commande.
    """
    # Créer le parseur d'arguments
    parser = argparse.ArgumentParser(description='Interface pour les modèles KP-FastSim-USFD')
    
    # Ajouter les sous-commandes
    subparsers = parser.add_subparsers(dest='command', help='Commande à exécuter')
    
    # Sous-commande pour afficher les profils
    profile_parser = subparsers.add_parser('profiles', help='Afficher les profils de roue et rail')
    profile_parser.add_argument('wheel_file', help='Fichier de profil UIC de la roue')
    profile_parser.add_argument('rail_file', help='Fichier de profil UIC du rail')
    profile_parser.add_argument('--lateral', type=float, default=0.0, 
                              help='Déplacement latéral de la roue par rapport au rail en mm')
    profile_parser.add_argument('--output', help='Fichier de sortie pour enregistrer le graphique')
    
    # Sous-commande pour simuler le contact
    contact_parser = subparsers.add_parser('contact', help='Simuler le contact roue-rail')
    contact_parser.add_argument('wheel_file', help='Fichier de profil UIC de la roue')
    contact_parser.add_argument('rail_file', help='Fichier de profil UIC du rail')
    contact_parser.add_argument('--lateral', type=float, default=0.0, 
                              help='Déplacement latéral de la roue par rapport au rail en mm')
    contact_parser.add_argument('--force', type=float, default=100000.0, 
                              help='Force normale en Newtons')
    contact_parser.add_argument('--output', help='Fichier de sortie pour enregistrer le graphique')
    
    # Sous-commande pour simuler l'usure
    wear_parser = subparsers.add_parser('wear', help='Simuler l\'usure de la roue')
    wear_parser.add_argument('wheel_file', help='Fichier de profil UIC de la roue')
    wear_parser.add_argument('rail_file', help='Fichier de profil UIC du rail')
    wear_parser.add_argument('--lateral', type=float, default=0.0, 
                           help='Déplacement latéral de la roue par rapport au rail en mm')
    wear_parser.add_argument('--force', type=float, default=100000.0, 
                           help='Force normale en Newtons')
    wear_parser.add_argument('--long-creep', type=float, default=0.001, 
                           help='Glissement longitudinal (adimensionnel)')
    wear_parser.add_argument('--lat-creep', type=float, default=0.0, 
                           help='Glissement latéral (adimensionnel)')
    wear_parser.add_argument('--spin', type=float, default=0.0, 
                           help='Spin (1/m)')
    wear_parser.add_argument('--distance', type=float, default=1000.0, 
                           help='Distance de glissement en mètres')
    wear_parser.add_argument('--output', help='Fichier de sortie pour enregistrer le graphique')
    
    # Sous-commande pour calculer la conicité équivalente
    conicity_parser = subparsers.add_parser('conicity', help='Calculer la conicité équivalente')
    conicity_parser.add_argument('wheel_file', help='Fichier de profil UIC de la roue')
    conicity_parser.add_argument('rail_file', help='Fichier de profil UIC du rail')
    conicity_parser.add_argument('--range', type=float, nargs=2, default=(-10.0, 10.0), 
                               help='Plage de déplacements latéraux en mm')
    conicity_parser.add_argument('--steps', type=int, default=20, 
                               help='Nombre de points pour le calcul')
    
    # Analyser les arguments
    args = parser.parse_args()
    
    # Créer le simulateur
    simulator = UICProfileSimulator()
    
    # Exécuter la commande appropriée
    if args.command == 'profiles':
        # Charger les profils
        wheel_profile, rail_profile = simulator.load_profiles(args.wheel_file, args.rail_file)
        
        # Afficher les profils
        plot_profiles(wheel_profile, rail_profile, args.lateral, args.output)
        
    elif args.command == 'contact':
        # Charger les profils
        wheel_profile, rail_profile = simulator.load_profiles(args.wheel_file, args.rail_file)
        
        # Créer le modèle
        model = simulator.create_model(wheel_profile['name'], rail_profile['name'], args.lateral)
        
        # Simuler le contact
        simulation_result = simulator.simulate_contact_and_wear(
            model_name=f"{wheel_profile['name']}_{rail_profile['name']}_{args.lateral:.1f}mm",
            normal_force=args.force
        )
        
        # Afficher les résultats
        print(f"Résultats de la simulation de contact:")
        print(f"  Force normale: {simulation_result['contact_solution']['total_forces']['normal']:.2f} N")
        print(f"  Demi-axe a: {simulation_result['contact_solution']['normal_solution']['contact_patch']['a'] * 1000:.2f} mm")
        print(f"  Demi-axe b: {simulation_result['contact_solution']['normal_solution']['contact_patch']['b'] * 1000:.2f} mm")
        print(f"  Surface de contact: {simulation_result['contact_solution']['normal_solution']['contact_patch']['area'] * 1e6:.2f} mm²")
        print(f"  Pression maximale: {simulation_result['contact_solution']['normal_solution']['contact_patch']['max_pressure'] / 1e6:.2f} MPa")
        print(f"  Pression moyenne: {simulation_result['contact_solution']['normal_solution']['contact_patch']['mean_pressure'] / 1e6:.2f} MPa")
        
        # Afficher la zone de contact
        plot_contact_patch(simulation_result['contact_solution'], args.output)
        
    elif args.command == 'wear':
        # Charger les profils
        wheel_profile, rail_profile = simulator.load_profiles(args.wheel_file, args.rail_file)
        
        # Créer le modèle
        model = simulator.create_model(wheel_profile['name'], rail_profile['name'], args.lateral)
        
        # Simuler le contact et l'usure
        simulation_result = simulator.simulate_contact_and_wear(
            model_name=f"{wheel_profile['name']}_{rail_profile['name']}_{args.lateral:.1f}mm",
            normal_force=args.force,
            longitudinal_creepage=args.long_creep,
            lateral_creepage=args.lat_creep,
            spin_creepage=args.spin,
            sliding_distance=args.distance
        )
        
        # Afficher les résultats
        print(f"Résultats de la simulation d'usure:")
        print(f"  Force normale: {simulation_result['contact_solution']['total_forces']['normal']:.2f} N")
        print(f"  Force tangentielle longitudinale: {simulation_result['contact_solution']['total_forces']['tangential_x']:.2f} N")
        print(f"  Force tangentielle latérale: {simulation_result['contact_solution']['total_forces']['tangential_y']:.2f} N")
        print(f"  Taux d'usure moyen: {simulation_result['wear_solution']['wear_rate']:.2f} μg/m/mm²")
        print(f"  Profondeur d'usure moyenne: {simulation_result['wear_solution']['wear_depth'] * 1000:.3f} μm")
        print(f"  Volume d'usure total: {simulation_result['wear_solution']['wear_volume']:.2f} mm³")
        
        # Afficher la distribution d'usure
        plot_wear_distribution(simulation_result, args.output)
        
    elif args.command == 'conicity':
        # Charger les profils
        wheel_profile, rail_profile = simulator.load_profiles(args.wheel_file, args.rail_file)
        
        # Calculer la conicité équivalente
        equivalent_conicity = simulator.calculate_equivalent_conicity(
            wheel_profile['name'], rail_profile['name'], args.range, args.steps
        )
        
        # Afficher le résultat
        print(f"Conicité équivalente: {equivalent_conicity:.6f}")
    
    else:
        # Afficher l'aide si aucune commande n'est spécifiée
        parser.print_help()

if __name__ == '__main__':
    main()
