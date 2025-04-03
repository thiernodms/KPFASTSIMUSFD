"""
Fonctions utilitaires pour les calculs géométriques liés au contact roue-rail.

Ce module fournit des fonctions pour calculer les courbures, les angles de contact,
et d'autres propriétés géométriques nécessaires aux modèles de contact.
"""

import numpy as np

def calculate_curvatures(wheel_radius, rail_radius, yaw_angle=0.0):
    """
    Calcule les courbures principales au point de contact roue-rail.
    
    Args:
        wheel_radius (float): Rayon de la roue en mètres.
        rail_radius (float): Rayon du rail en mètres (rayon de courbure de la tête du rail).
        yaw_angle (float): Angle de lacet en radians. Par défaut 0.0.
        
    Returns:
        dict: Dictionnaire contenant les courbures principales:
            - 'longitudinal': Courbure dans la direction longitudinale (1/m)
            - 'lateral': Courbure dans la direction latérale (1/m)
    """
    # Courbure longitudinale (direction de roulement)
    # Pour un contact roue-rail, la courbure longitudinale est principalement
    # déterminée par le rayon de la roue
    longitudinal_curvature = 1.0 / wheel_radius
    
    # Courbure latérale (direction transversale)
    # Pour un contact roue-rail, la courbure latérale est influencée par
    # le rayon du rail et l'angle de lacet
    lateral_curvature = 1.0 / rail_radius
    
    # Effet de l'angle de lacet sur les courbures
    if abs(yaw_angle) > 1e-6:  # Si l'angle de lacet n'est pas négligeable
        # L'angle de lacet modifie les courbures principales
        # Ces formules sont des approximations basées sur la géométrie du contact
        cos_yaw = np.cos(yaw_angle)
        sin_yaw = np.sin(yaw_angle)
        
        # Modification de la courbure longitudinale
        longitudinal_curvature = longitudinal_curvature * cos_yaw**2 + lateral_curvature * sin_yaw**2
        
        # Modification de la courbure latérale
        lateral_curvature = lateral_curvature * cos_yaw**2 + longitudinal_curvature * sin_yaw**2
    
    return {
        'longitudinal': longitudinal_curvature,
        'lateral': lateral_curvature
    }

def calculate_contact_angle(wheel_profile, rail_profile, lateral_displacement):
    """
    Calcule l'angle de contact entre la roue et le rail.
    
    Args:
        wheel_profile (array): Profil de la roue sous forme de points (x, y).
        rail_profile (array): Profil du rail sous forme de points (x, y).
        lateral_displacement (float): Déplacement latéral de la roue par rapport au rail en mètres.
        
    Returns:
        float: Angle de contact en radians.
    """
    # Cette fonction est une implémentation simplifiée
    # Dans une implémentation réelle, il faudrait calculer l'angle de contact
    # en fonction des profils de roue et de rail et du déplacement latéral
    
    # Pour l'instant, nous utilisons une approximation linéaire
    # basée sur le déplacement latéral
    nominal_contact_angle = 0.05  # Angle de contact nominal (environ 3 degrés)
    angle_variation_rate = 0.2    # Taux de variation de l'angle avec le déplacement latéral
    
    contact_angle = nominal_contact_angle + angle_variation_rate * lateral_displacement
    
    return contact_angle

def calculate_equivalent_conicity(wheel_profile, rail_profile, lateral_displacement_range=(-0.01, 0.01), steps=20):
    """
    Calcule la conicité équivalente pour une plage de déplacements latéraux.
    
    Args:
        wheel_profile (array): Profil de la roue sous forme de points (x, y).
        rail_profile (array): Profil du rail sous forme de points (x, y).
        lateral_displacement_range (tuple): Plage de déplacements latéraux en mètres.
        steps (int): Nombre de points pour le calcul.
        
    Returns:
        float: Conicité équivalente (sans unité).
    """
    # Cette fonction est une implémentation simplifiée
    # Dans une implémentation réelle, il faudrait calculer la conicité équivalente
    # en fonction des profils de roue et de rail
    
    # Pour l'instant, nous utilisons une approximation
    min_displacement, max_displacement = lateral_displacement_range
    displacements = np.linspace(min_displacement, max_displacement, steps)
    
    # Calculer l'angle de contact pour chaque déplacement
    contact_angles = [calculate_contact_angle(wheel_profile, rail_profile, y) for y in displacements]
    
    # La conicité équivalente est la pente moyenne de la fonction de différence de rayon
    # par rapport au déplacement latéral
    # Pour simplifier, nous utilisons la pente moyenne des angles de contact
    equivalent_conicity = np.mean(np.diff(contact_angles) / np.diff(displacements))
    
    return equivalent_conicity

def discretize_contact_patch(a, b, discretization):
    """
    Discrétise la zone de contact elliptique en une grille de points.
    
    Args:
        a (float): Demi-longueur de la zone de contact dans la direction longitudinale (m).
        b (float): Demi-longueur de la zone de contact dans la direction latérale (m).
        discretization (int): Nombre de points de discrétisation dans chaque direction.
        
    Returns:
        tuple: (X, Y, mask) où X et Y sont les coordonnées des points et mask est un masque
               booléen indiquant les points à l'intérieur de l'ellipse.
    """
    # Créer une grille de points
    x = np.linspace(-a, a, discretization)
    y = np.linspace(-b, b, discretization)
    X, Y = np.meshgrid(x, y)
    
    # Créer un masque pour les points à l'intérieur de l'ellipse
    mask = (X/a)**2 + (Y/b)**2 <= 1.0
    
    return X, Y, mask
