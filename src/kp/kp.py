"""
Implémentation du modèle de contact non-hertzien Kik-Piotrowski (KP).

Ce module implémente le modèle KP original pour calculer la géométrie de contact
et la distribution de pression entre une roue et un rail.

Références:
    [1] Kik, W., & Piotrowski, J. (1996). A fast, approximate method to calculate normal
        load at contact between wheel and rail and creep forces during rolling.
    [2] Piotrowski, J., & Kik, W. (2008). A simplified model of wheel/rail contact mechanics
        for non-Hertzian problems and its application in rail vehicle dynamic simulations.
"""

import numpy as np
from scipy import optimize
from ..utils.geometry import calculate_curvatures

class KikPiotrowski:
    """
    Implémentation du modèle de contact non-hertzien Kik-Piotrowski (KP).
    
    Ce modèle permet de calculer la géométrie de contact et la distribution de pression
    pour des contacts non-elliptiques entre une roue et un rail.
    """
    
    def __init__(self, wheel_radius, rail_radius, wheel_youngs_modulus=210e9, 
                 rail_youngs_modulus=210e9, wheel_poisson_ratio=0.3, rail_poisson_ratio=0.3,
                 discretization=50):
        """
        Initialise le modèle KP avec les paramètres géométriques et matériels.
        
        Args:
            wheel_radius (float): Rayon de la roue en mètres.
            rail_radius (float): Rayon du rail en mètres.
            wheel_youngs_modulus (float): Module de Young de la roue en Pa. Par défaut 210 GPa.
            rail_youngs_modulus (float): Module de Young du rail en Pa. Par défaut 210 GPa.
            wheel_poisson_ratio (float): Coefficient de Poisson de la roue. Par défaut 0.3.
            rail_poisson_ratio (float): Coefficient de Poisson du rail. Par défaut 0.3.
            discretization (int): Nombre de points de discrétisation pour la zone de contact.
        """
        self.wheel_radius = wheel_radius
        self.rail_radius = rail_radius
        self.wheel_youngs_modulus = wheel_youngs_modulus
        self.rail_youngs_modulus = rail_youngs_modulus
        self.wheel_poisson_ratio = wheel_poisson_ratio
        self.rail_poisson_ratio = rail_poisson_ratio
        self.discretization = discretization
        
        # Calcul du module de Young équivalent
        self.equivalent_youngs_modulus = 1.0 / (
            (1.0 - wheel_poisson_ratio**2) / wheel_youngs_modulus +
            (1.0 - rail_poisson_ratio**2) / rail_youngs_modulus
        )
        
    def calculate_contact_patch(self, normal_force, yaw_angle=0.0, lateral_displacement=0.0):
        """
        Calcule la géométrie de la zone de contact selon le modèle KP.
        
        Args:
            normal_force (float): Force normale appliquée en Newtons.
            yaw_angle (float): Angle de lacet en radians. Par défaut 0.0.
            lateral_displacement (float): Déplacement latéral en mètres. Par défaut 0.0.
            
        Returns:
            dict: Dictionnaire contenant les paramètres de la zone de contact:
                - 'a': Demi-longueur de la zone de contact dans la direction longitudinale (m)
                - 'b': Demi-longueur de la zone de contact dans la direction latérale (m)
                - 'area': Surface de la zone de contact (m²)
                - 'max_pressure': Pression maximale (Pa)
                - 'mean_pressure': Pression moyenne (Pa)
                - 'contact_points': Coordonnées des points de contact (x, y)
                - 'pressure_distribution': Distribution de pression aux points de contact
        """
        # Calcul des courbures principales
        curvatures = calculate_curvatures(self.wheel_radius, self.rail_radius, yaw_angle)
        
        # Calcul des paramètres de la zone de contact selon le modèle KP
        # Formule simplifiée pour la demi-longueur a (direction longitudinale)
        a = (3 * normal_force / (4 * self.equivalent_youngs_modulus * 
             (curvatures['longitudinal'] + curvatures['lateral'])))**(1/3)
        
        # Formule simplifiée pour la demi-longueur b (direction latérale)
        b = a * np.sqrt(curvatures['longitudinal'] / curvatures['lateral'])
        
        # Calcul de la surface de contact
        area = np.pi * a * b
        
        # Calcul de la pression maximale (au centre de la zone de contact)
        max_pressure = 1.5 * normal_force / area
        
        # Calcul de la pression moyenne
        mean_pressure = normal_force / area
        
        # Création de la grille de points pour la zone de contact
        x = np.linspace(-a, a, self.discretization)
        y = np.linspace(-b, b, self.discretization)
        X, Y = np.meshgrid(x, y)
        
        # Calcul de la distribution de pression selon le modèle KP
        # Dans le modèle KP, la distribution de pression est semi-ellipsoïdale
        # Calculer la distribution de pression (semi-ellipsoïdale)
        ellipse_term = 1 - (X/a)**2 - (Y/b)**2
        # Éviter les valeurs négatives avant de calculer la racine carrée
        ellipse_term[ellipse_term < 0] = 0
        pressure_distribution = max_pressure * np.sqrt(ellipse_term)

        pressure_distribution[pressure_distribution < 0] = 0  # Éliminer les valeurs négatives
        
        # Coordonnées des points de contact
        contact_points = np.column_stack((X.flatten(), Y.flatten()))
        
        return {
            'a': a,
            'b': b,
            'area': area,
            'max_pressure': max_pressure,
            'mean_pressure': mean_pressure,
            'contact_points': contact_points,
            'pressure_distribution': pressure_distribution.flatten()
        }
    
    def calculate_penetration(self, normal_force):
        """
        Calcule la pénétration (interpénétration) entre la roue et le rail.
        
        Args:
            normal_force (float): Force normale appliquée en Newtons.
            
        Returns:
            float: Pénétration en mètres.
        """
        # Calcul des courbures principales
        curvatures = calculate_curvatures(self.wheel_radius, self.rail_radius, 0.0)
        
        # Calcul de la pénétration selon le modèle KP
        # Formule simplifiée pour la demi-longueur a (direction longitudinale)
        a = (3 * normal_force / (4 * self.equivalent_youngs_modulus * 
             (curvatures['longitudinal'] + curvatures['lateral'])))**(1/3)
        
        # Formule pour la pénétration
        penetration = a**2 * (curvatures['longitudinal'] + curvatures['lateral']) / 3
        
        return penetration
    
    def calculate_normal_force(self, penetration):
        """
        Calcule la force normale à partir de la pénétration.
        
        Args:
            penetration (float): Pénétration en mètres.
            
        Returns:
            float: Force normale en Newtons.
        """
        # Calcul des courbures principales
        curvatures = calculate_curvatures(self.wheel_radius, self.rail_radius, 0.0)
        
        # Calcul de la force normale à partir de la pénétration
        # Inversion de la formule de pénétration
        a = np.sqrt(3 * penetration / (curvatures['longitudinal'] + curvatures['lateral']))
        normal_force = 4 * self.equivalent_youngs_modulus * (curvatures['longitudinal'] + 
                       curvatures['lateral']) * a**3 / 3
        
        return normal_force
    
    def solve_contact_problem(self, penetration=None, normal_force=None, yaw_angle=0.0, 
                             lateral_displacement=0.0):
        """
        Résout le problème de contact en calculant soit la force normale à partir de la pénétration,
        soit la pénétration à partir de la force normale, puis calcule la géométrie de contact.
        
        Args:
            penetration (float, optional): Pénétration en mètres.
            normal_force (float, optional): Force normale en Newtons.
            yaw_angle (float): Angle de lacet en radians. Par défaut 0.0.
            lateral_displacement (float): Déplacement latéral en mètres. Par défaut 0.0.
            
        Returns:
            dict: Dictionnaire contenant les résultats du problème de contact.
            
        Raises:
            ValueError: Si ni penetration ni normal_force ne sont fournis, ou si les deux sont fournis.
        """
        if (penetration is None and normal_force is None) or (penetration is not None and normal_force is not None):
            raise ValueError("Vous devez fournir soit penetration soit normal_force, mais pas les deux.")
        
        if penetration is not None:
            normal_force = self.calculate_normal_force(penetration)
        else:
            penetration = self.calculate_penetration(normal_force)
        
        contact_patch = self.calculate_contact_patch(normal_force, yaw_angle, lateral_displacement)
        
        return {
            'penetration': penetration,
            'normal_force': normal_force,
            'contact_patch': contact_patch
        }
