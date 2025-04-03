"""
Implémentation du modèle de contact non-hertzien Modified Kik-Piotrowski (MKP).

Ce module implémente la version modifiée du modèle KP pour calculer la géométrie de contact
et la distribution de pression entre une roue et un rail, avec une meilleure précision
pour les contacts non-elliptiques.

Références:
    [1] Piotrowski, J., & Chollet, H. (2005). Wheel–rail contact models for vehicle system
        dynamics including multi-point contact. Vehicle System Dynamics, 43(6-7), 455-483.
    [2] Ayasse, J. B., & Chollet, H. (2005). Determination of the wheel rail contact patch
        in semi-Hertzian conditions. Vehicle System Dynamics, 43(3), 161-172.
"""

import numpy as np
from scipy import optimize
from .kp import KikPiotrowski
from ..utils.geometry import calculate_curvatures

class ModifiedKikPiotrowski(KikPiotrowski):
    """
    Implémentation du modèle de contact non-hertzien Modified Kik-Piotrowski (MKP).
    
    Cette classe étend le modèle KP original avec des améliorations pour mieux traiter
    les contacts non-elliptiques, en particulier pour les contacts entre le boudin de
    la roue et le coin du rail.
    """
    
    def __init__(self, wheel_radius, rail_radius, wheel_youngs_modulus=210e9, 
                 rail_youngs_modulus=210e9, wheel_poisson_ratio=0.3, rail_poisson_ratio=0.3,
                 discretization=50, semi_axes_ratio_limit=5.0):
        """
        Initialise le modèle MKP avec les paramètres géométriques et matériels.
        
        Args:
            wheel_radius (float): Rayon de la roue en mètres.
            rail_radius (float): Rayon du rail en mètres.
            wheel_youngs_modulus (float): Module de Young de la roue en Pa. Par défaut 210 GPa.
            rail_youngs_modulus (float): Module de Young du rail en Pa. Par défaut 210 GPa.
            wheel_poisson_ratio (float): Coefficient de Poisson de la roue. Par défaut 0.3.
            rail_poisson_ratio (float): Coefficient de Poisson du rail. Par défaut 0.3.
            discretization (int): Nombre de points de discrétisation pour la zone de contact.
            semi_axes_ratio_limit (float): Limite du rapport des demi-axes pour la transition
                                          entre contact elliptique et non-elliptique.
        """
        super().__init__(wheel_radius, rail_radius, wheel_youngs_modulus, 
                        rail_youngs_modulus, wheel_poisson_ratio, rail_poisson_ratio,
                        discretization)
        self.semi_axes_ratio_limit = semi_axes_ratio_limit
        
    def calculate_contact_patch(self, normal_force, yaw_angle=0.0, lateral_displacement=0.0):
        """
        Calcule la géométrie de la zone de contact selon le modèle MKP.
        
        Le modèle MKP améliore le modèle KP original en tenant compte de la non-ellipticité
        de la zone de contact, en particulier pour les contacts avec un grand rapport d'axes.
        
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
                - 'is_non_elliptical': Booléen indiquant si le contact est non-elliptique
        """
        # Calcul des courbures principales
        curvatures = calculate_curvatures(self.wheel_radius, self.rail_radius, yaw_angle)
        
        # Calcul des paramètres de la zone de contact selon le modèle KP
        # Formule pour la demi-longueur a (direction longitudinale)
        a = (3 * normal_force / (4 * self.equivalent_youngs_modulus * 
             (curvatures['longitudinal'] + curvatures['lateral'])))**(1/3)
        
        # Formule pour la demi-longueur b (direction latérale)
        b = a * np.sqrt(curvatures['longitudinal'] / curvatures['lateral'])
        
        # Vérifier si le contact est fortement non-elliptique
        semi_axes_ratio = max(a/b, b/a)
        is_non_elliptical = semi_axes_ratio > self.semi_axes_ratio_limit
        
        # Appliquer les corrections du modèle MKP pour les contacts non-elliptiques
        if is_non_elliptical:
            # Facteur de correction pour la surface de contact
            # Basé sur l'article de Piotrowski & Chollet (2005)
            correction_factor = 1.0 - 0.25 * (1.0 - self.semi_axes_ratio_limit / semi_axes_ratio)
            
            # Ajuster les demi-longueurs
            if a > b:
                a = a * np.sqrt(correction_factor)
            else:
                b = b * np.sqrt(correction_factor)
        
        # Calcul de la surface de contact (modifiée pour les contacts non-elliptiques)
        if is_non_elliptical:
            # Pour les contacts non-elliptiques, la surface n'est plus exactement π*a*b
            area = np.pi * a * b * correction_factor
        else:
            area = np.pi * a * b
        
        # Calcul de la pression maximale (au centre de la zone de contact)
        # Pour les contacts non-elliptiques, la pression maximale est ajustée
        if is_non_elliptical:
            max_pressure = 1.5 * normal_force / area * (1.0 + 0.1 * (semi_axes_ratio - self.semi_axes_ratio_limit))
        else:
            max_pressure = 1.5 * normal_force / area
        
        # Calcul de la pression moyenne
        mean_pressure = normal_force / area
        
        # Création de la grille de points pour la zone de contact
        x = np.linspace(-a, a, self.discretization)
        y = np.linspace(-b, b, self.discretization)
        X, Y = np.meshgrid(x, y)
        
        # Calcul de la distribution de pression
        # Pour les contacts non-elliptiques, la distribution n'est plus semi-ellipsoïdale
        if is_non_elliptical:
            # Utiliser une fonction modifiée pour la distribution de pression
            # Basée sur l'article de Piotrowski & Chollet (2005)
            r = np.sqrt((X/a)**2 + (Y/b)**2)
            pressure_distribution = max_pressure * (1.0 - r**2)**(0.5 + 0.1 * (semi_axes_ratio - self.semi_axes_ratio_limit))
        else:
            # Pour les contacts elliptiques, utiliser la distribution semi-ellipsoïdale standard
            pressure_distribution = max_pressure * np.sqrt(1.0 - (X/a)**2 - (Y/b)**2)
        
        # Éliminer les valeurs négatives
        pressure_distribution[pressure_distribution < 0] = 0
        
        # Coordonnées des points de contact
        contact_points = np.column_stack((X.flatten(), Y.flatten()))
        
        return {
            'a': a,
            'b': b,
            'area': area,
            'max_pressure': max_pressure,
            'mean_pressure': mean_pressure,
            'contact_points': contact_points,
            'pressure_distribution': pressure_distribution.flatten(),
            'is_non_elliptical': is_non_elliptical,
            'semi_axes_ratio': semi_axes_ratio
        }
    
    def calculate_penetration(self, normal_force, yaw_angle=0.0):
        """
        Calcule la pénétration (interpénétration) entre la roue et le rail avec le modèle MKP.
        
        Args:
            normal_force (float): Force normale appliquée en Newtons.
            yaw_angle (float): Angle de lacet en radians. Par défaut 0.0.
            
        Returns:
            float: Pénétration en mètres.
        """
        # Calcul des courbures principales
        curvatures = calculate_curvatures(self.wheel_radius, self.rail_radius, yaw_angle)
        
        # Calcul de la pénétration selon le modèle MKP
        contact_patch = self.calculate_contact_patch(normal_force, yaw_angle)
        a = contact_patch['a']
        b = contact_patch['b']
        
        # Formule pour la pénétration, ajustée pour les contacts non-elliptiques
        if contact_patch['is_non_elliptical']:
            # Facteur de correction pour la pénétration
            correction_factor = 1.0 + 0.15 * (contact_patch['semi_axes_ratio'] - self.semi_axes_ratio_limit)
            penetration = a * b * (curvatures['longitudinal'] + curvatures['lateral']) / (a + b) * correction_factor
        else:
            # Formule standard pour les contacts elliptiques
            penetration = a**2 * (curvatures['longitudinal'] + curvatures['lateral']) / 3
        
        return penetration
    
    def calculate_normal_force(self, penetration, yaw_angle=0.0):
        """
        Calcule la force normale à partir de la pénétration avec le modèle MKP.
        
        Cette méthode utilise une approche itérative pour résoudre l'équation non-linéaire
        reliant la pénétration à la force normale dans le modèle MKP.
        
        Args:
            penetration (float): Pénétration en mètres.
            yaw_angle (float): Angle de lacet en radians. Par défaut 0.0.
            
        Returns:
            float: Force normale en Newtons.
        """
        # Fonction objectif pour l'optimisation
        def objective(force):
            calculated_penetration = self.calculate_penetration(force, yaw_angle)
            return (calculated_penetration - penetration)**2
        
        # Estimation initiale de la force normale (basée sur le modèle KP standard)
        curvatures = calculate_curvatures(self.wheel_radius, self.rail_radius, yaw_angle)
        a_estimate = np.sqrt(3 * penetration / (curvatures['longitudinal'] + curvatures['lateral']))
        initial_force = 4 * self.equivalent_youngs_modulus * (curvatures['longitudinal'] + 
                        curvatures['lateral']) * a_estimate**3 / 3
        
        # Résoudre l'équation non-linéaire pour trouver la force normale
        result = optimize.minimize(objective, initial_force, method='Nelder-Mead')
        
        if result.success:
            return result.x[0]
        else:
            # En cas d'échec de l'optimisation, retourner l'estimation initiale
            return initial_force
