"""
Intégration du modèle KP (Kik-Piotrowski) avec l'algorithme FastSim.

Ce module implémente l'intégration du modèle de contact normal KP avec
l'algorithme de contact tangentiel FastSim pour une simulation complète
du contact roue-rail.
"""

import numpy as np
from ..kp.kp import KikPiotrowski
from ..fastsim.fastsim import FastSim

class KPFastSim:
    """
    Intégration du modèle KP (Kik-Piotrowski) avec l'algorithme FastSim.
    
    Cette classe combine le modèle de contact normal KP avec l'algorithme
    de contact tangentiel FastSim pour une simulation complète du contact roue-rail.
    """
    
    def __init__(self, wheel_radius, rail_radius, wheel_youngs_modulus=210e9, 
                 rail_youngs_modulus=210e9, wheel_poisson_ratio=0.3, rail_poisson_ratio=0.3,
                 shear_modulus=8.0e10, friction_coefficient=0.3, discretization=50):
        """
        Initialise l'intégration KP-FastSim avec les paramètres géométriques et matériels.
        
        Args:
            wheel_radius (float): Rayon de la roue en mètres.
            rail_radius (float): Rayon du rail en mètres.
            wheel_youngs_modulus (float): Module de Young de la roue en Pa. Par défaut 210 GPa.
            rail_youngs_modulus (float): Module de Young du rail en Pa. Par défaut 210 GPa.
            wheel_poisson_ratio (float): Coefficient de Poisson de la roue. Par défaut 0.3.
            rail_poisson_ratio (float): Coefficient de Poisson du rail. Par défaut 0.3.
            shear_modulus (float): Module de cisaillement en Pa. Par défaut 80 GPa.
            friction_coefficient (float): Coefficient de frottement. Par défaut 0.3.
            discretization (int): Nombre de points de discrétisation pour la zone de contact.
        """
        # Initialiser le modèle de contact normal KP
        self.kp_model = KikPiotrowski(
            wheel_radius=wheel_radius,
            rail_radius=rail_radius,
            wheel_youngs_modulus=wheel_youngs_modulus,
            rail_youngs_modulus=rail_youngs_modulus,
            wheel_poisson_ratio=wheel_poisson_ratio,
            rail_poisson_ratio=rail_poisson_ratio,
            discretization=discretization
        )
        
        # Initialiser l'algorithme de contact tangentiel FastSim
        self.fastsim_model = FastSim(
            poisson_ratio=(wheel_poisson_ratio + rail_poisson_ratio) / 2,  # Moyenne des deux
            shear_modulus=shear_modulus,
            friction_coefficient=friction_coefficient,
            discretization=discretization
        )
    
    def solve_contact_problem(self, normal_force=None, penetration=None, creepages=None, 
                             yaw_angle=0.0, lateral_displacement=0.0):
        """
        Résout le problème de contact complet (normal et tangentiel).
        
        Args:
            normal_force (float, optional): Force normale en Newtons.
            penetration (float, optional): Pénétration en mètres.
            creepages (dict, optional): Dictionnaire contenant les glissements:
                - 'longitudinal': Glissement longitudinal (adimensionnel)
                - 'lateral': Glissement latéral (adimensionnel)
                - 'spin': Spin (1/m)
            yaw_angle (float): Angle de lacet en radians. Par défaut 0.0.
            lateral_displacement (float): Déplacement latéral en mètres. Par défaut 0.0.
            
        Returns:
            dict: Dictionnaire contenant les résultats du problème de contact.
            
        Raises:
            ValueError: Si ni normal_force ni penetration ne sont fournis.
        """
        # Résoudre le problème de contact normal avec le modèle KP
        normal_solution = self.kp_model.solve_contact_problem(
            normal_force=normal_force,
            penetration=penetration,
            yaw_angle=yaw_angle,
            lateral_displacement=lateral_displacement
        )
        
        # Si aucun glissement n'est fourni, retourner uniquement la solution normale
        if creepages is None:
            return {
                'normal_solution': normal_solution,
                'tangential_solution': None
            }
        
        # Résoudre le problème de contact tangentiel avec FastSim
        tangential_solution = self.fastsim_model.solve_tangential_problem(
            contact_patch=normal_solution['contact_patch'],
            creepages=creepages,
            normal_force=normal_solution['normal_force']
        )
        
        # Combiner les résultats
        return {
            'normal_solution': normal_solution,
            'tangential_solution': tangential_solution,
            'total_forces': {
                'normal': normal_solution['normal_force'],
                'tangential_x': tangential_solution['tangential_forces'][0],
                'tangential_y': tangential_solution['tangential_forces'][1],
                'moment': tangential_solution['moment']
            }
        }
    
    def calculate_contact_forces(self, penetration, longitudinal_creepage, lateral_creepage, 
                                spin_creepage=0.0, yaw_angle=0.0, lateral_displacement=0.0):
        """
        Calcule les forces de contact à partir de la pénétration et des glissements.
        
        Cette méthode est une interface simplifiée pour le calcul des forces de contact.
        
        Args:
            penetration (float): Pénétration en mètres.
            longitudinal_creepage (float): Glissement longitudinal (adimensionnel).
            lateral_creepage (float): Glissement latéral (adimensionnel).
            spin_creepage (float): Spin (1/m). Par défaut 0.0.
            yaw_angle (float): Angle de lacet en radians. Par défaut 0.0.
            lateral_displacement (float): Déplacement latéral en mètres. Par défaut 0.0.
            
        Returns:
            tuple: (Fn, Fx, Fy, Mz) forces normales, tangentielles et moment en N et N.m.
        """
        # Créer le dictionnaire des glissements
        creepages = {
            'longitudinal': longitudinal_creepage,
            'lateral': lateral_creepage,
            'spin': spin_creepage
        }
        
        # Résoudre le problème de contact complet
        solution = self.solve_contact_problem(
            penetration=penetration,
            creepages=creepages,
            yaw_angle=yaw_angle,
            lateral_displacement=lateral_displacement
        )
        
        # Extraire les forces et le moment
        Fn = solution['total_forces']['normal']
        Fx = solution['total_forces']['tangential_x']
        Fy = solution['total_forces']['tangential_y']
        Mz = solution['total_forces']['moment']
        
        return Fn, Fx, Fy, Mz
