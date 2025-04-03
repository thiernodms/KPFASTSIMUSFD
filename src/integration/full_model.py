"""
Intégration complète des modèles KP, FastSim et USFD pour la simulation du contact roue-rail
et la prédiction de l'usure.

Ce module implémente l'intégration complète des trois modèles pour une simulation
de bout en bout du contact roue-rail et de l'usure qui en résulte.
"""

import numpy as np
from ..kp.kp import KikPiotrowski
from ..fastsim.fastsim import FastSim
from ..usfd.usfd import USFDWearModel

class KPFastSimUSFD:
    """
    Intégration complète des modèles KP, FastSim et USFD.
    
    Cette classe combine le modèle de contact normal KP, l'algorithme de contact
    tangentiel FastSim et le modèle d'usure USFD pour une simulation complète
    du contact roue-rail et de la prédiction de l'usure.
    """
    
    def __init__(self, wheel_radius, rail_radius, wheel_youngs_modulus=210e9, 
                 rail_youngs_modulus=210e9, wheel_poisson_ratio=0.3, rail_poisson_ratio=0.3,
                 shear_modulus=8.0e10, friction_coefficient=0.3, discretization=50,
                 wheel_material="R8T", rail_material="UIC60 900A"):
        """
        Initialise l'intégration KP-FastSim-USFD avec les paramètres géométriques et matériels.
        
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
            wheel_material (str): Matériau de la roue. Par défaut "R8T".
            rail_material (str): Matériau du rail. Par défaut "UIC60 900A".
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
        
        # Initialiser le modèle d'usure USFD
        self.usfd_model = USFDWearModel(
            wheel_material=wheel_material,
            rail_material=rail_material
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
                'tangential_solution': None,
                'wear_solution': None
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
    
    def predict_wear(self, contact_solution, sliding_distance, density=7850.0):
        """
        Prédit l'usure de la roue en utilisant le modèle USFD.
        
        Args:
            contact_solution (dict): Solution du problème de contact.
            sliding_distance (float): Distance de glissement en mètres.
            density (float): Densité du matériau de la roue en kg/m³. Par défaut 7850 kg/m³ (acier).
            
        Returns:
            dict: Dictionnaire contenant les résultats de la prédiction d'usure:
                - 'wear_rate': Taux d'usure moyen en μg/(m·mm²)
                - 'wear_depth': Profondeur d'usure moyenne en mm
                - 'wear_volume': Volume d'usure total en mm³
                - 'local_wear_rate': Taux d'usure local aux points de contact
                - 'local_wear_depth': Profondeur d'usure locale aux points de contact
                
        Raises:
            ValueError: Si la solution du problème de contact ne contient pas de solution tangentielle.
        """
        # Vérifier que la solution du problème de contact contient une solution tangentielle
        if contact_solution['tangential_solution'] is None:
            raise ValueError("La solution du problème de contact ne contient pas de solution tangentielle.")
        
        # Extraire les informations nécessaires de la solution du problème de contact
        normal_force = contact_solution['total_forces']['normal']
        tangential_force_x = contact_solution['total_forces']['tangential_x']
        tangential_force_y = contact_solution['total_forces']['tangential_y']
        tangential_force = (tangential_force_x, tangential_force_y)
        
        # Extraire les glissements locaux
        slip_x, slip_y = contact_solution['tangential_solution']['slip']
        
        # Extraire la surface de contact
        contact_area = contact_solution['normal_solution']['contact_patch']['area']
        
        # Calculer la puissance de frottement locale (T-gamma) pour chaque point de contact
        # Pour simplifier, nous utilisons la norme des forces tangentielles et des glissements
        slip_norm = np.sqrt(slip_x**2 + slip_y**2)
        tangential_force_norm = np.sqrt(tangential_force_x**2 + tangential_force_y**2)
        
        # Calculer T-gamma local pour chaque point de contact
        local_tgamma = []
        for i in range(len(slip_x)):
            # Calculer la contribution de ce point à la force tangentielle totale
            # Nous supposons une distribution proportionnelle à la norme du glissement
            point_tangential_force = tangential_force_norm * slip_norm[i] / np.sum(slip_norm)
            point_tgamma = point_tangential_force * slip_norm[i]
            local_tgamma.append(point_tgamma)
        
        local_tgamma = np.array(local_tgamma)
        
        # Calculer le taux d'usure local pour chaque point de contact
        local_wear_rate = self.usfd_model.calculate_wear_rate(local_tgamma)
        
        # Calculer la profondeur d'usure locale pour chaque point de contact
        local_wear_depth = self.usfd_model.calculate_wear_depth(local_tgamma, sliding_distance, density)
        
        # Calculer le taux d'usure moyen
        average_wear_rate = np.mean(local_wear_rate)
        
        # Calculer la profondeur d'usure moyenne
        average_wear_depth = np.mean(local_wear_depth)
        
        # Calculer le volume d'usure total
        wear_volume = self.usfd_model.calculate_wear_volume(
            np.mean(local_tgamma),
            sliding_distance,
            contact_area,
            density
        )
        
        return {
            'wear_rate': average_wear_rate,
            'wear_depth': average_wear_depth,
            'wear_volume': wear_volume,
            'local_wear_rate': local_wear_rate,
            'local_wear_depth': local_wear_depth
        }
    
    def simulate_contact_and_wear(self, normal_force=None, penetration=None, creepages=None,
                                 yaw_angle=0.0, lateral_displacement=0.0, sliding_distance=1000.0,
                                 density=7850.0):
        """
        Simule le contact roue-rail et prédit l'usure qui en résulte.
        
        Cette méthode combine la résolution du problème de contact et la prédiction de l'usure
        en une seule étape.
        
        Args:
            normal_force (float, optional): Force normale en Newtons.
            penetration (float, optional): Pénétration en mètres.
            creepages (dict, optional): Dictionnaire contenant les glissements:
                - 'longitudinal': Glissement longitudinal (adimensionnel)
                - 'lateral': Glissement latéral (adimensionnel)
                - 'spin': Spin (1/m)
            yaw_angle (float): Angle de lacet en radians. Par défaut 0.0.
            lateral_displacement (float): Déplacement latéral en mètres. Par défaut 0.0.
            sliding_distance (float): Distance de glissement en mètres. Par défaut 1000 m.
            density (float): Densité du matériau de la roue en kg/m³. Par défaut 7850 kg/m³ (acier).
            
        Returns:
            dict: Dictionnaire contenant les résultats de la simulation:
                - 'contact_solution': Solution du problème de contact
                - 'wear_solution': Solution du problème d'usure
        """
        # Résoudre le problème de contact
        contact_solution = self.solve_contact_problem(
            normal_force=normal_force,
            penetration=penetration,
            creepages=creepages,
            yaw_angle=yaw_angle,
            lateral_displacement=lateral_displacement
        )
        
        # Si aucun glissement n'est fourni, retourner uniquement la solution de contact
        if creepages is None:
            return {
                'contact_solution': contact_solution,
                'wear_solution': None
            }
        
        # Prédire l'usure
        wear_solution = self.predict_wear(
            contact_solution=contact_solution,
            sliding_distance=sliding_distance,
            density=density
        )
        
        # Combiner les résultats
        return {
            'contact_solution': contact_solution,
            'wear_solution': wear_solution
        }
