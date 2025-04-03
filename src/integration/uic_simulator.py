"""
Intégration des modèles KP-FastSim-USFD avec support des profils UIC.

Ce module étend la classe KPFastSimUSFD pour prendre en charge les profils UIC
de roue et rail dans les simulations de contact et d'usure.
"""

import numpy as np
from ..integration.full_model import KPFastSimUSFD
from ..utils.uic_profile import UICProfileLoader

class UICProfileSimulator:
    """
    Classe pour simuler le contact roue-rail et l'usure avec des profils UIC.
    
    Cette classe combine le modèle KP-FastSim-USFD avec le chargeur de profils UIC
    pour permettre des simulations réalistes avec des profils de roue et rail standards.
    """
    
    def __init__(self, wheel_youngs_modulus=210e9, rail_youngs_modulus=210e9, 
                 wheel_poisson_ratio=0.3, rail_poisson_ratio=0.3, shear_modulus=8.0e10, 
                 friction_coefficient=0.3, discretization=50, wheel_material="R8T", 
                 rail_material="UIC60 900A"):
        """
        Initialise le simulateur avec les paramètres matériels.
        
        Args:
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
        # Initialiser le chargeur de profils UIC
        self.profile_loader = UICProfileLoader()
        
        # Stocker les paramètres matériels
        self.wheel_youngs_modulus = wheel_youngs_modulus
        self.rail_youngs_modulus = rail_youngs_modulus
        self.wheel_poisson_ratio = wheel_poisson_ratio
        self.rail_poisson_ratio = rail_poisson_ratio
        self.shear_modulus = shear_modulus
        self.friction_coefficient = friction_coefficient
        self.discretization = discretization
        self.wheel_material = wheel_material
        self.rail_material = rail_material
        
        # Dictionnaire pour stocker les modèles KP-FastSim-USFD créés
        self.models = {}
    
    def load_profiles(self, wheel_file_path, rail_file_path, wheel_name=None, rail_name=None):
        """
        Charge les profils UIC de roue et rail.
        
        Args:
            wheel_file_path (str): Chemin vers le fichier de profil UIC de la roue.
            rail_file_path (str): Chemin vers le fichier de profil UIC du rail.
            wheel_name (str, optional): Nom à donner au profil de roue. Si None, utilise le nom du fichier.
            rail_name (str, optional): Nom à donner au profil de rail. Si None, utilise le nom du fichier.
            
        Returns:
            tuple: (wheel_profile, rail_profile) Dictionnaires contenant les données des profils.
        """
        # Charger les profils
        wheel_profile = self.profile_loader.load_wheel_profile(wheel_file_path, wheel_name)
        rail_profile = self.profile_loader.load_rail_profile(rail_file_path, rail_name)
        
        return wheel_profile, rail_profile
    
    def create_model(self, wheel_profile_name, rail_profile_name, lateral_displacement=0.0, model_name=None):
        """
        Crée un modèle KP-FastSim-USFD pour une paire de profils UIC.
        
        Args:
            wheel_profile_name (str): Nom du profil de roue.
            rail_profile_name (str): Nom du profil de rail.
            lateral_displacement (float): Déplacement latéral de la roue par rapport au rail en mm.
            model_name (str, optional): Nom à donner au modèle. Si None, utilise une combinaison des noms de profils.
            
        Returns:
            KPFastSimUSFD: Modèle KP-FastSim-USFD configuré pour les profils donnés.
        """
        # Si aucun nom de modèle n'est fourni, créer un nom à partir des noms de profils
        if model_name is None:
            model_name = f"{wheel_profile_name}_{rail_profile_name}_{lateral_displacement:.1f}mm"
        
        # Calculer la géométrie de contact
        contact_geometry = self.profile_loader.calculate_contact_geometry(
            wheel_profile_name, rail_profile_name, lateral_displacement
        )
        
        # Extraire les rayons de courbure
        wheel_radius = 1.0 / contact_geometry['wheel_curvature'] if contact_geometry['wheel_curvature'] > 0 else 1e6
        rail_radius = 1.0 / contact_geometry['rail_curvature'] if contact_geometry['rail_curvature'] > 0 else 1e6
        
        # Convertir de mm à m
        wheel_radius /= 1000.0
        rail_radius /= 1000.0
        
        # Créer le modèle KP-FastSim-USFD
        model = KPFastSimUSFD(
            wheel_radius=wheel_radius,
            rail_radius=rail_radius,
            wheel_youngs_modulus=self.wheel_youngs_modulus,
            rail_youngs_modulus=self.rail_youngs_modulus,
            wheel_poisson_ratio=self.wheel_poisson_ratio,
            rail_poisson_ratio=self.rail_poisson_ratio,
            shear_modulus=self.shear_modulus,
            friction_coefficient=self.friction_coefficient,
            discretization=self.discretization,
            wheel_material=self.wheel_material,
            rail_material=self.rail_material
        )
        
        # Stocker le modèle
        self.models[model_name] = {
            'model': model,
            'wheel_profile_name': wheel_profile_name,
            'rail_profile_name': rail_profile_name,
            'lateral_displacement': lateral_displacement,
            'contact_geometry': contact_geometry
        }
        
        return model
    
    def get_model(self, model_name):
        """
        Récupère un modèle KP-FastSim-USFD par son nom.
        
        Args:
            model_name (str): Nom du modèle à récupérer.
            
        Returns:
            KPFastSimUSFD: Modèle KP-FastSim-USFD.
            
        Raises:
            KeyError: Si le modèle n'existe pas.
        """
        if model_name not in self.models:
            raise KeyError(f"Le modèle {model_name} n'a pas été créé.")
        
        return self.models[model_name]['model']
    
    def simulate_contact_and_wear(self, model_name, normal_force=None, penetration=None, 
                                 longitudinal_creepage=0.0, lateral_creepage=0.0, 
                                 spin_creepage=0.0, sliding_distance=1000.0, density=7850.0):
        """
        Simule le contact roue-rail et prédit l'usure qui en résulte.
        
        Args:
            model_name (str): Nom du modèle à utiliser.
            normal_force (float, optional): Force normale en Newtons.
            penetration (float, optional): Pénétration en mètres.
            longitudinal_creepage (float): Glissement longitudinal (adimensionnel). Par défaut 0.0.
            lateral_creepage (float): Glissement latéral (adimensionnel). Par défaut 0.0.
            spin_creepage (float): Spin (1/m). Par défaut 0.0.
            sliding_distance (float): Distance de glissement en mètres. Par défaut 1000 m.
            density (float): Densité du matériau de la roue en kg/m³. Par défaut 7850 kg/m³ (acier).
            
        Returns:
            dict: Dictionnaire contenant les résultats de la simulation:
                - 'contact_solution': Solution du problème de contact
                - 'wear_solution': Solution du problème d'usure
                - 'contact_geometry': Géométrie de contact entre les profils
                
        Raises:
            KeyError: Si le modèle n'existe pas.
        """
        # Récupérer le modèle
        model = self.get_model(model_name)
        
        # Créer le dictionnaire des glissements
        creepages = {
            'longitudinal': longitudinal_creepage,
            'lateral': lateral_creepage,
            'spin': spin_creepage
        }
        
        # Simuler le contact et l'usure
        simulation_result = model.simulate_contact_and_wear(
            normal_force=normal_force,
            penetration=penetration,
            creepages=creepages,
            sliding_distance=sliding_distance,
            density=density
        )
        
        # Ajouter la géométrie de contact
        simulation_result['contact_geometry'] = self.models[model_name]['contact_geometry']
        
        return simulation_result
    
    def calculate_equivalent_conicity(self, wheel_profile_name, rail_profile_name, 
                                     lateral_displacement_range=(-10.0, 10.0), steps=20):
        """
        Calcule la conicité équivalente pour une paire de profils UIC.
        
        Args:
            wheel_profile_name (str): Nom du profil de roue.
            rail_profile_name (str): Nom du profil de rail.
            lateral_displacement_range (tuple): Plage de déplacements latéraux en mm.
            steps (int): Nombre de points pour le calcul.
            
        Returns:
            float: Conicité équivalente (sans unité).
        """
        # Récupérer les profils
        wheel_profile = self.profile_loader.get_profile(wheel_profile_name)
        rail_profile = self.profile_loader.get_profile(rail_profile_name)
        
        # Calculer la géométrie de contact pour chaque déplacement latéral
        min_displacement, max_displacement = lateral_displacement_range
        displacements = np.linspace(min_displacement, max_displacement, steps)
        
        # Calculer le rayon de roulement effectif pour chaque déplacement
        rolling_radii = []
        for displacement in displacements:
            contact_geometry = self.profile_loader.calculate_contact_geometry(
                wheel_profile_name, rail_profile_name, displacement
            )
            rolling_radii.append(contact_geometry['rolling_radius'])
        
        # Convertir en tableau numpy
        rolling_radii = np.array(rolling_radii)
        
        # Calculer la différence de rayon de roulement
        delta_r = rolling_radii - rolling_radii[steps // 2]  # Différence par rapport au point central
        
        # Calculer la conicité équivalente
        # La conicité équivalente est la pente moyenne de la fonction de différence de rayon
        # par rapport au déplacement latéral
        equivalent_conicity = np.mean(np.diff(delta_r) / np.diff(displacements))
        
        return equivalent_conicity
