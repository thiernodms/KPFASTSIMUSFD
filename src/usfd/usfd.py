"""
Implémentation du modèle d'usure USFD (University of Sheffield) pour les roues ferroviaires.

Ce module implémente le modèle d'usure USFD basé sur la méthode T-gamma pour prédire
l'usure des roues ferroviaires en fonction de l'énergie de frottement dissipée dans
la zone de contact roue-rail.

Références:
    [1] Lewis, R., & Dwyer-Joyce, R. S. (2004). Wear mechanisms and transitions in railway
        wheel steels. Proceedings of the Institution of Mechanical Engineers, Part J: Journal
        of Engineering Tribology, 218(6), 467-478.
    [2] Braghin, F., Lewis, R., Dwyer-Joyce, R. S., & Bruni, S. (2006). A mathematical model
        to predict railway wheel profile evolution due to wear. Wear, 261(11-12), 1253-1264.
    [3] Tassini, N., Quost, X., Lewis, R., Dwyer-Joyce, R., Ariaudo, C., & Kuka, N. (2010).
        A numerical model of twin disc test arrangement for the evaluation of railway wheel
        wear prediction methods. Wear, 268(5-6), 660-667.
"""

import numpy as np

class USFDWearModel:
    """
    Implémentation du modèle d'usure USFD (University of Sheffield) pour les roues ferroviaires.
    
    Ce modèle calcule le taux d'usure en fonction de la puissance de frottement locale (T-gamma)
    en utilisant une fonction par morceaux avec trois régimes d'usure (léger, sévère et catastrophique).
    """
    
    def __init__(self, wheel_material="R8T", rail_material="UIC60 900A"):
        """
        Initialise le modèle d'usure USFD avec les paramètres matériels.
        
        Args:
            wheel_material (str): Matériau de la roue. Par défaut "R8T".
            rail_material (str): Matériau du rail. Par défaut "UIC60 900A".
        """
        self.wheel_material = wheel_material
        self.rail_material = rail_material
        
        # Définir les coefficients du modèle d'usure en fonction des matériaux
        # Ces coefficients sont basés sur des essais expérimentaux
        # Référence: Lewis & Dwyer-Joyce (2004)
        
        if wheel_material == "R8T" and rail_material == "UIC60 900A":
            # Coefficients pour la combinaison R8T (roue) / UIC60 900A (rail)
            # Régime léger (mild)
            self.K1 = 5.3  # μg/m/mm²
            self.T1 = 10.4  # N/mm²
            
            # Régime sévère (severe)
            self.K2 = 55.0  # μg/m/mm²
            self.T2 = 77.2  # N/mm²
            
            # Régime catastrophique (catastrophic)
            self.K3 = 61.9  # μg/m/mm²
            self.C3 = 4778.7  # Constante pour le régime catastrophique
        else:
            # Coefficients par défaut (basés sur R8T/UIC60 900A)
            # Dans une implémentation réelle, il faudrait ajouter d'autres combinaisons de matériaux
            self.K1 = 5.3
            self.T1 = 10.4
            self.K2 = 55.0
            self.T2 = 77.2
            self.K3 = 61.9
            self.C3 = 4778.7
            print(f"Warning: No specific wear coefficients for {wheel_material}/{rail_material}. Using default values.")
    
    def calculate_wear_rate(self, tgamma):
        """
        Calcule le taux d'usure en fonction de la puissance de frottement locale (T-gamma).
        
        Args:
            tgamma (float or array): Puissance de frottement locale en N/mm².
            
        Returns:
            float or array: Taux d'usure en μg/(m·mm²).
        """
        # Fonction par morceaux avec trois régimes d'usure
        # Référence: Braghin et al. (2006)
        
        # Convertir en tableau numpy si ce n'est pas déjà le cas
        if not isinstance(tgamma, np.ndarray):
            tgamma = np.array([tgamma])
            scalar_input = True
        else:
            scalar_input = False
        
        # Initialiser le tableau de taux d'usure
        wear_rate = np.zeros_like(tgamma)
        
        # Régime léger (mild)
        mask_mild = tgamma < self.T1
        wear_rate[mask_mild] = self.K1 * tgamma[mask_mild]
        
        # Régime sévère (severe)
        mask_severe = (tgamma >= self.T1) & (tgamma <= self.T2)
        wear_rate[mask_severe] = self.K2
        
        # Régime catastrophique (catastrophic)
        mask_catastrophic = tgamma > self.T2
        wear_rate[mask_catastrophic] = self.K3 * tgamma[mask_catastrophic] - self.C3
        
        # Retourner un scalaire si l'entrée était un scalaire
        if scalar_input:
            return wear_rate[0]
        else:
            return wear_rate
    
    def calculate_wear_depth(self, tgamma, sliding_distance, density=7850.0):
        """
        Calcule la profondeur d'usure en fonction de la puissance de frottement locale
        et de la distance de glissement.
        
        Args:
            tgamma (float or array): Puissance de frottement locale en N/mm².
            sliding_distance (float): Distance de glissement en mètres.
            density (float): Densité du matériau de la roue en kg/m³. Par défaut 7850 kg/m³ (acier).
            
        Returns:
            float or array: Profondeur d'usure en mm.
        """
        # Calculer le taux d'usure en μg/(m·mm²)
        wear_rate = self.calculate_wear_rate(tgamma)
        
        # Convertir le taux d'usure en mm³/(m·mm²)
        # 1 μg = 1e-9 kg, donc wear_rate / (density * 1e-9) donne le volume en mm³
        volumetric_wear_rate = wear_rate / (density * 1e-9)
        
        # Calculer la profondeur d'usure en mm
        # La profondeur d'usure est le volume d'usure divisé par la surface
        # Pour une surface unitaire (1 mm²), la profondeur est égale au volume
        wear_depth = volumetric_wear_rate * sliding_distance / 1000.0  # Conversion m en mm
        
        return wear_depth
    
    def calculate_wear_volume(self, tgamma, sliding_distance, contact_area, density=7850.0):
        """
        Calcule le volume d'usure en fonction de la puissance de frottement locale,
        de la distance de glissement et de la surface de contact.
        
        Args:
            tgamma (float or array): Puissance de frottement locale en N/mm².
            sliding_distance (float): Distance de glissement en mètres.
            contact_area (float): Surface de contact en mm².
            density (float): Densité du matériau de la roue en kg/m³. Par défaut 7850 kg/m³ (acier).
            
        Returns:
            float: Volume d'usure en mm³.
        """
        # Calculer le taux d'usure en μg/(m·mm²)
        wear_rate = self.calculate_wear_rate(tgamma)
        
        # Si tgamma est un tableau, calculer la moyenne pondérée du taux d'usure
        if isinstance(wear_rate, np.ndarray) and len(wear_rate) > 1:
            # Dans une implémentation réelle, il faudrait tenir compte de la distribution
            # de la puissance de frottement sur la zone de contact
            wear_rate = np.mean(wear_rate)
        
        # Convertir le taux d'usure en mm³/(m·mm²)
        volumetric_wear_rate = wear_rate / (density * 1e-9)
        
        # Calculer le volume d'usure en mm³
        wear_volume = volumetric_wear_rate * sliding_distance * contact_area / 1000.0  # Conversion m en mm
        
        return wear_volume
    
    def calculate_tgamma(self, tangential_force, creepage, contact_area=None):
        """
        Calcule la puissance de frottement locale (T-gamma) à partir de la force tangentielle
        et du glissement.
        
        Args:
            tangential_force (float or tuple): Force tangentielle en Newtons. Si tuple, (Fx, Fy).
            creepage (float or tuple): Glissement (adimensionnel). Si tuple, (longitudinal, lateral).
            contact_area (float, optional): Surface de contact en mm². Si fournie, la puissance
                                           de frottement est normalisée par la surface.
            
        Returns:
            float: Puissance de frottement locale en N/mm².
        """
        # Calculer la puissance de frottement totale
        if isinstance(tangential_force, tuple) and isinstance(creepage, tuple):
            # Cas vectoriel
            Fx, Fy = tangential_force
            cx, cy = creepage
            tgamma_total = abs(Fx * cx + Fy * cy)
        else:
            # Cas scalaire
            tgamma_total = abs(tangential_force * creepage)
        
        # Normaliser par la surface de contact si fournie
        if contact_area is not None:
            tgamma_local = tgamma_total / contact_area
        else:
            tgamma_local = tgamma_total
        
        return tgamma_local
