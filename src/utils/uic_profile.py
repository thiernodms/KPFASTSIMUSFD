"""
Module pour le chargement et le traitement des profils UIC de roue et rail.

Ce module fournit des fonctionnalités pour charger, analyser et utiliser
les profils UIC standard pour les roues et les rails dans les simulations
de contact roue-rail.
"""

import numpy as np
import os
import re

class UICProfileLoader:
    """
    Classe pour charger et traiter les profils UIC de roue et rail.
    
    Cette classe permet de charger les fichiers de profil UIC, d'extraire
    les coordonnées et de calculer les propriétés géométriques nécessaires
    pour les simulations de contact roue-rail.
    """
    
    def __init__(self):
        """
        Initialise le chargeur de profils UIC.
        """
        # Dictionnaire pour stocker les profils chargés
        self.profiles = {}
    
    def load_wheel_profile(self, file_path, profile_name=None):
        """
        Charge un profil de roue à partir d'un fichier UIC.
        
        Args:
            file_path (str): Chemin vers le fichier de profil UIC de la roue.
            profile_name (str, optional): Nom à donner au profil. Si None, utilise le nom du fichier.
            
        Returns:
            dict: Dictionnaire contenant les données du profil de roue.
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
            ValueError: Si le fichier n'est pas au format UIC attendu.
        """
        # Vérifier que le fichier existe
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
        
        # Si aucun nom de profil n'est fourni, utiliser le nom du fichier
        if profile_name is None:
            profile_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Lire le fichier
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Extraire les coordonnées
        x = []
        y = []
        
        for line in lines:
            # Ignorer les lignes de commentaire
            if line.strip().startswith('#'):
                continue
            
            # Ignorer les lignes vides
            if not line.strip():
                continue
            
            # Extraire les coordonnées
            try:
                # Diviser la ligne en valeurs
                values = re.split(r'\s+', line.strip())
                
                # S'assurer qu'il y a au moins deux valeurs
                if len(values) >= 2:
                    x_val = float(values[0])
                    y_val = float(values[1])
                    x.append(x_val)
                    y.append(y_val)
            except ValueError:
                # Ignorer les lignes qui ne peuvent pas être converties en nombres
                continue
        
        # Vérifier que des coordonnées ont été extraites
        if not x or not y:
            raise ValueError(f"Aucune coordonnée n'a pu être extraite du fichier {file_path}.")
        
        # Convertir en tableaux numpy
        x = np.array(x)
        y = np.array(y)
        
        # Créer le dictionnaire du profil
        profile = {
            'type': 'wheel',
            'name': profile_name,
            'x': x,
            'y': y,
            'file_path': file_path
        }
        
        # Calculer les propriétés géométriques
        profile.update(self._calculate_wheel_properties(x, y))
        
        # Stocker le profil
        self.profiles[profile_name] = profile
        
        return profile
    
    def load_rail_profile(self, file_path, profile_name=None):
        """
        Charge un profil de rail à partir d'un fichier UIC.
        
        Args:
            file_path (str): Chemin vers le fichier de profil UIC du rail.
            profile_name (str, optional): Nom à donner au profil. Si None, utilise le nom du fichier.
            
        Returns:
            dict: Dictionnaire contenant les données du profil de rail.
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
            ValueError: Si le fichier n'est pas au format UIC attendu.
        """
        # Vérifier que le fichier existe
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
        
        # Si aucun nom de profil n'est fourni, utiliser le nom du fichier
        if profile_name is None:
            profile_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Lire le fichier
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Extraire les coordonnées
        x = []
        y = []
        
        for line in lines:
            # Ignorer les lignes de commentaire
            if line.strip().startswith('#'):
                continue
            
            # Ignorer les lignes vides
            if not line.strip():
                continue
            
            # Extraire les coordonnées
            try:
                # Diviser la ligne en valeurs
                values = re.split(r'\s+', line.strip())
                
                # S'assurer qu'il y a au moins deux valeurs
                if len(values) >= 2:
                    x_val = float(values[0])
                    y_val = float(values[1])
                    x.append(x_val)
                    y.append(y_val)
            except ValueError:
                # Ignorer les lignes qui ne peuvent pas être converties en nombres
                continue
        
        # Vérifier que des coordonnées ont été extraites
        if not x or not y:
            raise ValueError(f"Aucune coordonnée n'a pu être extraite du fichier {file_path}.")
        
        # Convertir en tableaux numpy
        x = np.array(x)
        y = np.array(y)
        
        # Créer le dictionnaire du profil
        profile = {
            'type': 'rail',
            'name': profile_name,
            'x': x,
            'y': y,
            'file_path': file_path
        }
        
        # Calculer les propriétés géométriques
        profile.update(self._calculate_rail_properties(x, y))
        
        # Stocker le profil
        self.profiles[profile_name] = profile
        
        return profile
    
    def _calculate_wheel_properties(self, x, y):
        """
        Calcule les propriétés géométriques d'un profil de roue.
        
        Args:
            x (array): Coordonnées x du profil.
            y (array): Coordonnées y du profil.
            
        Returns:
            dict: Dictionnaire contenant les propriétés géométriques.
        """
        # Trouver le point le plus bas du profil (tread)
        tread_index = np.argmin(y)
        tread_x = x[tread_index]
        tread_y = y[tread_index]
        
        # Trouver le point le plus à gauche du profil (flange)
        flange_index = np.argmin(x)
        flange_x = x[flange_index]
        flange_y = y[flange_index]
        
        # Calculer le rayon de roulement nominal (distance du centre de la roue au tread)
        # Note: Dans un profil UIC, le centre de la roue est généralement à (0, 0)
        rolling_radius = abs(tread_y)
        
        # Calculer la hauteur du boudin (flange height)
        flange_height = flange_y - tread_y
        
        # Calculer l'épaisseur du boudin (flange thickness)
        # C'est la distance horizontale entre le point le plus à gauche du boudin
        # et le point où le boudin rencontre le tread
        flange_thickness = abs(flange_x - tread_x)
        
        # Estimer le rayon du boudin (flange radius)
        # Pour une estimation simple, nous utilisons les points autour du boudin
        # et ajustons un cercle
        # Note: Cette estimation est très simplifiée et pourrait être améliorée
        flange_points_indices = np.where(x < (flange_x + flange_thickness / 2))[0]
        flange_points_x = x[flange_points_indices]
        flange_points_y = y[flange_points_indices]
        
        # Estimer le rayon du boudin en utilisant la courbure moyenne
        flange_radius = 10.0  # Valeur par défaut
        if len(flange_points_x) > 3:
            # Calculer les différences finies pour estimer la courbure
            dx = np.diff(flange_points_x)
            dy = np.diff(flange_points_y)
            ds = np.sqrt(dx**2 + dy**2)
            
            # Éviter la division par zéro
            ds[ds < 1e-10] = 1e-10
            
            # Calculer les angles
            angles = np.arctan2(dy, dx)
            dtheta = np.diff(angles)
            
            # Normaliser les angles entre -pi et pi
            dtheta = np.mod(dtheta + np.pi, 2 * np.pi) - np.pi
            
            # Calculer la courbure
            curvature = dtheta / ds[:-1]
            
            # Éviter les valeurs aberrantes
            valid_curvature = curvature[np.abs(curvature) < 1.0]
            
            if len(valid_curvature) > 0:
                # La courbure moyenne est l'inverse du rayon
                mean_curvature = np.mean(np.abs(valid_curvature))
                if mean_curvature > 1e-10:
                    flange_radius = 1.0 / mean_curvature
        
        return {
            'rolling_radius': rolling_radius,
            'flange_height': flange_height,
            'flange_thickness': flange_thickness,
            'flange_radius': flange_radius,
            'tread_position': (tread_x, tread_y),
            'flange_position': (flange_x, flange_y)
        }
    
    def _calculate_rail_properties(self, x, y):
        """
        Calcule les propriétés géométriques d'un profil de rail.
        
        Args:
            x (array): Coordonnées x du profil.
            y (array): Coordonnées y du profil.
            
        Returns:
            dict: Dictionnaire contenant les propriétés géométriques.
        """
        # Trouver le point le plus haut du profil (rail head)
        head_index = np.argmax(y)
        head_x = x[head_index]
        head_y = y[head_index]
        
        # Calculer la largeur de la tête du rail
        # Pour cela, nous cherchons les points où la hauteur est proche de celle de la tête
        head_threshold = head_y - 2.0  # 2 mm en dessous de la tête
        head_points_indices = np.where(y > head_threshold)[0]
        head_points_x = x[head_points_indices]
        
        # La largeur de la tête est la distance entre les points extrêmes
        head_width = np.max(head_points_x) - np.min(head_points_x)
        
        # Estimer le rayon de la tête du rail
        # Pour une estimation simple, nous utilisons les points autour de la tête
        # et ajustons un cercle
        # Note: Cette estimation est très simplifiée et pourrait être améliorée
        head_radius = 300.0  # Valeur par défaut pour un rail UIC60
        
        # Calculer la hauteur du rail
        rail_height = np.max(y) - np.min(y)
        
        # Calculer la largeur du pied du rail
        foot_threshold = np.min(y) + 5.0  # 5 mm au-dessus du pied
        foot_points_indices = np.where(y < foot_threshold)[0]
        foot_points_x = x[foot_points_indices]
        
        # La largeur du pied est la distance entre les points extrêmes
        foot_width = np.max(foot_points_x) - np.min(foot_points_x)
        
        return {
            'head_position': (head_x, head_y),
            'head_width': head_width,
            'head_radius': head_radius,
            'rail_height': rail_height,
            'foot_width': foot_width
        }
    
    def get_profile(self, profile_name):
        """
        Récupère un profil chargé par son nom.
        
        Args:
            profile_name (str): Nom du profil à récupérer.
            
        Returns:
            dict: Dictionnaire contenant les données du profil.
            
        Raises:
            KeyError: Si le profil n'existe pas.
        """
        if profile_name not in self.profiles:
            raise KeyError(f"Le profil {profile_name} n'a pas été chargé.")
        
        return self.profiles[profile_name]
    
    def calculate_contact_geometry(self, wheel_profile_name, rail_profile_name, lateral_displacement=0.0):
        """
        Calcule la géométrie de contact entre un profil de roue et un profil de rail.
        
        Args:
            wheel_profile_name (str): Nom du profil de roue.
            rail_profile_name (str): Nom du profil de rail.
            lateral_displacement (float): Déplacement latéral de la roue par rapport au rail en mm.
            
        Returns:
            dict: Dictionnaire contenant les paramètres de la géométrie de contact:
                - 'contact_point_wheel': Coordonnées du point de contact sur la roue
                - 'contact_point_rail': Coordonnées du point de contact sur le rail
                - 'contact_angle': Angle de contact en radians
                - 'rolling_radius': Rayon de roulement effectif en mm
                - 'wheel_curvature': Courbure de la roue au point de contact en 1/mm
                - 'rail_curvature': Courbure du rail au point de contact en 1/mm
                
        Raises:
            KeyError: Si l'un des profils n'existe pas.
        """
        # Récupérer les profils
        wheel_profile = self.get_profile(wheel_profile_name)
        rail_profile = self.get_profile(rail_profile_name)
        
        # Vérifier les types de profil
        if wheel_profile['type'] != 'wheel':
            raise ValueError(f"Le profil {wheel_profile_name} n'est pas un profil de roue.")
        if rail_profile['type'] != 'rail':
            raise ValueError(f"Le profil {rail_profile_name} n'est pas un profil de rail.")
        
        # Extraire les coordonnées
        wheel_x = wheel_profile['x']
        wheel_y = wheel_profile['y']
        rail_x = rail_profile['x']
        rail_y = rail_profile['y']
        
        # Appliquer le déplacement latéral à la roue
        wheel_x_displaced = wheel_x + lateral_displacement
        
        # Trouver le point de contact
        # Pour chaque point de la roue, calculer la distance verticale au rail
        # Le point de contact est celui où cette distance est minimale
        
        # Interpoler le profil du rail pour obtenir les hauteurs aux positions x de la roue
        rail_y_interp = np.interp(wheel_x_displaced, rail_x, rail_y, left=np.nan, right=np.nan)
        
        # Calculer la distance verticale
        vertical_distance = wheel_y - rail_y_interp
        
        # Ignorer les points où l'interpolation a échoué
        valid_indices = ~np.isnan(vertical_distance)
        
        if not np.any(valid_indices):
            raise ValueError("Aucun point de contact valide n'a été trouvé.")
        
        # Trouver l'indice du point de contact
        contact_index = np.argmin(np.abs(vertical_distance[valid_indices]))
        contact_index = np.where(valid_indices)[0][contact_index]
        
        # Coordonnées du point de contact sur la roue
        contact_point_wheel_x = wheel_x[contact_index]
        contact_point_wheel_y = wheel_y[contact_index]
        
        # Coordonnées du point de contact sur le rail
        contact_point_rail_x = wheel_x_displaced[contact_index]
        contact_point_rail_y = rail_y_interp[contact_index]
        
        # Calculer l'angle de contact
        # Pour cela, nous calculons les pentes locales des profils
        
        # Pente locale de la roue
        if contact_index > 0 and contact_index < len(wheel_x) - 1:
            wheel_slope = (wheel_y[contact_index + 1] - wheel_y[contact_index - 1]) / (wheel_x[contact_index + 1] - wheel_x[contact_index - 1])
        else:
            # Au bord du profil, utiliser une différence finie à un côté
            if contact_index == 0:
                wheel_slope = (wheel_y[1] - wheel_y[0]) / (wheel_x[1] - wheel_x[0])
            else:
                wheel_slope = (wheel_y[-1] - wheel_y[-2]) / (wheel_x[-1] - wheel_x[-2])
        
        # Pente locale du rail
        # Trouver les indices des points du rail les plus proches du point de contact
        rail_distances = np.abs(rail_x - contact_point_rail_x)
        rail_nearest_indices = np.argsort(rail_distances)[:2]
        rail_nearest_indices.sort()
        
        if len(rail_nearest_indices) >= 2:
            rail_slope = (rail_y[rail_nearest_indices[1]] - rail_y[rail_nearest_indices[0]]) / (rail_x[rail_nearest_indices[1]] - rail_x[rail_nearest_indices[0]])
        else:
            # Si nous n'avons qu'un seul point, utiliser une valeur par défaut
            rail_slope = 0.0
        
        # Calculer l'angle de contact
        contact_angle = np.arctan(wheel_slope) - np.arctan(rail_slope)
        
        # Calculer le rayon de roulement effectif
        rolling_radius = wheel_profile['rolling_radius'] - contact_point_wheel_y
        
        # Calculer les courbures locales
        # Pour une estimation simple, nous utilisons les points voisins
        # et ajustons un cercle
        
        # Courbure de la roue
        wheel_curvature = self._estimate_curvature(wheel_x, wheel_y, contact_index)
        
        # Courbure du rail
        rail_index = rail_nearest_indices[0]
        rail_curvature = self._estimate_curvature(rail_x, rail_y, rail_index)
        
        return {
            'contact_point_wheel': (contact_point_wheel_x, contact_point_wheel_y),
            'contact_point_rail': (contact_point_rail_x, contact_point_rail_y),
            'contact_angle': contact_angle,
            'rolling_radius': rolling_radius,
            'wheel_curvature': wheel_curvature,
            'rail_curvature': rail_curvature
        }
    
    def _estimate_curvature(self, x, y, index, window_size=5):
        """
        Estime la courbure locale à un point donné.
        
        Args:
            x (array): Coordonnées x du profil.
            y (array): Coordonnées y du profil.
            index (int): Indice du point où estimer la courbure.
            window_size (int): Taille de la fenêtre pour l'estimation.
            
        Returns:
            float: Courbure estimée en 1/mm.
        """
        # Déterminer les indices des points à utiliser
        start_index = max(0, index - window_size // 2)
        end_index = min(len(x), index + window_size // 2 + 1)
        
        # Extraire les points
        x_local = x[start_index:end_index]
        y_local = y[start_index:end_index]
        
        # Si nous avons moins de 3 points, retourner une valeur par défaut
        if len(x_local) < 3:
            return 0.001  # 1 m de rayon de courbure
        
        # Calculer les différences finies pour estimer la courbure
        dx = np.diff(x_local)
        dy = np.diff(y_local)
        ds = np.sqrt(dx**2 + dy**2)
        
        # Éviter la division par zéro
        ds[ds < 1e-10] = 1e-10
        
        # Calculer les angles
        angles = np.arctan2(dy, dx)
        dtheta = np.diff(angles)
        
        # Normaliser les angles entre -pi et pi
        dtheta = np.mod(dtheta + np.pi, 2 * np.pi) - np.pi
        
        # Calculer la courbure
        curvature = dtheta / ds[:-1]
        
        # Retourner la courbure moyenne
        mean_curvature = np.mean(np.abs(curvature))
        
        # Si la courbure est trop faible, utiliser une valeur par défaut
        if mean_curvature < 1e-10:
            mean_curvature = 0.001  # 1 m de rayon de courbure
        
        return mean_curvature
