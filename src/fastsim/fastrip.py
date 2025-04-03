"""
Implémentation de l'algorithme FaStrip pour le contact tangentiel roue-rail.

Ce module implémente l'algorithme FaStrip, une version améliorée de FastSim,
qui combine la théorie des bandes (Strip theory), la théorie linéaire de Kalker
et FastSim pour une meilleure précision.

Références:
    [1] Shen, Z. Y., & Li, Z. (1996). A fast non-steady state creep force model based on the
        simplified theory. Wear, 191(1-2), 242-244.
    [2] Vollebregt, E. A. H. (2014). Numerical modeling of measured railway creep versus
        creep-force curves with CONTACT. Wear, 314(1-2), 87-95.
    [3] Spiryagin, M., Polach, O., & Cole, C. (2013). Creep force modelling for rail traction
        vehicles based on the Fastsim algorithm. Vehicle System Dynamics, 51(11), 1765-1783.
"""

import numpy as np
from .fastsim import FastSim
from ..utils.geometry import discretize_contact_patch

class FaStrip(FastSim):
    """
    Implémentation de l'algorithme FaStrip pour le contact tangentiel roue-rail.
    
    FaStrip est une amélioration de FastSim qui combine la théorie des bandes,
    la théorie linéaire de Kalker et FastSim pour une meilleure précision,
    en particulier pour les contacts avec des rapports d'axes extrêmes.
    """
    
    def __init__(self, poisson_ratio=0.3, shear_modulus=8.0e10, friction_coefficient=0.3, 
                 discretization=50, num_strips=20):
        """
        Initialise l'algorithme FaStrip avec les paramètres matériels.
        
        Args:
            poisson_ratio (float): Coefficient de Poisson du matériau. Par défaut 0.3.
            shear_modulus (float): Module de cisaillement en Pa. Par défaut 80 GPa.
            friction_coefficient (float): Coefficient de frottement. Par défaut 0.3.
            discretization (int): Nombre de points de discrétisation pour la zone de contact.
            num_strips (int): Nombre de bandes pour la discrétisation dans la direction latérale.
        """
        super().__init__(poisson_ratio, shear_modulus, friction_coefficient, discretization)
        self.num_strips = num_strips
        
    def calculate_strip_flexibility_coefficients(self, a, b, strip_width):
        """
        Calcule les coefficients de flexibilité pour chaque bande.
        
        Args:
            a (float): Demi-longueur de la zone de contact dans la direction longitudinale (m).
            b (float): Demi-longueur de la zone de contact dans la direction latérale (m).
            strip_width (float): Largeur de la bande en mètres.
            
        Returns:
            tuple: (L1, L2, L3) coefficients de flexibilité pour les directions x, y et spin.
        """
        # Dans FaStrip, les coefficients de flexibilité sont calculés pour chaque bande
        # en utilisant la théorie des bandes
        
        # Rapport largeur/longueur de la bande
        strip_ratio = strip_width / (2 * a)
        
        # Coefficients de flexibilité selon la théorie des bandes
        # Ces formules sont basées sur les travaux de Shen & Li (1996)
        L1 = np.pi / (4 * self.shear_modulus * strip_width) * (1 + 1.15 * strip_ratio)
        L2 = np.pi / (4 * self.shear_modulus * strip_width) * (1 + 1.15 / strip_ratio)
        L3 = np.pi / (4 * self.shear_modulus * strip_width) * 0.5 * np.sqrt(strip_ratio)
        
        return L1, L2, L3
    
    def solve_tangential_problem(self, contact_patch, creepages, normal_force):
        """
        Résout le problème de contact tangentiel avec l'algorithme FaStrip.
        
        Args:
            contact_patch (dict): Dictionnaire contenant les paramètres de la zone de contact.
            creepages (dict): Dictionnaire contenant les glissements:
                - 'longitudinal': Glissement longitudinal (adimensionnel)
                - 'lateral': Glissement latéral (adimensionnel)
                - 'spin': Spin (1/m)
            normal_force (float): Force normale appliquée en Newtons.
            
        Returns:
            dict: Dictionnaire contenant les résultats du problème tangentiel:
                - 'tangential_forces': Forces tangentielles (Fx, Fy) en Newtons
                - 'moment': Moment de spin en N.m
                - 'shear_stresses': Contraintes de cisaillement aux points de contact
                - 'slip': Glissement local aux points de contact
                - 'adhesion_area': Proportion de la zone d'adhérence
        """
        # Extraire les paramètres de la zone de contact
        a = contact_patch['a']
        b = contact_patch['b']
        pressure_distribution = contact_patch['pressure_distribution']
        
        # Extraire les glissements
        longitudinal_creepage = creepages['longitudinal']
        lateral_creepage = creepages['lateral']
        spin_creepage = creepages['spin']
        
        # Discrétiser la zone de contact
        X, Y, mask = discretize_contact_patch(a, b, self.discretization)
        
        # Initialiser les tableaux pour les contraintes de cisaillement et le glissement
        shear_stress_x = np.zeros_like(X)
        shear_stress_y = np.zeros_like(Y)
        slip_x = np.zeros_like(X)
        slip_y = np.zeros_like(Y)
        
        # Pression de contact aux points de la grille
        pressure = np.zeros_like(X)
        pressure[mask] = pressure_distribution.reshape(X.shape)[mask]
        
        # Limite de traction (loi de Coulomb)
        traction_bound = self.friction_coefficient * pressure
        
        # Diviser la zone de contact en bandes dans la direction latérale
        strip_width = 2 * b / self.num_strips
        strip_edges = np.linspace(-b, b, self.num_strips + 1)
        
        # Pas de discrétisation pour la direction de roulement
        dx = 2 * a / self.discretization
        
        # Nombre de points adhérents (pour calculer la proportion de la zone d'adhérence)
        adhesion_points = 0
        total_points = np.sum(mask)
        
        # Traiter chaque bande séparément
        for k in range(self.num_strips):
            # Limites de la bande courante
            y_min = strip_edges[k]
            y_max = strip_edges[k + 1]
            
            # Masque pour les points dans la bande courante
            strip_mask = mask & (Y >= y_min) & (Y < y_max)
            
            # Si la bande est vide, passer à la suivante
            if not np.any(strip_mask):
                continue
            
            # Calculer les coefficients de flexibilité pour la bande courante
            L1, L2, L3 = self.calculate_strip_flexibility_coefficients(a, b, strip_width)
            
            # Position centrale de la bande
            y_center = (y_min + y_max) / 2
            
            # Parcourir la bande dans la direction de roulement (de l'arrière vers l'avant)
            for i in range(self.discretization - 1, -1, -1):
                for j in range(self.discretization):
                    if strip_mask[j, i]:
                        # Coordonnées du point courant
                        x = X[j, i]
                        y = Y[j, i]
                        
                        # Calcul des déplacements élastiques
                        if i < self.discretization - 1 and strip_mask[j, i+1]:
                            # Utiliser les valeurs du point précédent dans la direction de roulement
                            elastic_x = shear_stress_x[j, i+1] * L1
                            elastic_y = shear_stress_y[j, i+1] * L2
                        else:
                            # Pour le premier point, initialiser à zéro
                            elastic_x = 0.0
                            elastic_y = 0.0
                        
                        # Calcul des glissements rigides avec effet du spin
                        # Dans FaStrip, l'effet du spin est traité différemment
                        # Le spin est appliqué au centre de la bande
                        rigid_slip_x = longitudinal_creepage - spin_creepage * y_center
                        rigid_slip_y = lateral_creepage + spin_creepage * x
                        
                        # Calcul des glissements totaux
                        slip_x[j, i] = rigid_slip_x - elastic_x / dx
                        slip_y[j, i] = rigid_slip_y - elastic_y / dx
                        
                        # Calcul des contraintes de cisaillement
                        shear_stress_x[j, i] = slip_x[j, i] * self.shear_modulus * dx
                        shear_stress_y[j, i] = slip_y[j, i] * self.shear_modulus * dx
                        
                        # Norme de la contrainte de cisaillement
                        shear_norm = np.sqrt(shear_stress_x[j, i]**2 + shear_stress_y[j, i]**2)
                        
                        # Vérifier si le point est en adhérence ou en glissement
                        if shear_norm > traction_bound[j, i]:
                            # Point en glissement (appliquer la loi de Coulomb)
                            scale_factor = traction_bound[j, i] / shear_norm
                            shear_stress_x[j, i] *= scale_factor
                            shear_stress_y[j, i] *= scale_factor
                            
                            # Recalculer les glissements
                            slip_x[j, i] = shear_stress_x[j, i] / (self.shear_modulus * dx)
                            slip_y[j, i] = shear_stress_y[j, i] / (self.shear_modulus * dx)
                        else:
                            # Point en adhérence
                            adhesion_points += 1
        
        # Calculer les forces tangentielles et le moment
        # Intégration des contraintes de cisaillement sur la zone de contact
        dA = dx * dx  # Élément de surface
        Fx = np.sum(shear_stress_x[mask]) * dA
        Fy = np.sum(shear_stress_y[mask]) * dA
        Mz = np.sum((X[mask] * shear_stress_y[mask] - Y[mask] * shear_stress_x[mask])) * dA
        
        # Proportion de la zone d'adhérence
        adhesion_area = adhesion_points / total_points if total_points > 0 else 0.0
        
        # Appliquer des facteurs de correction basés sur la théorie linéaire de Kalker
        # Ces facteurs améliorent la précision pour les faibles glissements
        # Basé sur les travaux de Vollebregt (2014)
        
        # Calculer les coefficients de Kalker pour la zone de contact complète
        L1_full, L2_full, L3_full = self.calculate_flexibility_coefficients(a, b)
        
        # Calculer les forces tangentielles selon la théorie linéaire de Kalker
        Fx_linear = -self.shear_modulus * a * b * longitudinal_creepage / L1_full
        Fy_linear = -self.shear_modulus * a * b * lateral_creepage / L2_full
        
        # Facteurs de correction (rapport entre la théorie linéaire et FaStrip)
        # Ces facteurs sont appliqués uniquement pour les faibles glissements
        # où la théorie linéaire est valide
        if abs(longitudinal_creepage) < 1e-4 and abs(lateral_creepage) < 1e-4:
            correction_x = Fx_linear / Fx if abs(Fx) > 1e-10 else 1.0
            correction_y = Fy_linear / Fy if abs(Fy) > 1e-10 else 1.0
            
            # Limiter les facteurs de correction
            correction_x = np.clip(correction_x, 0.8, 1.2)
            correction_y = np.clip(correction_y, 0.8, 1.2)
            
            # Appliquer les corrections
            Fx *= correction_x
            Fy *= correction_y
        
        return {
            'tangential_forces': (Fx, Fy),
            'moment': Mz,
            'shear_stresses': (shear_stress_x[mask], shear_stress_y[mask]),
            'slip': (slip_x[mask], slip_y[mask]),
            'adhesion_area': adhesion_area
        }
