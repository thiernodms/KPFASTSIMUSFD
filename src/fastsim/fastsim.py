"""
Implémentation de l'algorithme FastSim de Kalker pour le contact tangentiel roue-rail.

Ce module implémente l'algorithme FastSim original développé par Kalker pour résoudre
efficacement les problèmes de contact tangentiel roue-rail.

Références:
    [1] Kalker, J. J. (1982). A fast algorithm for the simplified theory of rolling contact.
        Vehicle System Dynamics, 11(1), 1-13.
    [2] Kalker, J. J. (1990). Three-dimensional elastic bodies in rolling contact.
        Springer Science & Business Media.
"""

import numpy as np
from ..utils.geometry import discretize_contact_patch

class FastSim:
    """
    Implémentation de l'algorithme FastSim de Kalker pour le contact tangentiel roue-rail.
    
    Cet algorithme permet de calculer les forces de glissement (creep forces) et les contraintes
    de cisaillement dans la zone de contact entre une roue et un rail.
    """
    
    def __init__(self, poisson_ratio=0.3, shear_modulus=8.0e10, friction_coefficient=0.3, 
                 discretization=50):
        """
        Initialise l'algorithme FastSim avec les paramètres matériels.
        
        Args:
            poisson_ratio (float): Coefficient de Poisson du matériau. Par défaut 0.3.
            shear_modulus (float): Module de cisaillement en Pa. Par défaut 80 GPa.
            friction_coefficient (float): Coefficient de frottement. Par défaut 0.3.
            discretization (int): Nombre de points de discrétisation pour la zone de contact.
        """
        self.poisson_ratio = poisson_ratio
        self.shear_modulus = shear_modulus
        self.friction_coefficient = friction_coefficient
        self.discretization = discretization
        
    def calculate_flexibility_coefficients(self, a, b):
        """
        Calcule les coefficients de flexibilité selon la théorie de Kalker.
        
        Args:
            a (float): Demi-longueur de la zone de contact dans la direction longitudinale (m).
            b (float): Demi-longueur de la zone de contact dans la direction latérale (m).
            
        Returns:
            tuple: (L1, L2, L3) coefficients de flexibilité pour les directions x, y et spin.
        """
        # Rapport des demi-axes
        g = max(a/b, b/a)
        
        # Coefficients de Kalker (C11, C22, C23)
        # Ces coefficients sont tabulés dans la théorie de Kalker
        # Nous utilisons ici des approximations polynomiales
        
        # Pour C11 (coefficient longitudinal)
        if a >= b:
            C11 = 2.39 + 0.338 * g
        else:
            C11 = 2.39 + 0.338 / g
        
        # Pour C22 (coefficient latéral)
        if a >= b:
            C22 = 2.39 + 0.517 / g
        else:
            C22 = 2.39 + 0.517 * g
        
        # Pour C23 (coefficient de spin)
        if a >= b:
            C23 = 0.442 + 0.165 / g
        else:
            C23 = 0.442 + 0.165 * g
        
        # Calcul des coefficients de flexibilité
        L1 = C11 / (4 * self.shear_modulus * a)  # Longitudinal
        L2 = C22 / (4 * self.shear_modulus * b)  # Latéral
        L3 = C23 / (4 * self.shear_modulus * np.sqrt(a * b))  # Spin
        
        return L1, L2, L3
    
    def solve_tangential_problem(self, contact_patch, creepages, normal_force):
        """
        Résout le problème de contact tangentiel avec l'algorithme FastSim.
        
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
        
        # Calculer les coefficients de flexibilité
        L1, L2, L3 = self.calculate_flexibility_coefficients(a, b)
        
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
        
        # Pas de discrétisation pour la direction de roulement
        dx = 2 * a / self.discretization
        
        # Nombre de points adhérents (pour calculer la proportion de la zone d'adhérence)
        adhesion_points = 0
        total_points = np.sum(mask)
        
        # Parcourir la grille dans la direction de roulement (de l'arrière vers l'avant)
        for i in range(self.discretization - 1, -1, -1):
            for j in range(self.discretization):
                if mask[j, i]:
                    # Coordonnées du point courant
                    x = X[j, i]
                    y = Y[j, i]
                    
                    # Calcul des déplacements élastiques
                    # Selon la théorie de Kalker, les déplacements élastiques sont proportionnels
                    # aux contraintes de cisaillement
                    if i < self.discretization - 1 and mask[j, i+1]:
                        # Utiliser les valeurs du point précédent dans la direction de roulement
                        elastic_x = shear_stress_x[j, i+1] * L1
                        elastic_y = shear_stress_y[j, i+1] * L2
                    else:
                        # Pour le premier point, initialiser à zéro
                        elastic_x = 0.0
                        elastic_y = 0.0
                    
                    # Calcul des glissements rigides
                    rigid_slip_x = longitudinal_creepage - spin_creepage * y
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
        
        return {
            'tangential_forces': (Fx, Fy),
            'moment': Mz,
            'shear_stresses': (shear_stress_x[mask], shear_stress_y[mask]),
            'slip': (slip_x[mask], slip_y[mask]),
            'adhesion_area': adhesion_area
        }
