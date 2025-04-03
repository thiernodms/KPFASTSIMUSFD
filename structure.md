# Structure d'implémentation des modèles KP-FastSim-USFD

## Vue d'ensemble

Cette implémentation modulaire des modèles KP-FastSim-USFD sera organisée en plusieurs composants distincts qui pourront être utilisés indépendamment ou combinés pour des simulations complètes de contact roue-rail et d'usure.

## Architecture du projet

```
kp_fastsim_usfd/
│
├── src/                        # Code source principal
│   ├── kp/                     # Modèle KP (Kik-Piotrowski)
│   │   ├── kp.py               # Implémentation du modèle KP de base
│   │   ├── mkp.py              # Implémentation du modèle KP modifié
│   │   └── utils.py            # Fonctions utilitaires pour les calculs KP
│   │
│   ├── fastsim/                # Algorithme FastSim
│   │   ├── fastsim.py          # Implémentation de l'algorithme FastSim
│   │   ├── fastrip.py          # Implémentation de l'algorithme FaStrip (version améliorée)
│   │   └── utils.py            # Fonctions utilitaires pour les calculs FastSim
│   │
│   ├── usfd/                   # Modèle d'usure USFD
│   │   ├── usfd.py             # Implémentation du modèle d'usure USFD
│   │   └── utils.py            # Fonctions utilitaires pour les calculs d'usure
│   │
│   ├── integration/            # Intégration des modèles
│   │   ├── kp_fastsim.py       # Intégration KP + FastSim
│   │   ├── mkp_fastsim.py      # Intégration MKP + FastSim
│   │   ├── mkp_fastrip.py      # Intégration MKP + FaStrip
│   │   └── full_model.py       # Modèle complet KP-FastSim-USFD
│   │
│   └── utils/                  # Utilitaires généraux
│       ├── geometry.py         # Fonctions géométriques
│       ├── math_utils.py       # Fonctions mathématiques
│       └── visualization.py    # Fonctions de visualisation
│
├── examples/                   # Exemples d'utilisation
│   ├── basic_contact.py        # Exemple de calcul de contact simple
│   ├── wear_prediction.py      # Exemple de prédiction d'usure
│   └── wheel_profile_opt.py    # Exemple d'optimisation de profil de roue
│
├── tests/                      # Tests unitaires et d'intégration
│   ├── test_kp.py              # Tests pour le modèle KP
│   ├── test_fastsim.py         # Tests pour FastSim
│   ├── test_usfd.py            # Tests pour le modèle USFD
│   └── test_integration.py     # Tests d'intégration
│
├── data/                       # Données pour les exemples et tests
│   ├── wheel_profiles/         # Profils de roues
│   ├── rail_profiles/          # Profils de rails
│   └── validation/             # Données de validation
│
├── docs/                       # Documentation
│   ├── api/                    # Documentation de l'API
│   ├── examples/               # Documentation des exemples
│   └── theory/                 # Documentation théorique
│
├── requirements.txt            # Dépendances Python
├── setup.py                    # Script d'installation
└── README.md                   # Documentation principale
```

## Modules principaux

### 1. Module KP (Kik-Piotrowski)

Ce module implémentera le modèle de contact non-hertzien KP pour déterminer la forme et la taille de la zone de contact ainsi que la distribution de pression.

**Fonctionnalités principales :**
- Calcul de la géométrie de contact non-elliptique
- Détermination de la distribution de pression
- Support pour les variantes KP, MKP (Modified KP) et EKP (Extended KP)

### 2. Module FastSim

Ce module implémentera l'algorithme FastSim de Kalker pour résoudre le problème de contact tangentiel.

**Fonctionnalités principales :**
- Discrétisation de la zone de contact
- Calcul des contraintes de cisaillement
- Calcul des forces de glissement (creep forces)
- Implémentation de FaStrip comme alternative améliorée

### 3. Module USFD

Ce module implémentera le modèle d'usure USFD basé sur la méthode T-gamma.

**Fonctionnalités principales :**
- Calcul de la puissance de frottement locale
- Implémentation de la fonction par morceaux pour les différents régimes d'usure
- Calcul du taux d'usure et du volume d'usure

### 4. Module d'intégration

Ce module combinera les modules précédents pour fournir une solution complète de simulation de contact roue-rail et d'usure.

**Fonctionnalités principales :**
- Intégration KP + FastSim
- Intégration MKP + FastSim
- Intégration MKP + FaStrip
- Modèle complet KP-FastSim-USFD

## Dépendances

L'implémentation utilisera les bibliothèques Python suivantes :
- NumPy pour les calculs numériques
- SciPy pour les fonctions scientifiques et l'optimisation
- Matplotlib pour la visualisation
- Pandas pour la gestion des données

## Interface utilisateur

Une interface simple en ligne de commande sera fournie pour exécuter les différentes simulations. Des exemples de scripts seront également fournis pour montrer comment utiliser les différents modules.

## Tests et validation

Des tests unitaires et d'intégration seront implémentés pour s'assurer que le code fonctionne correctement. Les résultats seront validés par rapport aux données publiées dans la littérature scientifique.

## Documentation

Une documentation complète sera fournie, comprenant :
- Documentation de l'API
- Exemples d'utilisation
- Explication théorique des modèles
- Instructions d'installation et d'utilisation
