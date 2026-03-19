from typing import Optional
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split # Ajout de cet import
import numpy as np

def _scale_pos_weight(y):
    # ratio of negatives to positives (for imbalance)
    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    return neg / pos

def train_model(X_train, y_train, preproc, seed: int = 42):
    """
    Build a Pipeline(preproc → XGBClassifier) and fit.
    Uses scale_pos_weight to handle class imbalance (anomaly = positive).
    Includes validation split and early stopping to prevent overfitting.
    """
    # 1. Création d'un jeu de validation interne (20%) pour l'early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        stratify=y_train, 
        random_state=seed
    )

    # 2. Préparation des données : le préprocesseur doit transformer le jeu de validation
    X_tr_prep = preproc.fit_transform(X_tr)
    X_val_prep = preproc.transform(X_val)

    # 3. Recalcul du poids sur le nouveau jeu d'entraînement interne
    spw = _scale_pos_weight(y_tr)

    # 4. Configuration optimisée du modèle XGBoost
    clf = XGBClassifier(
        n_estimators=1000,          # Augmenté : l'early stopping l'arrêtera au bon moment
        max_depth=5,                # Légèrement augmenté pour capter des signaux complexes
        learning_rate=0.05,         # Apprentissage plus lent et stable
        subsample=0.8,              # Ajout de régularisation pour éviter l'overfitting
        colsample_bytree=0.8,       # Ajout de régularisation
        random_state=seed,
        n_jobs=-1,
        eval_metric="aucpr",        # "aucpr" est bien meilleur que "logloss" pour le déséquilibre
        early_stopping_rounds=50,   # Arrête l'entraînement si pas d'amélioration après 50 arbres
        scale_pos_weight=spw,       # Gestion du déséquilibre
        verbosity=0
    )

    # 5. Entraînement en surveillant le jeu de validation pré-traité
    clf.fit(
        X_tr_prep, y_tr,
        eval_set=[(X_val_prep, y_val)],
        verbose=False
    )

    # 6. Reconstitution de la pipeline finale attendue par run.py
    pipe = Pipeline([("preproc", preproc), ("model", clf)])
    
    return pipe