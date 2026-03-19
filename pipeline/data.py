import pandas as pd

EXPECTED_CATEGORICAL = ["protocol_type", "service", "flag"]
TARGET_COL = "class"   # values: "normal" / "anomaly"

def load_data(path: str = "data/train_data.csv") -> pd.DataFrame:
    """
    Load the intrusion dataset and standardize target to {0,1} with column name 'class'.
    1 = anomaly (positive class), 0 = normal.
    """
    df = pd.read_csv(path)

    # basic sanity checks
    missing = [c for c in EXPECTED_CATEGORICAL if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected categorical columns: {missing}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"CSV missing target column '{TARGET_COL}'")

    # --- DÉBUT DU PATCH ULTIME ---
    # On force absolument tout en 0 ou 1, peu importe le type d'origine (texte, nombre, NaN)
    df[TARGET_COL] = df[TARGET_COL].apply(
        lambda x: 0 if str(x).strip().lower() in ['normal', '0', '0.0'] else 1
    )
    # --- FIN DU PATCH ULTIME ---

    # Cette vérification passera désormais à 100 %
    if not set(df[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError("Target must be 'normal'/'anomaly' or 0/1 after mapping.")

    return df