import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report

FEATURE_NAMES = [
    "amount",
    "tx_count_last_10m",
    "tx_count_last_1h",
    "amount_rolling_avg_1h",
    "amount_deviation_from_avg",
    "geo_distance_velocity_kmh",
    "is_night_transaction",
    "category_risk_index",
]


def generate_synthetic_fraud_dataset(n_samples: int = 50000, fraud_rate: float = 0.008, random_state: int = 42):
    """
    Generates a realistic, highly imbalanced synthetic financial transaction dataset
    with embedded fraud attack vectors (bursts, amount spikes, impossible travel).
    """
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # 1. Legitimate Transactions (99.2%)
    legit_amount = np.random.lognormal(mean=3.5, sigma=0.8, size=n_legit)  # ~$15 to $150
    legit_tx_10m = np.random.choice([1, 2, 3], p=[0.85, 0.12, 0.03], size=n_legit)
    legit_tx_1h = legit_tx_10m + np.random.choice([0, 1, 2], p=[0.7, 0.25, 0.05], size=n_legit)
    legit_rolling_avg = legit_amount * np.random.uniform(0.85, 1.15, size=n_legit)
    legit_z_score = np.random.normal(loc=0.0, scale=0.8, size=n_legit)
    legit_velocity = np.random.exponential(scale=12.0, size=n_legit)
    legit_velocity = np.clip(legit_velocity, 0.0, 75.0)  # driving/walking speeds
    legit_night = np.random.choice([0, 1], p=[0.92, 0.08], size=n_legit)
    legit_cat_risk = np.random.choice([0.05, 0.08, 0.15, 0.25, 0.30], size=n_legit)
    legit_labels = np.zeros(n_legit, dtype=int)

    # 2. Fraudulent Transactions (0.8%)
    # Injected attack vectors:
    fraud_types = np.random.choice(["burst", "amount_spike", "impossible_travel", "combined"], size=n_fraud)
    
    fraud_amount = np.zeros(n_fraud)
    fraud_tx_10m = np.zeros(n_fraud, dtype=int)
    fraud_tx_1h = np.zeros(n_fraud, dtype=int)
    fraud_rolling_avg = np.zeros(n_fraud)
    fraud_z_score = np.zeros(n_fraud)
    fraud_velocity = np.zeros(n_fraud)
    fraud_night = np.zeros(n_fraud, dtype=int)
    fraud_cat_risk = np.zeros(n_fraud)
    fraud_labels = np.ones(n_fraud, dtype=int)

    for i in range(n_fraud):
        ftype = fraud_types[i]
        baseline_amt = np.random.uniform(30.0, 120.0)
        fraud_rolling_avg[i] = baseline_amt

        if ftype == "burst":
            fraud_amount[i] = np.random.uniform(80.0, 450.0)
            fraud_tx_10m[i] = np.random.randint(6, 20)
            fraud_tx_1h[i] = fraud_tx_10m[i] + np.random.randint(5, 15)
            fraud_z_score[i] = np.random.uniform(1.5, 4.0)
            fraud_velocity[i] = np.random.uniform(5.0, 60.0)
            fraud_night[i] = np.random.choice([0, 1], p=[0.6, 0.4])
            fraud_cat_risk[i] = np.random.choice([0.60, 0.75, 0.85])

        elif ftype == "amount_spike":
            fraud_amount[i] = np.random.uniform(1800.0, 9500.0)
            fraud_tx_10m[i] = np.random.choice([1, 2])
            fraud_tx_1h[i] = np.random.choice([2, 3, 4])
            fraud_z_score[i] = np.random.uniform(4.5, 14.0)
            fraud_velocity[i] = np.random.uniform(0.0, 40.0)
            fraud_night[i] = np.random.choice([0, 1], p=[0.5, 0.5])
            fraud_cat_risk[i] = np.random.choice([0.70, 0.85, 0.95])

        elif ftype == "impossible_travel":
            fraud_amount[i] = np.random.uniform(120.0, 1200.0)
            fraud_tx_10m[i] = np.random.choice([2, 3])
            fraud_tx_1h[i] = np.random.choice([3, 4, 5])
            fraud_z_score[i] = np.random.uniform(0.8, 3.0)
            fraud_velocity[i] = np.random.uniform(750.0, 3500.0)  # Impossible speed
            fraud_night[i] = np.random.choice([0, 1], p=[0.7, 0.3])
            fraud_cat_risk[i] = np.random.choice([0.30, 0.60, 0.85])

        else:  # combined severe attack
            fraud_amount[i] = np.random.uniform(1500.0, 6000.0)
            fraud_tx_10m[i] = np.random.randint(5, 18)
            fraud_tx_1h[i] = fraud_tx_10m[i] + np.random.randint(5, 15)
            fraud_z_score[i] = np.random.uniform(4.0, 10.0)
            fraud_velocity[i] = np.random.uniform(850.0, 2200.0)
            fraud_night[i] = 1
            fraud_cat_risk[i] = 0.95

    # Combine into DataFrame
    df_legit = pd.DataFrame({
        "amount": legit_amount,
        "tx_count_last_10m": legit_tx_10m,
        "tx_count_last_1h": legit_tx_1h,
        "amount_rolling_avg_1h": legit_rolling_avg,
        "amount_deviation_from_avg": legit_z_score,
        "geo_distance_velocity_kmh": legit_velocity,
        "is_night_transaction": legit_night,
        "category_risk_index": legit_cat_risk,
        "label": legit_labels,
    })

    df_fraud = pd.DataFrame({
        "amount": fraud_amount,
        "tx_count_last_10m": fraud_tx_10m,
        "tx_count_last_1h": fraud_tx_1h,
        "amount_rolling_avg_1h": fraud_rolling_avg,
        "amount_deviation_from_avg": fraud_z_score,
        "geo_distance_velocity_kmh": fraud_velocity,
        "is_night_transaction": fraud_night,
        "category_risk_index": fraud_cat_risk,
        "label": fraud_labels,
    })

    df = pd.concat([df_legit, df_fraud], ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def train_model(save_dir: str = "src/inference/model"):
    """Trains an XGBoost model using cost-sensitive learning for PR-AUC optimization."""
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "fraud_xgboost_model.json")
    meta_path = os.path.join(save_dir, "model_meta.json")

    print("[*] Generating synthetic high-velocity training dataset (50,000 transactions)...")
    df = generate_synthetic_fraud_dataset(n_samples=50000, fraud_rate=0.01)

    X = df[FEATURE_NAMES]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cost-sensitive weighting: scale_pos_weight = negative_samples / positive_samples
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    print(f"[*] Class imbalance: {neg_count} legit vs {pos_count} fraud (scale_pos_weight = {scale_pos_weight:.2f})")

    print("[*] Training XGBoost model with PR-AUC objective...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluation
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.70).astype(int)

    pr_auc = average_precision_score(y_val, val_probs)
    roc_auc = roc_auc_score(y_val, val_probs)

    print(f"[+] Model Evaluation Results:")
    print(f"    - PR-AUC:  {pr_auc:.4f}")
    print(f"    - ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report (threshold = 0.70):")
    print(classification_report(y_val, val_preds, digits=4))

    # Save model
    model.save_model(model_path)
    print(f"[+] Model successfully saved to: {model_path}")

    # Save feature importances and metadata
    feature_importances = dict(zip(FEATURE_NAMES, [float(x) for x in model.feature_importances_]))
    meta = {
        "feature_names": FEATURE_NAMES,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "scale_pos_weight": float(scale_pos_weight),
        "thresholds": {
            "legitimate_below": 0.40,
            "review_required_between": [0.40, 0.75],
            "fraud_suspected_above": 0.75
        },
        "feature_importances": feature_importances
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[+] Model metadata written to: {meta_path}")

    return model, meta


if __name__ == "__main__":
    train_model()
