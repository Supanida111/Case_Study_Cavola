"""
Calvora — SQLite Database Integration
Loads clean & clustered CSVs into SQLite tables for the dashboard.
"""
import json
import sqlite3
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
DB = BASE / "data" / "calvora.db"
CLEAN = BASE / "data" / "calvora_clean.csv"
CLUSTERED = BASE / "data" / "calvora_clustered.csv"
CLUSTER_METRICS = BASE / "data" / "clustering_metrics.json"
SUPERVISED_METRICS = BASE / "data" / "supervised_metrics.json"
CORR_MATRIX = BASE / "data" / "correlation_matrix.csv"
ANOMALY_REPORT = BASE / "data" / "anomaly_report.json"


def init_db():
    conn = sqlite3.connect(DB)
    print(f"Connected to {DB}")

    # Survey responses (aware + clustered)
    df = pd.read_csv(CLUSTERED)
    df.to_sql("survey_responses", conn, if_exists="replace", index=False)
    print(f"  survey_responses: {len(df)} rows")

    # Drop legacy nonaware table if it exists (cleanup from previous build)
    conn.execute("DROP TABLE IF EXISTS nonaware_respondents")

    # Model metrics tables
    with open(CLUSTER_METRICS, "r", encoding="utf-8") as f:
        cm = json.load(f)
    cluster_eval = pd.DataFrame(cm["k_evaluation"])
    cluster_eval.to_sql("cluster_k_evaluation", conn,
                         if_exists="replace", index=False)
    print(f"  cluster_k_evaluation: {len(cluster_eval)} rows")

    with open(SUPERVISED_METRICS, "r", encoding="utf-8") as f:
        sm = json.load(f)
    rows = []
    for name, r in sm["results"].items():
        rows.append({
            "model": name,
            "cv_f1_mean": r["cv_f1_mean"],
            "cv_f1_std": r["cv_f1_std"],
            "test_accuracy": r["test_accuracy"],
            "test_f1": r["test_f1"],
            "is_best": int(name == sm["best_model"]),
        })
    sup_df = pd.DataFrame(rows)
    sup_df.to_sql("supervised_metrics", conn, if_exists="replace", index=False)
    print(f"  supervised_metrics: {len(sup_df)} rows")

    # Feature importance
    fi = sm["feature_importance_rf"]
    fi_df = pd.DataFrame([{"feature": k, "importance": v} for k, v in fi.items()])
    fi_df.to_sql("feature_importance", conn, if_exists="replace", index=False)
    print(f"  feature_importance: {len(fi_df)} rows")

    # Correlation matrix (long format for easy querying)
    corr = pd.read_csv(CORR_MATRIX, index_col=0)
    corr_long = corr.stack().reset_index()
    corr_long.columns = ["feature_a", "feature_b", "pearson_r"]
    corr_long.to_sql("correlation_matrix", conn,
                       if_exists="replace", index=False)
    print(f"  correlation_matrix: {len(corr_long)} rows")

    # Anomaly summary
    with open(ANOMALY_REPORT, "r", encoding="utf-8") as f:
        ar = json.load(f)
    anom_rows = [
        {"method": "Isolation Forest", "count": ar["n_iforest"]},
        {"method": "Local Outlier Factor", "count": ar["n_lof"]},
        {"method": "Consensus (both)", "count": ar["n_consensus"]},
    ]
    pd.DataFrame(anom_rows).to_sql("anomaly_summary", conn,
                                     if_exists="replace", index=False)
    print(f"  anomaly_summary: {len(anom_rows)} rows")

    conn.commit()
    conn.close()
    print("DB initialized.")


def query(sql, params=None):
    """Helper for app pages to query the DB."""
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


if __name__ == "__main__":
    init_db()
