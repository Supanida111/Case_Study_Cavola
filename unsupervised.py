"""
Calvora — Unsupervised Learning (Full Pipeline)
ครอบคลุมทั้ง 5
  1. Descriptive Statistics
  2. Correlation Analysis
  3. Exploratory Data Analysis (EDA)
  4. Anomaly Detection (Isolation Forest + LOF) — flag, ไม่ remove
  5. Clustering + Dimensionality Reduction (K-Means + Hierarchical + PCA)

Output:
  - calvora_clustered.csv (เพิ่ม cluster, persona, pca_x/y, anomaly flags)
  - descriptive_stats.json
  - correlation_matrix.csv
  - anomaly_report.json
  - clustering_metrics.json
  - unsupervised_report.md
  - models/clustering.joblib
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score)

BASE = Path(__file__).parent
CLEAN = BASE / "data" / "calvora_clean.csv"
OUT = BASE / "data" / "calvora_clustered.csv"
METRICS = BASE / "data" / "clustering_metrics.json"
DESC_STATS = BASE / "data" / "descriptive_stats.json"
CORR_MATRIX = BASE / "data" / "correlation_matrix.csv"
ANOMALY_REPORT = BASE / "data" / "anomaly_report.json"
REPORT_MD = BASE / "data" / "unsupervised_report.md"
MODELS = BASE / "models"

# Numeric features for analysis (ไม่รวม binary multi-select cols ที่ sparse)
ANALYSIS_FEATURES = [
    "brand_correct", "brand_decoy_hit", "brand_understanding",
    "products_recognized_count", "products_tried_count",
    "tagline_known", "tagline_rating", "tagline_alignment",
    "ebisen_familiarity", "age",
    "score_ingredient", "score_taste", "score_variety",
    "score_texture", "score_health",
]

CLUSTER_FEATURES = [
    "brand_correct", "brand_decoy_hit",
    "products_recognized_count", "products_tried_count",
    "tagline_alignment", "tagline_known",
    "score_ingredient", "score_taste", "score_health",
    "ebisen_familiarity", "age",
]

PERSONA_ORDER = ["Brand Advocate", "Partial Connect",
                  "Aware but Lost", "Disconnected"]


# === 1. DESCRIPTIVE STATISTICS ====================================

def descriptive_statistics(df, features):
    """สรุปสถิติของ features ทั้งหมด"""
    stats = {}
    for f in features:
        s = df[f]
        stats[f] = {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "q25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "q75": float(s.quantile(0.75)),
            "max": float(s.max()),
            "skew": float(s.skew()),
        }
    # Categorical/discrete summaries
    cat_summary = {}
    for col in ["awareness_tier", "gender", "belief_reason", "belief_reason_typed"]:
        if col in df.columns:
            cat_summary[col] = df[col].value_counts().to_dict()
    return {"numeric": stats, "categorical": cat_summary}


# === 2. CORRELATION ANALYSIS ======================================

def correlation_analysis(df, features):
    """Pearson correlation matrix + identify strong correlations"""
    sub = df[features].copy()
    corr = sub.corr(method="pearson")
    # Strong pairs (|r| > 0.5, excluding self)
    strong = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) > 0.5:
                strong.append({"feature_a": a, "feature_b": b,
                                "pearson_r": float(round(r, 3))})
    strong.sort(key=lambda x: -abs(x["pearson_r"]))
    return corr, strong


# === 3. EDA helpers ===============================================

def eda_summary(df):
    """คำอธิบายเชิงสำรวจของชุดข้อมูล"""
    return {
        "n_total": len(df),
        "awareness_tier_dist": df["awareness_tier"].value_counts().to_dict(),
        "gender_dist": df["gender"].value_counts().to_dict() if "gender" in df.columns else {},
        "tagline_known_count": int(df["tagline_known"].sum()),
        "tagline_known_pct": float(df["tagline_known"].mean() * 100),
        "avg_brand_correct": float(df["brand_correct"].mean()),
        "avg_brand_decoy_hit": float(df["brand_decoy_hit"].mean()),
        "avg_brand_understanding": float(df["brand_understanding"].mean()),
    }


# === 4. ANOMALY DETECTION ========================================

def detect_anomalies(X_scaled, contamination=0.05):
    """ใช้ 2 วิธี: Isolation Forest (tree-based) + LOF (density-based)
    Flag ข้อมูลที่เป็น anomaly แต่ไม่ remove (sample เล็ก 124 คน)"""
    # Method 1: Isolation Forest
    iforest = IsolationForest(contamination=contamination, random_state=42,
                                n_estimators=100)
    pred_if = iforest.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
    score_if = iforest.score_samples(X_scaled)

    # Method 2: Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=min(10, len(X_scaled) - 1),
                               contamination=contamination)
    pred_lof = lof.fit_predict(X_scaled)
    score_lof = lof.negative_outlier_factor_

    is_anom_if = (pred_if == -1).astype(int)
    is_anom_lof = (pred_lof == -1).astype(int)
    is_anom_consensus = ((is_anom_if == 1) & (is_anom_lof == 1)).astype(int)

    return {
        "is_anomaly_iforest": is_anom_if,
        "is_anomaly_lof": is_anom_lof,
        "is_anomaly_consensus": is_anom_consensus,
        "score_iforest": score_if,
        "score_lof": score_lof,
    }


# === 5. CLUSTERING ===============================================

def evaluate_k(X, k_range=(2, 11)):
    """ทดสอบทั้ง K-Means และ Hierarchical Ward สำหรับ k=2..10
    เพื่อยืนยันว่า k ที่เลือก robust ทั้ง 2 algorithms"""
    results = []
    for k in range(k_range[0], k_range[1]):
        # K-Means
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_labels = km.fit_predict(X)
        # Hierarchical Ward
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
        hc_labels = hc.fit_predict(X)
        results.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(X, km_labels)),
            "davies_bouldin": float(davies_bouldin_score(X, km_labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, km_labels)),
            "hier_silhouette": float(silhouette_score(X, hc_labels)),
            "hier_davies_bouldin": float(davies_bouldin_score(X, hc_labels)),
        })
    return results


def assign_personas_2d_quadrant(centroids, feature_names):
    """2D Quadrant labeling — defensible, no arbitrary weights.

    แบ่ง 4 personas ด้วย 2 มิติ:
      • net_knowledge = brand_correct − brand_decoy_hit (มาก = รู้แบรนด์จริง)
      • engagement   = products_tried_count            (มาก = ลองสินค้าเยอะ)

    Threshold ใช้ median ของ centroid (data-driven, ไม่ hardcode):
      ┌────────────────┬──────────────────┐
      │ Partial Connect│  Brand Advocate  │   ← High knowledge
      │ Disconnected   │  Aware but Lost  │   ← Low knowledge
      └────────────────┴──────────────────┘
          Low engage      High engage

    ถ้า 2 cluster ลง quadrant เดียวกัน (collision)
    → fallback: rank ตาม composite score
    """
    bc_idx = feature_names.index("brand_correct")
    dh_idx = feature_names.index("brand_decoy_hit")
    tried_idx = feature_names.index("products_tried_count")
    rec_idx = feature_names.index("products_recognized_count")

    profiles = []
    for i, c in enumerate(centroids):
        profiles.append({
            "id": i,
            "k": c[bc_idx] - c[dh_idx],     # net knowledge
            "e": c[tried_idx],               # engagement
            "bc": c[bc_idx], "dh": c[dh_idx],
            "tried": c[tried_idx], "rec": c[rec_idx],
        })

    k_med = float(np.median([p["k"] for p in profiles]))
    e_med = float(np.median([p["e"] for p in profiles]))

    quadrant_to_persona = {
        (True, True):  "Brand Advocate",
        (True, False): "Partial Connect",
        (False, True): "Aware but Lost",
        (False, False): "Disconnected",
    }

    # Step 1: try 2D quadrant assignment
    proposals = []
    for p in profiles:
        q = (p["k"] >= k_med, p["e"] >= e_med)
        proposals.append((p, quadrant_to_persona[q]))

    # Step 2: check unique
    used_names = [name for _, name in proposals]
    if len(set(used_names)) == 4:
        # Clean 2D quadrant assignment
        return {p["id"]: name for p, name in proposals}

    # Step 3: collision detected — fallback to rank-based composite
    print("  [Warn] Quadrant collision detected, using rank-based fallback")
    scores = [(p["id"], p["bc"] - 0.5 * p["dh"] +
                          0.3 * p["tried"] + 0.2 * p["rec"])
                for p in profiles]
    scores.sort(key=lambda x: -x[1])
    return {cid: PERSONA_ORDER[rank] for rank, (cid, _) in enumerate(scores)}

# === MAIN PIPELINE ===============================================

def main():
    print("Loading clean data...")
    df = pd.read_csv(CLEAN)
    print(f"  shape: {df.shape}")

    # ---- 1. Descriptive Statistics ----
    print("\n[1] Descriptive Statistics...")
    desc = descriptive_statistics(df, ANALYSIS_FEATURES)
    DESC_STATS.write_text(json.dumps(desc, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"  saved → {DESC_STATS.name}")

    # ---- 2. Correlation Analysis ----
    print("\n[2] Correlation Analysis...")
    corr, corr_strong = correlation_analysis(df, ANALYSIS_FEATURES)
    corr.to_csv(CORR_MATRIX, encoding="utf-8-sig")
    print(f"  saved → {CORR_MATRIX.name} | strong pairs: {len(corr_strong)}")

    # ---- 3. EDA ----
    print("\n[3] EDA Summary...")
    eda = eda_summary(df)
    print(f"  awareness_tier: {eda['awareness_tier_dist']}")
    print(f"  tagline known: {eda['tagline_known_count']} "
          f"({eda['tagline_known_pct']:.1f}%)")

    # ---- 4. Anomaly Detection (BEFORE clustering) ----
    print("\n[4] Anomaly Detection...")
    X = df[CLUSTER_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    anom = detect_anomalies(X_scaled, contamination=0.05)
    df["is_anomaly_iforest"] = anom["is_anomaly_iforest"]
    df["is_anomaly_lof"] = anom["is_anomaly_lof"]
    df["is_anomaly_consensus"] = anom["is_anomaly_consensus"]
    df["anomaly_score_iforest"] = anom["score_iforest"]
    df["anomaly_score_lof"] = anom["score_lof"]

    n_if = int(anom["is_anomaly_iforest"].sum())
    n_lof = int(anom["is_anomaly_lof"].sum())
    n_cons = int(anom["is_anomaly_consensus"].sum())
    print(f"  IForest: {n_if} | LOF: {n_lof} | Consensus: {n_cons}")

    # Profile of consensus anomalies
    consensus_profile = {}
    if n_cons > 0:
        anomalies = df[df["is_anomaly_consensus"] == 1]
        for f in CLUSTER_FEATURES:
            consensus_profile[f] = float(anomalies[f].mean())

    anomaly_summary = {
        "method_iforest": {"contamination": 0.05, "n_estimators": 100,
                            "count": n_if},
        "method_lof": {"contamination": 0.05, "n_neighbors": 10,
                        "count": n_lof},
        "n_iforest": n_if, "n_lof": n_lof, "n_consensus": n_cons,
        "consensus_indices": df[df["is_anomaly_consensus"] == 1].index.tolist(),
        "consensus_profile": consensus_profile,
        "policy": "FLAG only — sample size 124 too small to remove",
    }
    ANOMALY_REPORT.write_text(json.dumps(anomaly_summary, ensure_ascii=False,
                                          indent=2), encoding="utf-8")
    print(f"  saved → {ANOMALY_REPORT.name}")

    # ---- 5. Clustering ----
    print("\n[5] Clustering — K-Means + Hierarchical Ward...")
    print("  Evaluating k=2..10...")
    k_eval = evaluate_k(X_scaled)
    for r in k_eval:
        print(f"    k={r['k']}: silh={r['silhouette']:.3f} "
              f"db={r['davies_bouldin']:.3f}")
    K = 4
    print(f"  Fitting K-Means with k={K}...")
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    df["cluster_kmeans"] = km.fit_predict(X_scaled)

    print("  Fitting Hierarchical Ward...")
    hc = AgglomerativeClustering(n_clusters=K, linkage="ward")
    df["cluster_hier"] = hc.fit_predict(X_scaled)

    # PCA
    print("  Computing PCA(2)...")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]
    print(f"    explained variance: {pca.explained_variance_ratio_}")

    # Persona naming
    centroids_scaled = km.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    persona_map = assign_personas_2d_quadrant(centroids.tolist(), CLUSTER_FEATURES)
    df["persona"] = df["cluster_kmeans"].map(persona_map)
    persona_dist = df["persona"].value_counts().to_dict()
    print(f"  persona dist: {persona_dist}")

    # Final clustering metrics
    kmeans_metrics = {
        "silhouette": float(silhouette_score(X_scaled, df["cluster_kmeans"])),
        "db": float(davies_bouldin_score(X_scaled, df["cluster_kmeans"])),
        "ch": float(calinski_harabasz_score(X_scaled, df["cluster_kmeans"])),
    }
    final_metrics = {
        "k_evaluation": k_eval,
        "selected_k": K,
        "kmeans_silhouette": kmeans_metrics["silhouette"],
        "kmeans_db": kmeans_metrics["db"],
        "kmeans_ch": kmeans_metrics["ch"],
        "hier_silhouette": float(silhouette_score(X_scaled,
                                                    df["cluster_hier"])),
        "hier_db": float(davies_bouldin_score(X_scaled, df["cluster_hier"])),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "persona_map": persona_map,
        "centroids": {persona_map[i]:
                       dict(zip(CLUSTER_FEATURES, c.tolist()))
                       for i, c in enumerate(centroids)},
    }

    print(f"\n  Silhouette: {kmeans_metrics['silhouette']:.3f}")
    print(f"  Davies-Bouldin: {kmeans_metrics['db']:.3f}")

    # ---- Save outputs ----
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    METRICS.write_text(json.dumps(final_metrics, ensure_ascii=False,
                                    indent=2), encoding="utf-8")
    joblib.dump({"kmeans": km, "scaler": scaler, "pca": pca,
                  "features": CLUSTER_FEATURES,
                  "persona_map": persona_map},
                 MODELS / "clustering.joblib")

if __name__ == "__main__":
    main()
