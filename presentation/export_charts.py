"""
Export all charts as static PNG (Thai font) for embedding in PPTX/DOCX.
Uses matplotlib with Tahoma to support Thai characters.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# Thai font support — set fallback chain explicitly
import matplotlib.font_manager as fm
THAI_FONTS = ["Tahoma", "Leelawadee UI", "Cordia New", "Sarabun",
              "Microsoft Sans Serif", "Segoe UI", "Arial"]
available = {f.name for f in fm.fontManager.ttflist}
chosen = next((f for f in THAI_FONTS if f in available), "DejaVu Sans")
rcParams["font.family"] = chosen
rcParams["font.sans-serif"] = THAI_FONTS + ["DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
print(f"Using Thai font: {chosen}")
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.0)
rcParams["font.family"] = chosen  # re-apply after seaborn

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "charts"
OUT.mkdir(exist_ok=True, parents=True)

PERSONA_TH = {
    "Brand Advocate": "ผู้สนับสนุนแบรนด์",
    "Partial Connect": "เชื่อมโยงบางส่วน",
    "Aware but Lost": "รู้จักแต่สับสน",
    "Disconnected": "ไม่เชื่อมโยง",
}
PERSONA_COLORS = {
    "ผู้สนับสนุนแบรนด์": "#28a745",
    "เชื่อมโยงบางส่วน": "#17a2b8",
    "รู้จักแต่สับสน": "#ffc107",
    "ไม่เชื่อมโยง": "#dc3545",
}


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path.name}")


def chart_understanding_histogram():
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(0, 13) - 0.5
    ax.hist(df["brand_understanding"], bins=bins,
             color="#a8d5f0", edgecolor="#5dade2", linewidth=1)
    ax.set_title("การกระจายตัวของคะแนนความเข้าใจในแบรนด์ (เต็ม 11 คะแนน)",
                  fontsize=14)
    ax.set_xlabel("คะแนนที่ได้รับ")
    ax.set_ylabel("จำนวนคน (คน)")
    ax.set_xticks(range(0, 12))
    save(fig, "01_understanding_histogram.png")


def chart_awareness_tier():
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    counts = df["awareness_tier"].value_counts()
    label_map = {"Low": "ระดับต่ำ", "Mid": "ระดับกลาง", "High": "ระดับสูง"}
    order = ["Low", "Mid", "High"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#dc3545", "#ffc107", "#28a745"]
    bars = ax.bar([label_map[t] for t in order],
                   [counts.get(t, 0) for t in order], color=colors)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                 f"{int(b.get_height())}", ha="center")
    ax.set_title("กลุ่มระดับความเข้าใจแบรนด์ Calvora", fontsize=14)
    ax.set_ylabel("จำนวนคน")
    save(fig, "02_awareness_tier.png")


def chart_subbrand_recognition():
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    counts = {}
    for col in df.columns:
        if col.startswith("recog_"):
            counts[col.replace("recog_", "")] = int(df[col].sum())
    cdf = pd.DataFrame([{"brand": k, "n": v} for k, v in counts.items()])
    cdf = cdf.sort_values("n")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(cdf["brand"], cdf["n"], color=plt.cm.Oranges(
        np.linspace(0.4, 0.9, len(cdf))))
    for b, v in zip(bars, cdf["n"]):
        ax.text(v + 1, b.get_y() + b.get_height() / 2,
                 str(int(v)), va="center")
    ax.set_title("การรู้จักแบรนด์ลูกในเครือ Calvora (จากผู้ตอบ 124 คน)",
                  fontsize=13)
    ax.set_xlabel("จำนวนคน")
    save(fig, "03_subbrand_recognition.png")


def chart_decoy_confusion():
    df = pd.read_csv(DATA / "decoy_per_brand.csv")
    df = df.sort_values("wrong_count", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#fadbd8", "#e89e95", "#c0392b"]
    bars = ax.barh(df["brand"], df["wrong_count"], color=colors)
    for b, v in zip(bars, df["wrong_count"]):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2,
                 str(int(v)), va="center")
    ax.set_title("แบรนด์คู่แข่งที่คนสับสนว่าเป็น Calvora มากที่สุด",
                  fontsize=13)
    ax.set_xlabel("จำนวนคนที่ตอบผิด (คน)")
    ax.set_ylabel("ชื่อแบรนด์คู่แข่ง")
    save(fig, "04_decoy_confusion.png")


def chart_k_evaluation():
    with open(DATA / "clustering_metrics.json", encoding="utf-8") as f:
        m = json.load(f)
    keval = pd.DataFrame(m["k_evaluation"])
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = ax1.twinx()
    l1, = ax1.plot(keval["k"], keval["silhouette"], "o-",
                    color="#17a2b8", label="Silhouette (สูง = ดี)")
    l2, = ax2.plot(keval["k"], keval["davies_bouldin"], "s--",
                    color="#dc3545", label="Davies-Bouldin (ต่ำ = ดี)")
    ax1.axvline(4, ls=":", color="black", alpha=0.5)
    ax1.text(4.05, ax1.get_ylim()[1] * 0.95, "เลือก k=4",
              fontsize=10, color="black")
    ax1.set_xlabel("จำนวนกลุ่ม (k)")
    ax1.set_ylabel("Silhouette", color="#17a2b8")
    ax2.set_ylabel("Davies-Bouldin", color="#dc3545")
    ax1.set_title("การเลือกจำนวนกลุ่ม (k = 2..10)", fontsize=14)
    ax1.legend(handles=[l1, l2], loc="upper right")
    save(fig, "05_k_evaluation.png")


def chart_pca_scatter():
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    df["persona_th"] = df["persona"].map(PERSONA_TH)
    fig, ax = plt.subplots(figsize=(9, 6))
    for persona, grp in df.groupby("persona_th"):
        ax.scatter(grp["pca_x"], grp["pca_y"],
                    c=PERSONA_COLORS.get(persona, "#888"),
                    label=persona, s=70, alpha=0.7,
                    edgecolors="white", linewidth=1)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("การกระจายตัวของลูกค้าด้วย PCA (2 มิติ) — แยกตามกลุ่ม",
                  fontsize=14)
    ax.legend(loc="best")
    save(fig, "06_pca_scatter.png")


def chart_persona_radar():
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    factors = ["score_ingredient", "score_taste", "score_variety",
               "score_texture", "score_health"]
    labels = ["วัตถุดิบคุณภาพ", "รสชาติอร่อย", "รสชาติหลากหลาย",
              "สัมผัสกรุบกรอบ", "สุขภาพดี"]
    angles = np.linspace(0, 2 * np.pi, len(factors), endpoint=False).tolist()
    angles += [angles[0]]
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
    for persona, grp in df.groupby("persona"):
        vals = [grp[f].mean() for f in factors]
        vals += [vals[0]]
        persona_th = PERSONA_TH.get(persona, persona)
        ax.plot(angles, vals, "o-",
                 color=PERSONA_COLORS.get(persona_th, "#888"),
                 label=persona_th, linewidth=2)
        ax.fill(angles, vals,
                 color=PERSONA_COLORS.get(persona_th, "#888"), alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(1, 5)
    ax.set_title("ความสำคัญของปัจจัยการซื้อ แยกตามกลุ่มลูกค้า",
                  fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    save(fig, "07_persona_radar.png")


def chart_model_comparison():
    with open(DATA / "supervised_metrics.json", encoding="utf-8") as f:
        m = json.load(f)
    rows = []
    for name, r in m["results"].items():
        rows.append({
            "model": name,
            "cv_f1": r["cv_f1_mean"], "cv_f1_std": r["cv_f1_std"],
            "test_acc": r["test_accuracy"], "test_f1": r["test_f1"],
        })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    w = 0.27
    ax.bar(x - w, df["cv_f1"], w, yerr=df["cv_f1_std"],
            label="CV F1 (เฉลี่ย ± std)", color="#17a2b8", capsize=5)
    ax.bar(x, df["test_acc"], w, label="Test Accuracy", color="#28a745")
    ax.bar(x + w, df["test_f1"], w, label="Test F1", color="#ffc107")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    ax.set_ylabel("คะแนน")
    ax.set_title("เปรียบเทียบประสิทธิภาพโมเดล (5 โมเดล)", fontsize=14)
    ax.legend()
    save(fig, "08_model_comparison.png")


def chart_confusion_matrix():
    with open(DATA / "supervised_metrics.json", encoding="utf-8") as f:
        m = json.load(f)
    best = m["best_model"]
    cm = np.array(m["results"][best]["confusion_matrix"])
    tiers = ["ต่ำ", "กลาง", "สูง"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                 xticklabels=tiers, yticklabels=tiers, ax=ax,
                 cbar_kws={"label": "จำนวน"})
    ax.set_xlabel("ทำนาย")
    ax.set_ylabel("จริง")
    ax.set_title(f"Confusion Matrix — {best}", fontsize=14)
    save(fig, "09_confusion_matrix.png")


def chart_feature_importance():
    with open(DATA / "supervised_metrics.json", encoding="utf-8") as f:
        m = json.load(f)
    fi = m["feature_importance_rf"]
    feature_th = {
        "tagline_known": "รู้จัก Tagline",
        "tagline_alignment": "ความเข้าใจ Tagline",
        "tagline_rating": "ให้คะแนน Tagline",
        "products_recognized_count": "จำนวนแบรนด์ที่รู้จัก",
        "products_tried_count": "จำนวนแบรนด์ที่เคยทาน",
        "score_ingredient": "ปัจจัย: วัตถุดิบ",
        "score_taste": "ปัจจัย: รสชาติ",
        "score_variety": "ปัจจัย: ความหลากหลาย",
        "score_texture": "ปัจจัย: สัมผัส",
        "score_health": "ปัจจัย: สุขภาพ",
        "ebisen_familiarity": "ความคุ้นเคย Ebisen",
        "age": "ช่วงอายุ",
        "gender_ชาย": "เพศ: ชาย",
        "gender_หญิง": "เพศ: หญิง",
        "gender_LGBTQ+": "เพศ: LGBTQ+",
    }
    items = sorted(fi.items(), key=lambda x: x[1])
    feats = [feature_th.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    bars = ax.barh(feats, vals, color=plt.cm.viridis(np.linspace(0.2, 0.9, len(feats))))
    for b, v in zip(bars, vals):
        ax.text(v + 0.002, b.get_y() + b.get_height() / 2,
                 f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("ความสำคัญ (Importance)")
    ax.set_title("ความสำคัญของแต่ละ Feature (Random Forest)", fontsize=14)
    save(fig, "10_feature_importance.png")


def chart_trust_factors():
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    cols = [c for c in df.columns
            if c.startswith("trust_") and c != "trust_signal"]
    counts = {c.replace("trust_", ""): int(df[c].sum()) for c in cols}
    cdf = pd.DataFrame([{"reason": k, "n": v}
                          for k, v in counts.items() if v > 0])
    cdf = cdf.sort_values("n")
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cdf["reason"], cdf["n"],
                    color=plt.cm.Greens(np.linspace(0.4, 0.85, len(cdf))))
    for b, v in zip(bars, cdf["n"]):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2,
                 str(int(v)), va="center")
    ax.set_title("ทำไมถึงเชื่อมั่นในวัตถุดิบจากธรรมชาติ?", fontsize=14)
    ax.set_xlabel("จำนวนคน (People)")
    ax.set_ylabel("เหตุผลที่เลือก")
    save(fig, "11_trust_factors.png")


def chart_ebisen_flavors():
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    cols = [c for c in df.columns if c.startswith("report_flavor_")]
    counts = {c.replace("report_flavor_", ""): int(df[c].sum())
              for c in cols}
    cdf = pd.DataFrame([{"flavor": k, "n": v} for k, v in counts.items()])
    cdf = cdf.sort_values("n")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.barh(cdf["flavor"], cdf["n"],
                    color=plt.cm.viridis(np.linspace(0.2, 0.85, len(cdf))))
    for b, v in zip(bars, cdf["n"]):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2,
                 str(int(v)), va="center")
    ax.set_title("รสชาติ/สินค้า Ebisen ที่เคยรับประทาน", fontsize=14)
    ax.set_xlabel("จำนวนคน (People)")
    ax.set_ylabel("รายการรสค้า")
    save(fig, "12_ebisen_flavors.png")


def chart_attribution_gap():
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    n = len(df)
    counts = {}
    for col in df.columns:
        if col.startswith("recog_"):
            counts[col.replace("recog_", "")] = int(df[col].sum())
    cdf = pd.DataFrame([{"brand": k, "recognized": 100 * v / n,
                          "gap": 100 * (n - v) / n}
                         for k, v in counts.items()])
    cdf = cdf.sort_values("recognized")
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.barh(cdf["brand"], cdf["recognized"],
             color="#28a745", label="รู้จัก (%)")
    ax.barh(cdf["brand"], cdf["gap"], left=cdf["recognized"],
             color="#dc3545", label="ช่องว่าง (%)")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% ของผู้ตอบ (124 คน)")
    ax.set_title("ช่องว่างการรู้จักแบรนด์ลูก (Brand Recognition Gap)",
                  fontsize=14)
    ax.legend(loc="lower right")
    save(fig, "13_attribution_gap.png")


def chart_pipeline_diagram():
    """Custom matplotlib diagram of the ML pipeline."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    boxes = [
        (0.5, 5, 2.4, 1.2, "snack.xlsx\n(141 × 46)", "#fff3cd"),
        (3.4, 5, 2.6, 1.2, "data_prep.py\n(NLP + Encoding)", "#cfe2ff"),
        (6.4, 6, 2.8, 1, "calvora_clean.csv\n(124 × 65)", "#d1e7dd"),
        (6.4, 4, 2.8, 1, "calvora_nonaware.csv\n(17 rows)", "#f8d7da"),
        (3.4, 2.5, 2.6, 1.2, "clustering.py\n(K-Means + Hierarchical)",
         "#cfe2ff"),
        (6.4, 2.5, 2.8, 1.2, "calvora_clustered.csv\n(+ persona, PCA)",
         "#d1e7dd"),
        (3.4, 0.3, 2.6, 1.2, "supervised.py\n(5 Models)", "#cfe2ff"),
        (6.4, 0.3, 2.8, 1.2, "best model\n(supervised_best.joblib)", "#d1e7dd"),
        (9.7, 3, 2.8, 1.2, "db.py → SQLite\n(calvora.db)", "#fff3cd"),
        (9.7, 0.8, 2.8, 1.2, "app.py (Dash)\n4 หน้า + Thai UI", "#e2d6f9"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color,
                                    edgecolor="#333", linewidth=1.3))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                 fontsize=10)

    arrows = [
        ((2.9, 5.6), (3.4, 5.6)),
        ((6.0, 5.7), (6.4, 6.5)),
        ((6.0, 5.5), (6.4, 4.5)),
        ((7.8, 6.0), (5.5, 3.7)),  # clean → clustering
        ((6.0, 3.1), (6.4, 3.1)),  # clustering → clustered
        ((7.8, 2.5), (5.5, 1.5)),  # clustered → supervised
        ((6.0, 0.9), (6.4, 0.9)),  # supervised → best model
        ((9.2, 3.1), (9.7, 3.5)),  # clustered → db
        ((11.1, 3.0), (11.1, 2.0)),  # db → app
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    ax.set_title("ภาพรวมระบบ — Calvora AI Marketing Pipeline",
                  fontsize=15, pad=12)
    save(fig, "14_pipeline_diagram.png")


def chart_persona_distribution():
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    counts = df["persona"].map(PERSONA_TH).value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [PERSONA_COLORS.get(p, "#888") for p in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, colors=colors,
        autopct="%1.0f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2))
    ax.set_title("สัดส่วนของแต่ละกลุ่มลูกค้า (124 คน)", fontsize=14)
    save(fig, "15_persona_pie.png")


def chart_correlation_heatmap():
    """Pearson correlation heatmap"""
    corr = pd.read_csv(DATA / "correlation_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                 center=0, vmin=-1, vmax=1, square=True,
                 annot_kws={"size": 8}, cbar_kws={"label": "Pearson r"},
                 ax=ax)
    ax.set_title("Correlation Matrix (Pearson)", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    save(fig, "16_correlation_heatmap.png")


def chart_anomaly_scatter():
    """Anomaly Detection on PCA projection"""
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    normal = df[df["is_anomaly_consensus"] == 0]
    consensus = df[df["is_anomaly_consensus"] == 1]
    only_if = df[(df["is_anomaly_iforest"] == 1) &
                 (df["is_anomaly_consensus"] == 0)]
    only_lof = df[(df["is_anomaly_lof"] == 1) &
                  (df["is_anomaly_consensus"] == 0)]

    ax.scatter(normal["pca_x"], normal["pca_y"], s=40, c="#cccccc",
                alpha=0.5, label=f"ปกติ ({len(normal)})")
    if len(only_if):
        ax.scatter(only_if["pca_x"], only_if["pca_y"], s=80, c="#fd7e14",
                    marker="^", edgecolors="white", linewidth=1,
                    label=f"IForest only ({len(only_if)})")
    if len(only_lof):
        ax.scatter(only_lof["pca_x"], only_lof["pca_y"], s=80, c="#17a2b8",
                    marker="D", edgecolors="white", linewidth=1,
                    label=f"LOF only ({len(only_lof)})")
    if len(consensus):
        ax.scatter(consensus["pca_x"], consensus["pca_y"], s=180, c="#dc3545",
                    marker="*", edgecolors="white", linewidth=1.5,
                    label=f"Consensus ({len(consensus)})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Anomaly Detection on PCA Projection", fontsize=14)
    ax.legend(loc="best")
    save(fig, "17_anomaly_scatter.png")


PERSONA_DISPLAY_ORDER = ["Brand Advocate", "Partial Connect",
                          "Aware but Lost", "Disconnected"]


def _persona_pivot_pct(df, cols, label_map=None):
    """Group by persona, mean × 100, reindex to standard order."""
    grp = df.groupby("persona")[cols].mean() * 100
    grp = grp.reindex(PERSONA_DISPLAY_ORDER).fillna(0)
    if label_map:
        grp.columns = [label_map.get(c, c) for c in grp.columns]
    grp.index = [PERSONA_TH.get(p, p) for p in grp.index]
    return grp


def chart_persona_demographics():
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    age_label = {1: "<20", 2: "20-29", 3: "30-39", 4: "40-49", 5: "50+"}
    df["age_lbl"] = df["age"].map(age_label).fillna("?")
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    age_order = ["<20", "20-29", "30-39", "40-49", "50+"]
    gender_colors = {"หญิง": "#e91e63", "ชาย": "#2196f3", "LGBTQ+": "#9c27b0"}
    for ax, p in zip(axes, PERSONA_DISPLAY_ORDER):
        sub = df[df["persona"] == p]
        n = len(sub)
        if n == 0:
            ax.set_title(f"{PERSONA_TH[p]} (n=0)")
            continue
        ct = sub.groupby(["age_lbl", "gender"]).size().unstack(fill_value=0)
        ct = ct.reindex(index=age_order, fill_value=0)
        bottom = np.zeros(len(age_order))
        for g in ct.columns:
            ax.bar(age_order, ct[g], bottom=bottom,
                    color=gender_colors.get(g, "#999"), label=g)
            for i, v in enumerate(ct[g]):
                if v > 0:
                    ax.text(i, bottom[i] + v / 2, f"{int(v)}",
                             ha="center", va="center", color="white",
                             fontsize=8, fontweight="bold")
            bottom += ct[g].values
        ax.set_title(f"{PERSONA_TH[p]} (n={n})", fontsize=11)
        ax.set_xlabel("ช่วงอายุ")
        if ax == axes[0]:
            ax.set_ylabel("จำนวนคน")
        ax.tick_params(axis="x", rotation=30)
    axes[-1].legend(title="เพศ", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.suptitle("ประชากรศาสตร์ (อายุ × เพศ) แยกตามกลุ่ม", fontsize=14)
    save(fig, "21_persona_demographics.png")


def chart_persona_occasion_heatmap():
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    occ_cols = ["occ_work_study", "occ_leisure", "occ_late_night",
                "occ_between_meals", "occ_other"]
    occ_th = {"occ_work_study": "ทำงาน/เรียน", "occ_leisure": "พักผ่อน/บันเทิง",
              "occ_late_night": "ดึก/ก่อนนอน",
              "occ_between_meals": "ระหว่างมื้อ", "occ_other": "อื่นๆ"}
    grp = _persona_pivot_pct(df, occ_cols, label_map=occ_th)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(grp, annot=True, fmt=".0f", cmap="YlOrRd",
                 cbar_kws={"label": "% ของกลุ่ม"}, ax=ax)
    ax.set_title("โอกาสในการบริโภคขนม — % ภายในแต่ละกลุ่ม", fontsize=13)
    ax.set_xlabel("โอกาส")
    ax.set_ylabel("กลุ่มลูกค้า")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    save(fig, "22_persona_occasion_heatmap.png")


def chart_persona_topmind_heatmap():
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    tm_cols = [c for c in df.columns if c.startswith("topmind_")]
    tm_th = {"topmind_subbrand": "แบรนด์ลูก", "topmind_parent_brand": "คาลโวร่า",
             "topmind_shrimp_cracker": "ข้าวเกรียบ", "topmind_shrimp": "กุ้ง",
             "topmind_potato": "มันฝรั่ง", "topmind_snack_generic": "ขนม (ทั่วไป)",
             "topmind_japan": "ญี่ปุ่น", "topmind_natural_health": "ธรรมชาติ/สุขภาพ",
             "topmind_positive_attr": "อร่อย/กรอบ",
             "topmind_visual_packaging": "ภาพ/สี"}
    grp = _persona_pivot_pct(df, tm_cols, label_map=tm_th)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(grp, annot=True, fmt=".0f", cmap="Purples",
                 cbar_kws={"label": "% ของกลุ่ม"}, ax=ax)
    ax.set_title("สิ่งที่นึกถึงเมื่อพูดถึง Calvora — แยกตามกลุ่ม", fontsize=13)
    ax.set_xlabel("Theme")
    ax.set_ylabel("กลุ่มลูกค้า")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    save(fig, "23_persona_topmind_heatmap.png")


def chart_persona_subbrand_trial_heatmap():
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    tried_cols = [c for c in df.columns if c.startswith("tried_")]
    grp = df.groupby("persona")[tried_cols].mean() * 100
    grp = grp.reindex(PERSONA_DISPLAY_ORDER).fillna(0)
    grp.columns = [c.replace("tried_", "") for c in grp.columns]
    grp.index = [PERSONA_TH.get(p, p) for p in grp.index]
    col_order = grp.mean().sort_values(ascending=False).index
    grp = grp[col_order]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(grp, annot=True, fmt=".0f", cmap="Greens",
                 cbar_kws={"label": "% เคยทาน"}, ax=ax)
    ax.set_title("อัตราการทดลองแบรนด์ — แยกตามกลุ่ม", fontsize=13)
    ax.set_xlabel("แบรนด์")
    ax.set_ylabel("กลุ่มลูกค้า")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    save(fig, "24_persona_subbrand_trial.png")


# def chart_persona_ebisen_engagement():
#     df = pd.read_csv(DATA / "calvora_clustered.csv")
#     metrics = []
#     for p in PERSONA_DISPLAY_ORDER:
#         sub = df[df["persona"] == p]
#         if len(sub) == 0:
#             continue
#         metrics.append({
#             "กลุ่ม": PERSONA_TH[p],
#             "เคยทาน Ebisen": 100 * (sub["ebisen_familiarity"] == 3).mean(),
#             "เชื่อว่าทำจากกุ้งแท้": 100 * sub["ebisen_belief"].fillna(0).mean(),
#             "อยากลองรสชาติใหม่": 100 * sub["trial_intent"].fillna(0).mean(),
#         })
#     pdf = pd.DataFrame(metrics).set_index("กลุ่ม")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     x = np.arange(len(pdf))
#     w = 0.27
#     colors = ["#16a085", "#f39c12", "#3498db"]
#     for i, col in enumerate(pdf.columns):
#         bars = ax.bar(x + (i - 1) * w, pdf[col], w, color=colors[i], label=col)
#         for b, v in zip(bars, pdf[col]):
#             ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%",
#                      ha="center", fontsize=9)
#     ax.set_xticks(x)
#     ax.set_xticklabels(pdf.index)
#     ax.set_ylabel("% ของกลุ่ม")
#     ax.set_title("ความผูกพันกับ Ebisen — แยกตามกลุ่ม", fontsize=13)
#     ax.legend(loc="upper right")
#     ax.set_ylim(0, max(110, pdf.values.max() + 15))
#     save(fig, "25_persona_ebisen_engagement.png")


def chart_persona_strengths_heatmap():
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    str_cols = [c for c in df.columns if c.startswith("strength_")]
    if not str_cols:
        return
    strength_th = {
        "ใช้วัตถุดิบจากธรรมชาติ (Use natural ingredients)": "วัตถุดิบธรรมชาติ",
        "ทำจากเนื้อสัตว์แท้ (From real meat)": "เนื้อสัตว์แท้",
        "มีรสชาติอร่อย (Tasty)": "รสชาติอร่อย",
        "มีคุณภาพดี (Good quality)": "คุณภาพดี",
        "เป็นแบรนด์ญี่ปุ่น (Japan brand)": "แบรนด์ญี่ปุ่น",
        "เพื่อสุขภาพที่ดี (For healthy lifestyles)": "เพื่อสุขภาพ",
    }
    grp = df.groupby("persona")[str_cols].mean() * 100
    grp = grp.reindex(PERSONA_DISPLAY_ORDER).fillna(0)
    grp.columns = [strength_th.get(c.replace("strength_", ""),
                                     c.replace("strength_", "")[:20])
                    for c in grp.columns]
    grp.index = [PERSONA_TH.get(p, p) for p in grp.index]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(grp, annot=True, fmt=".0f", cmap="BuGn",
                 cbar_kws={"label": "% ของกลุ่ม"}, ax=ax)
    ax.set_title("จุดเด่นของ Calvora ที่แต่ละกลุ่มรับรู้", fontsize=13)
    ax.set_xlabel("จุดเด่น")
    ax.set_ylabel("กลุ่มลูกค้า")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    save(fig, "26_persona_strengths.png")


def chart_belief_reason_crosstab():
    """Stacked bar — believer vs disbeliever × reason"""
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    df = df[df["belief_reason"] != "no_response"]
    pivot = df.groupby(["belief_reason", "ebisen_belief"]).size().unstack(fill_value=0)
    pivot.columns = ["ไม่เชื่อ" if c == 0 else "เชื่อ" for c in pivot.columns]
    reason_th = {
        "sensory": "รสชาติ/กลิ่น", "brand_image": "ภาพลักษณ์แบรนด์",
        "skeptical": "สงสัย (ผง/แต่ง)", "label_evidence": "ฉลาก/ส่วนผสม",
        "experience": "ประสบการณ์ตรง", "other": "อื่นๆ",
    }
    pivot.index = [reason_th.get(i, i) for i in pivot.index]
    pivot = pivot.sort_values(by=pivot.columns.tolist(), ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors_map = {"เชื่อ": "#28a745", "ไม่เชื่อ": "#dc3545"}
    pivot.plot(kind="bar", stacked=True, ax=ax,
                color=[colors_map.get(c, "#888") for c in pivot.columns])
    for container in ax.containers:
        labels = [f"{int(v)}" if v > 0 else "" for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type="center",
                      color="white", fontweight="bold")
    ax.set_title("เหตุผลที่เชื่อ vs ไม่เชื่อว่า Ebisen ทำจากกุ้งแท้",
                  fontsize=14)
    ax.set_xlabel("เหตุผล")
    ax.set_ylabel("จำนวนคน")
    ax.legend(title="ผู้บริโภค")
    plt.xticks(rotation=20, ha="right")
    save(fig, "19_belief_reason_crosstab.png")


def chart_eating_occasion_flags():
    """Multi-label eating occasion bar"""
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    occ_cols = [c for c in df.columns if c.startswith("occ_")]
    counts = {c.replace("occ_", ""): int(df[c].sum()) for c in occ_cols}
    label_th = {
        "work_study": "ทำงาน/เรียน", "leisure": "พักผ่อน/บันเทิง",
        "late_night": "ดึก/ก่อนนอน", "between_meals": "ระหว่างมื้อ",
        "other": "อื่นๆ",
    }
    cdf = pd.DataFrame([{"occasion": label_th.get(k, k), "n": v}
                         for k, v in counts.items()])
    cdf = cdf.sort_values("n", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(cdf["occasion"], cdf["n"],
                    color=plt.cm.Blues(np.linspace(0.4, 0.9, len(cdf))))
    for b, v in zip(bars, cdf["n"]):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2,
                 str(int(v)), va="center")
    ax.set_title("โอกาสในการบริโภคขนม (Multi-label)", fontsize=14)
    ax.set_xlabel("จำนวนคน")
    save(fig, "20_eating_occasion_flags.png")


def chart_descriptive_box():
    """Boxplot สรุปการกระจายของ features หลัก"""
    df = pd.read_csv(DATA / "calvora_clustered.csv")
    features = ["brand_correct", "brand_decoy_hit", "brand_understanding",
                "products_recognized_count", "products_tried_count"]
    labels = ["correct\n(0-7)", "decoy\n(0-3)", "understand\n(0-11)",
              "recog\ncount", "tried\ncount"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot([df[f] for f in features], labels=labels,
                     patch_artist=True, medianprops=dict(color="black"))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(features)))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_title("Descriptive Statistics — Boxplot ของ features หลัก",
                  fontsize=13)
    ax.set_ylabel("ค่า")
    save(fig, "18_descriptive_boxplot.png")


if __name__ == "__main__":
    print("Exporting charts...")
    chart_understanding_histogram()
    chart_awareness_tier()
    chart_subbrand_recognition()
    chart_decoy_confusion()
    chart_k_evaluation()
    chart_pca_scatter()
    chart_persona_radar()
    chart_model_comparison()
    chart_confusion_matrix()
    chart_feature_importance()
    chart_trust_factors()
    chart_ebisen_flavors()
    chart_attribution_gap()
    chart_pipeline_diagram()
    chart_persona_distribution()
    chart_correlation_heatmap()
    chart_anomaly_scatter()
    chart_descriptive_box()
    chart_belief_reason_crosstab()
    chart_eating_occasion_flags()
    chart_persona_demographics()
    chart_persona_occasion_heatmap()
    chart_persona_topmind_heatmap()
    chart_persona_subbrand_trial_heatmap()
    # chart_persona_ebisen_engagement()
    chart_persona_strengths_heatmap()
    print(f"\nDone. {len(list(OUT.glob('*.png')))} files in {OUT}")
