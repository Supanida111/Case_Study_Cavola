"""หน้า Unsupervised Learning — การจัดกลุ่มลูกค้า"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from db import query

BASE = Path(__file__).resolve().parent.parent
METRICS = BASE / "data" / "clustering_metrics.json"
DESC_STATS = BASE / "data" / "descriptive_stats.json"
ANOMALY_REPORT = BASE / "data" / "anomaly_report.json"

PERSONA_COLORS = {
    "Brand Advocate": "#88D66C",
    "Partial Connect": "#6CB4EE",
    "Aware but Lost": "#FFD93D",
    "Disconnected": "#FF8E8E",
}

PERSONA_TH = {
    "Brand Advocate": "ผู้สนับสนุนแบรนด์",
    "Partial Connect": "เชื่อมโยงบางส่วน",
    "Aware but Lost": "รู้จักแต่สับสน",
    "Disconnected": "ไม่เชื่อมโยง",
}

THAI_FONT = dict(family="Mali, Itim, Quicksand, sans-serif")


def load_metrics():
    with open(METRICS, "r", encoding="utf-8") as f:
        return json.load(f)


def load_desc_stats():
    with open(DESC_STATS, "r", encoding="utf-8") as f:
        return json.load(f)


def load_anomaly():
    with open(ANOMALY_REPORT, "r", encoding="utf-8") as f:
        return json.load(f)


def descriptive_stats_table():
    desc = load_desc_stats()
    rows = []
    for f, s in desc["numeric"].items():
        rows.append({
            "Feature": f,
            "Mean": round(s["mean"], 2),
            "Std": round(s["std"], 2),
            "Min": int(s["min"]),
            "Median": round(s["median"], 1),
            "Max": int(s["max"]),
        })
    df = pd.DataFrame(rows)
    return dbc.Table.from_dataframe(df, striped=True, bordered=True,
                                     hover=True, responsive=True, size="sm")


def correlation_heatmap():
    corr = query("SELECT * FROM correlation_matrix")
    pivot = corr.pivot(index="feature_a", columns="feature_b",
                        values="pearson_r")
    fig = px.imshow(pivot, color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, aspect="auto",
                     title="Correlation Matrix (Pearson r)",
                     labels=dict(color="r"))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10),
                      font=THAI_FONT)
    return fig


def anomaly_scatter():
    df = query("SELECT pca_x, pca_y, persona, "
                "is_anomaly_iforest, is_anomaly_lof, "
                "is_anomaly_consensus FROM survey_responses")
    fig = go.Figure()
    # Normal points
    normal = df[df["is_anomaly_consensus"] == 0]
    fig.add_trace(go.Scatter(x=normal["pca_x"], y=normal["pca_y"],
                              mode="markers",
                              marker=dict(size=8, color="#D1D1D1",
                                           opacity=0.5),
                              name="ปกติ"))
    # IForest only
    only_if = df[(df["is_anomaly_iforest"] == 1) &
                  (df["is_anomaly_consensus"] == 0)]
    if len(only_if):
        fig.add_trace(go.Scatter(x=only_if["pca_x"], y=only_if["pca_y"],
                                  mode="markers",
                                  marker=dict(size=11,
                                               color="#FFB067",
                                               symbol="triangle-up",
                                               line=dict(color="white",
                                                         width=1)),
                                  name="IForest only"))
    # LOF only
    only_lof = df[(df["is_anomaly_lof"] == 1) &
                   (df["is_anomaly_consensus"] == 0)]
    if len(only_lof):
        fig.add_trace(go.Scatter(x=only_lof["pca_x"], y=only_lof["pca_y"],
                                  mode="markers",
                                  marker=dict(size=11,
                                               color="#6CB4EE",
                                               symbol="diamond",
                                               line=dict(color="white",
                                                         width=1)),
                                  name="LOF only"))
    # Consensus
    consensus = df[df["is_anomaly_consensus"] == 1]
    if len(consensus):
        fig.add_trace(go.Scatter(x=consensus["pca_x"], y=consensus["pca_y"],
                                  mode="markers",
                                  marker=dict(size=14,
                                               color="#FF8E8E",
                                               symbol="star",
                                               line=dict(color="white",
                                                         width=1.5)),
                                  name="Consensus (ทั้ง 2 วิธี)"))
    fig.update_layout(title="Anomaly Detection on PCA Projection",
                       height=480, margin=dict(l=10, r=10, t=50, b=10),
                       font=THAI_FONT,
                       xaxis_title="PC1", yaxis_title="PC2")
    return fig


def indicator_card(title, value, decimals=3):
    return dbc.Card([
        dbc.CardBody([
            html.Div(title, className="text-muted small"),
            html.H4(f"{value:.{decimals}f}" if isinstance(value, float)
                    else str(value), style={"color": "#88D66C", "fontWeight": "700"}),
        ])
    ], className="shadow-sm")


def k_evaluation_chart():
    """กราฟ Silhouette ของทั้ง K-Means และ Hierarchical Ward สำหรับ k=2..10
    เพื่อ verify ว่า k=4 robust ทั้ง 2 algorithms"""
    df = query("SELECT * FROM cluster_k_evaluation")
    fig = go.Figure()
    # K-Means Silhouette
    fig.add_trace(go.Scatter(x=df["k"], y=df["silhouette"],
                              mode="lines+markers",
                              name="K-Means Silhouette (วิธีหลัก)",
                              line=dict(color="#6CB4EE", width=3),
                              marker=dict(size=10)))
    # Hierarchical Silhouette (ถ้ามี)
    if "hier_silhouette" in df.columns:
        fig.add_trace(go.Scatter(x=df["k"], y=df["hier_silhouette"],
                                  mode="lines+markers",
                                  name="Hierarchical Ward Silhouette (ยืนยัน)",
                                  line=dict(color="#B19CD9", width=2,
                                             dash="dot"),
                                  marker=dict(size=8, symbol="diamond")))
    # Davies-Bouldin (K-Means) on secondary axis
    fig.add_trace(go.Scatter(x=df["k"], y=df["davies_bouldin"],
                              mode="lines+markers",
                              name="K-Means Davies-Bouldin (ต่ำ=ดี)",
                              line=dict(color="#FF8E8E", width=2, dash="dash"),
                              marker=dict(size=8, symbol="square"),
                              yaxis="y2"))
    fig.update_layout(
        title="การเลือกจำนวนกลุ่ม k — เทียบ K-Means vs Hierarchical Ward",
        xaxis_title="จำนวนกลุ่ม (k)",
        yaxis=dict(title="Silhouette Score (สูง = ดี)"),
        yaxis2=dict(title="Davies-Bouldin Index (ต่ำ = ดี)",
                    overlaying="y", side="right"),
        height=440, margin=dict(l=10, r=10, t=50, b=80),
        font=THAI_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                     xanchor="center", x=0.5),
    )
    fig.add_vline(x=4, line_dash="dash", line_color="red",
                   annotation_text="เลือก k=4",
                   annotation_position="top")
    return fig


def pca_scatter():
    df = query("SELECT pca_x, pca_y, persona FROM survey_responses")
    df["กลุ่ม"] = df["persona"].map(PERSONA_TH).fillna(df["persona"])
    fig = px.scatter(df, x="pca_x", y="pca_y", color="กลุ่ม",
                     color_discrete_map={PERSONA_TH[k]: v
                                          for k, v in PERSONA_COLORS.items()},
                     title="การกระจายตัวของลูกค้าด้วย PCA (2 มิติ)",
                     hover_data=["กลุ่ม"])
    fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10),
                      font=THAI_FONT,
                      xaxis_title="PC1", yaxis_title="PC2")
    return fig


def persona_radar():
    df = query("SELECT persona, score_ingredient, score_taste, score_variety, "
                "score_texture, score_health FROM survey_responses")
    agg = df.groupby("persona").mean().reset_index()
    fig = go.Figure()
    factors = ["score_ingredient", "score_taste", "score_variety",
               "score_texture", "score_health"]
    labels = ["วัตถุดิบคุณภาพ", "รสชาติอร่อย", "รสชาติหลากหลาย",
              "สัมผัสกรุบกรอบ", "สุขภาพดี"]
    for _, row in agg.iterrows():
        name_th = PERSONA_TH.get(row["persona"], row["persona"])
        fig.add_trace(go.Scatterpolar(
            r=[row[f] for f in factors] + [row[factors[0]]],
            theta=labels + [labels[0]],
            fill="toself",
            name=name_th,
            line=dict(color=PERSONA_COLORS.get(row["persona"], "#888")),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
        title="ความสำคัญของปัจจัยการซื้อ แยกตามกลุ่ม",
        height=500, margin=dict(l=10, r=10, t=50, b=10),
        font=THAI_FONT,
    )
    return fig


PERSONA_DISPLAY_ORDER = ["Brand Advocate", "Partial Connect",
                          "Aware but Lost", "Disconnected"]


def _persona_pivot(metric_cols, label_map=None, normalize_by_persona=True):
    """Helper: build pivot table of persona × metric_cols.
    Returns normalized rates (% within persona) by default."""
    cols_str = ", ".join(metric_cols)
    df = query(f"SELECT persona, {cols_str} FROM survey_responses")
    if label_map:
        df = df.rename(columns=label_map)
        metric_cols = [label_map.get(c, c) for c in metric_cols]
    grp = df.groupby("persona")[metric_cols].mean() * 100  # % rate
    grp = grp.reindex(PERSONA_DISPLAY_ORDER).fillna(0)
    grp.index = [PERSONA_TH.get(p, p) for p in grp.index]
    return grp


def persona_demographics_chart():
    """Age × Gender distribution per persona"""
    df = query("SELECT persona, age, gender FROM survey_responses")
    age_label = {1: "<20", 2: "20-29", 3: "30-39", 4: "40-49", 5: "50+"}
    df["age_lbl"] = df["age"].map(age_label).fillna("?")
    df["กลุ่ม"] = df["persona"].map(PERSONA_TH).fillna(df["persona"])

    # Faceted by persona — stacked age × gender count
    pivot = df.groupby(["กลุ่ม", "age_lbl", "gender"]).size().reset_index(name="n")
    fig = px.bar(pivot, x="age_lbl", y="n", color="gender",
                  facet_col="กลุ่ม",
                  category_orders={"age_lbl": ["<20","20-29","30-39","40-49","50+"],
                                    "กลุ่ม": [PERSONA_TH[p] for p in PERSONA_DISPLAY_ORDER]},
                  color_discrete_map={"หญิง": "#e91e63", "ชาย": "#2196f3",
                                       "LGBTQ+": "#9c27b0"},
                  title="อายุ × เพศ แยกตามกลุ่ม",
                  text="n")
    fig.update_traces(textposition="inside")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=70, b=10),
                       font=THAI_FONT, yaxis_title="จำนวนคน",
                       legend_title="เพศ")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


def persona_occasion_heatmap():
    """Eating occasion (multi-label) × persona — % within persona"""
    occ_cols = ["occ_work_study", "occ_leisure", "occ_late_night",
                "occ_between_meals", "occ_other"]
    occ_th = {"occ_work_study": "ทำงาน/เรียน",
              "occ_leisure": "พักผ่อน/บันเทิง",
              "occ_late_night": "ดึก/ก่อนนอน",
              "occ_between_meals": "ระหว่างมื้อ",
              "occ_other": "อื่นๆ"}
    grp = _persona_pivot(occ_cols, label_map=occ_th)
    fig = px.imshow(grp, text_auto=".0f", aspect="auto",
                     color_continuous_scale="YlOrRd",
                     labels=dict(x="โอกาส", y="กลุ่มลูกค้า", color="% ของกลุ่ม"),
                     title="โอกาสในการบริโภคขนม — % ภายในแต่ละกลุ่ม")
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10),
                       font=THAI_FONT)
    return fig


def persona_topmind_heatmap():
    """Top-of-mind themes × persona — % within persona"""
    tm_cols = [c for c in ["topmind_subbrand", "topmind_parent_brand",
                            "topmind_shrimp_cracker", "topmind_shrimp",
                            "topmind_potato", "topmind_snack_generic",
                            "topmind_japan", "topmind_natural_health",
                            "topmind_positive_attr", "topmind_visual_packaging"]]
    tm_th = {"topmind_subbrand": "แบรนด์ลูก", "topmind_parent_brand": "คาลโวร่า",
             "topmind_shrimp_cracker": "ข้าวเกรียบ", "topmind_shrimp": "กุ้ง",
             "topmind_potato": "มันฝรั่ง", "topmind_snack_generic": "ขนม (ทั่วไป)",
             "topmind_japan": "ญี่ปุ่น", "topmind_natural_health": "ธรรมชาติ/สุขภาพ",
             "topmind_positive_attr": "อร่อย/กรอบ", "topmind_visual_packaging": "ภาพ/สี"}
    grp = _persona_pivot(tm_cols, label_map=tm_th)
    fig = px.imshow(grp, text_auto=".0f", aspect="auto",
                     color_continuous_scale="Purples",
                     labels=dict(x="Theme", y="กลุ่มลูกค้า", color="% ของกลุ่ม"),
                     title="สิ่งที่นึกถึงเมื่อพูดถึง Calvora — % ภายในแต่ละกลุ่ม")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=80),
                       font=THAI_FONT,
                       xaxis=dict(tickangle=-30))
    return fig


def persona_subbrand_trial_heatmap():
    """Sub-brand trial rate × persona"""
    df = query("SELECT * FROM survey_responses")
    tried_cols = [c for c in df.columns if c.startswith("tried_")]
    grp = df.groupby("persona")[tried_cols].mean() * 100
    grp = grp.reindex(PERSONA_DISPLAY_ORDER).fillna(0)
    grp.columns = [c.replace("tried_", "") for c in grp.columns]
    grp.index = [PERSONA_TH.get(p, p) for p in grp.index]
    # Sort columns by overall popularity (descending)
    col_order = grp.mean().sort_values(ascending=False).index
    grp = grp[col_order]
    fig = px.imshow(grp, text_auto=".0f", aspect="auto",
                     color_continuous_scale="Greens",
                     labels=dict(x="แบรนด์ลูก", y="กลุ่มลูกค้า",
                                  color="% เคยทาน"),
                     title="แบรนด์ที่ลูกค้าเคยกิน % ภายในแต่ละกลุ่ม")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=80),
                       font=THAI_FONT,
                       xaxis=dict(tickangle=-30))
    return fig


def persona_ebisen_engagement_chart():
    """Ebisen familiarity + belief + trial_intent by persona"""
    df = query("SELECT persona, ebisen_familiarity, ebisen_belief, "
                "trial_intent FROM survey_responses")
    rows = []
    for p in PERSONA_DISPLAY_ORDER:
        sub = df[df["persona"] == p]
        if len(sub) == 0:
            continue
        rows.append({"กลุ่ม": PERSONA_TH[p],
                     "metric": "เคยทาน Ebisen (%)",
                     "value": 100 * (sub["ebisen_familiarity"] == 3).mean()})
        rows.append({"กลุ่ม": PERSONA_TH[p],
                     "metric": "เชื่อว่าทำจากกุ้งแท้ (%)",
                     "value": 100 * sub["ebisen_belief"].fillna(0).mean()})
        rows.append({"กลุ่ม": PERSONA_TH[p],
                     "metric": "อยากลองรสชาติใหม่ (%)",
                     "value": 100 * sub["trial_intent"].fillna(0).mean()})
    pdf = pd.DataFrame(rows)
    fig = px.bar(pdf, x="กลุ่ม", y="value", color="metric",
                  barmode="group",
                  title="ความผูกพันกับ Ebisen — แยกตามกลุ่ม",
                  color_discrete_sequence=["#16a085", "#f39c12", "#3498db"],
                  text="value")
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10),
                       font=THAI_FONT, yaxis_title="% ของกลุ่ม",
                       xaxis_title="", legend_title="ตัวชี้วัด")
    return fig


def persona_strengths_heatmap():
    """Calvora strengths attribution × persona"""
    df = query("SELECT * FROM survey_responses")
    str_cols = [c for c in df.columns if c.startswith("strength_")]
    if not str_cols:
        return go.Figure()
    grp = df.groupby("persona")[str_cols].mean() * 100
    grp = grp.reindex(PERSONA_DISPLAY_ORDER).fillna(0)
    # Shorten labels
    strength_th = {
        "ใช้วัตถุดิบจากธรรมชาติ (Use natural ingredients)": "วัตถุดิบธรรมชาติ",
        "ทำจากเนื้อสัตว์แท้ (From real meat)": "เนื้อสัตว์แท้",
        "มีรสชาติอร่อย (Tasty)": "รสชาติอร่อย",
        "มีคุณภาพดี (Good quality)": "คุณภาพดี",
        "เป็นแบรนด์ญี่ปุ่น (Japan brand)": "แบรนด์ญี่ปุ่น",
        "เพื่อสุขภาพที่ดี (For healthy lifestyles)": "เพื่อสุขภาพ",
    }
    grp.columns = [strength_th.get(c.replace("strength_", ""),
                                     c.replace("strength_", "")[:20])
                    for c in grp.columns]
    grp.index = [PERSONA_TH.get(p, p) for p in grp.index]
    fig = px.imshow(grp, text_auto=".0f", aspect="auto",
                     color_continuous_scale="Teal",
                     labels=dict(x="จุดเด่น", y="กลุ่มลูกค้า",
                                  color="% ของกลุ่ม"),
                     title="จุดเด่นของ Calvora ที่แต่ละกลุ่มรับรู้")
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10),
                       font=THAI_FONT)
    return fig


def persona_table():
    df = query("SELECT persona, COUNT(*) as count, "
                "AVG(brand_understanding) as avg_understanding, "
                "AVG(brand_correct) as avg_correct, "
                "AVG(brand_decoy_hit) as avg_decoy, "
                "AVG(products_tried_count) as avg_tried, "
                "AVG(age) as avg_age FROM survey_responses GROUP BY persona "
                "ORDER BY avg_understanding DESC")
    df["กลุ่ม"] = df["persona"].map(PERSONA_TH).fillna(df["persona"])
    df["จำนวน (คน)"] = df["count"]
    df["คะแนนเข้าใจแบรนด์ (เฉลี่ย/11)"] = df["avg_understanding"].round(2)
    df["ตอบถูกแบรนด์จริง (เฉลี่ย/7)"] = df["avg_correct"].round(2)
    df["สับสนคู่แข่ง (เฉลี่ย/3)"] = df["avg_decoy"].round(2)
    df["เคยทาน (จำนวนแบรนด์เฉลี่ย)"] = df["avg_tried"].round(2)
    df["ช่วงอายุ (เฉลี่ย)"] = df["avg_age"].round(2)
    show = df[["กลุ่ม", "จำนวน (คน)", "คะแนนเข้าใจแบรนด์ (เฉลี่ย/11)",
               "ตอบถูกแบรนด์จริง (เฉลี่ย/7)", "สับสนคู่แข่ง (เฉลี่ย/3)",
               "เคยทาน (จำนวนแบรนด์เฉลี่ย)", "ช่วงอายุ (เฉลี่ย)"]]
    return dbc.Table.from_dataframe(show, striped=True, bordered=True,
                                     hover=True, responsive=True)


def layout():
    m = load_metrics()
    anom = load_anomaly()
    return dbc.Container([
        html.H2("Unsupervised Learning — การวิเคราะห์เชิงสำรวจและจัดกลุ่ม"),
        html.P("ครอบคลุม 5 ส่วน: Descriptive Statistics → Correlation → "
               "EDA → Anomaly Detection → Clustering + Dimensional Reduction",
               className="text-muted"),

        # === Section 1: Descriptive Statistics ===
        html.H4("1. Descriptive Statistics", className="mt-3"),
        html.P("สรุปสถิติของ 15 features ที่ใช้ในการวิเคราะห์",
               className="text-muted small"),
        descriptive_stats_table(),

        # === Section 2: Correlation Analysis ===
        html.H4("2. Correlation Analysis", className="mt-4"),
        html.P("ความสัมพันธ์ Pearson ระหว่างทุกคู่ feature "
               "(สีแดง = positive, สีน้ำเงิน = negative)",
               className="text-muted small"),
        dcc.Graph(figure=correlation_heatmap()),

        # === Section 3: EDA Indicators ===
        html.H4("3. Exploratory Data Analysis (EDA)", className="mt-4"),
        dbc.Row([
            dbc.Col(indicator_card("ผู้ตอบ (Aware)", "124", 0), md=3),
            dbc.Col(indicator_card("Tagline known", "3 (2.4%)", 0), md=3),
            dbc.Col(indicator_card("brand_correct (เฉลี่ย)",
                                    "3.07 / 7", 0), md=3),
            dbc.Col(indicator_card("brand_understanding (เฉลี่ย)",
                                    "5.87 / 11", 0), md=3),
        ], className="mb-3"),

        # === Section 4: Anomaly Detection ===
        html.H4("4. Anomaly Detection", className="mt-4"),
        html.P("ตรวจจับ outliers ก่อน clustering ด้วย Isolation Forest "
               "+ LOF (contamination=5%) — ใช้ flag เท่านั้น ไม่ remove "
               "เนื่องจาก sample เล็ก",
               className="text-muted small"),
        dbc.Row([
            dbc.Col(indicator_card("Isolation Forest",
                                    f"{anom['n_iforest']} คน", 0), md=3),
            dbc.Col(indicator_card("Local Outlier Factor",
                                    f"{anom['n_lof']} คน", 0), md=3),
            dbc.Col(indicator_card("Consensus (ทั้ง 2 วิธี)",
                                    f"{anom['n_consensus']} คน", 0), md=3),
            dbc.Col(indicator_card("Policy", "FLAG only", 0), md=3),
        ], className="mb-3"),
        dcc.Graph(figure=anomaly_scatter()),

        # === Section 5: Clustering ===
        html.H4("5. Clustering + Dimensional Reduction", className="mt-4"),
        html.P("K-Means (วิธีหลัก) + Hierarchical Ward (ยืนยัน) + "
               "PCA(2) สำหรับ visualization",
               className="text-muted small"),

        dbc.Row([
            dbc.Col(indicator_card("จำนวนกลุ่มที่เลือก (k)",
                                    m["selected_k"], 0), md=3),
            dbc.Col(indicator_card("Silhouette (K-Means)",
                                    m["kmeans_silhouette"]), md=3),
            dbc.Col(indicator_card("Davies-Bouldin (ต่ำ=ดี)",
                                    m["kmeans_db"]), md=3),
            dbc.Col(indicator_card("Calinski-Harabasz",
                                    m["kmeans_ch"], 1), md=3),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=k_evaluation_chart()), md=12),
        ], className="mb-2"),

        dbc.Alert([
            html.H6("💡 ทำไมใช้ K-Means เป็นวิธีหลัก + Hierarchical Ward เป็นวิธียืนยัน?",
                     className="alert-heading"),
            html.Hr(),
            html.P([
                html.Strong("วัตถุประสงค์: "),
                "ตรวจสอบว่าโครงสร้างกลุ่มลูกค้า ",
                html.Strong("ไม่ขึ้นกับ algorithm "),
                "(robustness check)"
            ], className="mb-2"),
            html.Ul([
                html.Li([html.Strong("K-Means: "),
                          "วิธีหลัก ใช้ผลลัพธ์เป็น persona — เน้นความเร็ว, "
                          "centroid-based"]),
                html.Li([html.Strong("Hierarchical Ward: "),
                          "วิธียืนยัน ใช้ same k → ถ้า Silhouette ใกล้กัน = "
                          "โครงสร้างเสถียร"]),
            ], className="mb-2"),
            html.P([
                html.Strong("ผลปัจจุบัน (k=4): "),
                "K-Means Silhouette = ",
                html.Code(f"{m['kmeans_silhouette']:.3f}"),
                " | Hierarchical Silhouette = ",
                html.Code(f"{m.get('hier_silhouette', 0):.3f}"),
                " → ต่างกันแค่ ",
                html.Strong(f"{abs(m['kmeans_silhouette'] - m.get('hier_silhouette', 0)):.3f}"),
                " = คลัสเตอร์เสถียรทั้ง 2 algorithms ✅"
            ], className="mb-0 small"),
        ], color="info", className="mb-4"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=pca_scatter()), md=6),
            dbc.Col(dcc.Graph(figure=persona_radar()), md=6),
        ], className="mb-3"),

        html.H4("ตารางสรุปกลุ่มลูกค้า (Persona Centroids)"),
        persona_table(),

        # === Section 6: Persona Deep Dive ===
        html.Hr(),
        html.H4("6. Persona Deep Dive — ลักษณะเฉพาะของแต่ละกลุ่ม",
                className="mt-4"),
        html.P("เจาะลึก insight ของแต่ละกลุ่มในมิติต่างๆ — "
               "ใครเป็นใคร, ทานตอนไหน, นึกถึงอะไร, ทดลองแบรนด์ลูกอะไร",
               className="text-muted small"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=persona_demographics_chart()), md=12),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=persona_occasion_heatmap()), md=6),
            dbc.Col(dcc.Graph(figure=persona_topmind_heatmap()), md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=persona_subbrand_trial_heatmap()), md=12),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=persona_strengths_heatmap()), md=12),
        ], className="mb-3"),
    ], fluid=True)
