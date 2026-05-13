"""หน้า Supervised Learning — ทำนายระดับความเข้าใจแบรนด์"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from db import query

BASE = Path(__file__).resolve().parent.parent
METRICS = BASE / "data" / "supervised_metrics.json"

THAI_FONT = dict(family="Mali, Itim, Quicksand, sans-serif")
TIER_TH = {"Low": "ต่ำ", "Mid": "กลาง", "High": "สูง"}


def load_metrics():
    with open(METRICS, "r", encoding="utf-8") as f:
        return json.load(f)


def indicator_card(title, value, color="primary"):
    return dbc.Card([
        dbc.CardBody([
            html.Div(title, className="text-muted small"),
            html.H4(value, style={"color": "#88D66C" if color=="success" else "#FFB067", "fontWeight": "700"}),
        ])
    ], className="shadow-sm")


def model_comparison_chart():
    df = query("SELECT * FROM supervised_metrics")

    # Reorder: put KNN baseline + KNN (Tuned) at the end, adjacent
    knn_models = ["K-Nearest Neighbors", "KNN (Tuned)"]
    others = df[~df["model"].isin(knn_models)].sort_values(
        "cv_f1_mean", ascending=True)
    knn_pair = df[df["model"].isin(knn_models)].sort_values(
        "model", ascending=True)  # baseline first, then tuned
    df = pd.concat([others, knn_pair], ignore_index=True)

    # Display labels: rename KNN baseline for clarity in chart
    label_map = {"K-Nearest Neighbors": "KNN (Base, k=5)",
                 "KNN (Tuned)": "KNN (Tuned, GridSearchCV)"}
    df["display_name"] = df["model"].map(label_map).fillna(df["model"])

    fig = go.Figure()
    fig.add_trace(go.Bar(name="CV F1 (เฉลี่ย)", x=df["display_name"],
                          y=df["cv_f1_mean"],
                          error_y=dict(type="data", array=df["cv_f1_std"]),
                          marker_color="#6CB4EE"))
    fig.add_trace(go.Bar(name="Test Accuracy", x=df["display_name"],
                          y=df["test_accuracy"], marker_color="#88D66C"))
    fig.add_trace(go.Bar(name="Test F1", x=df["display_name"],
                          y=df["test_f1"], marker_color="#FFD93D"))

    # Highlight KNN pair with a background shape + annotation
    knn_idx_start = len(others) - 0.5
    fig.add_vrect(x0=knn_idx_start, x1=knn_idx_start + 2,
                   fillcolor="#F0F9EE", opacity=0.4,
                   layer="below", line_width=0,
                   annotation_text="🔍 KNN Comparison",
                   annotation_position="top left",
                   annotation_font_size=11,
                   annotation_font_color="#2e7d32")

    # Arrow showing improvement from base → tuned (on CV F1)
    base_row = df[df["model"] == "K-Nearest Neighbors"]
    tuned_row = df[df["model"] == "KNN (Tuned)"]
    if len(base_row) and len(tuned_row):
        base_y = float(base_row["cv_f1_mean"].iloc[0])
        tuned_y = float(tuned_row["cv_f1_mean"].iloc[0])
        improvement = tuned_y - base_y
        # annotation pointing improvement
        fig.add_annotation(
            x=tuned_row["display_name"].iloc[0],
            y=tuned_y + 0.08,
            text=f"⬆ {improvement:+.3f} ({improvement/base_y*100:+.0f}%)",
            showarrow=False, font=dict(size=13, color="#2e7d32"),
            bgcolor="#e8f5e9", borderpad=4,
        )

    fig.update_layout(barmode="group",
                       title="เปรียบเทียบประสิทธิภาพโมเดล "
                              "(5-fold CV + 20% Test) — KNN Base vs Tuned",
                       yaxis_title="คะแนน",
                       yaxis=dict(range=[0, max(df["cv_f1_mean"].max(),
                                                  df["test_accuracy"].max(),
                                                  df["test_f1"].max()) * 1.25]),
                       height=440,
                       margin=dict(l=10, r=10, t=60, b=80),
                       font=THAI_FONT)
    return fig


def confusion_heatmap(metrics):
    best = metrics["best_model"]
    cm = np.array(metrics["results"][best]["confusion_matrix"])
    tiers_th = [TIER_TH[t] for t in metrics["tier_order"]]
    fig = px.imshow(cm, text_auto=True, aspect="auto",
                     x=tiers_th, y=tiers_th, color_continuous_scale=["#F0F9EE", "#6CB4EE"],
                     labels=dict(x="ทำนาย", y="จริง"),
                     title=f"Confusion Matrix — {best}")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10),
                       font=THAI_FONT)
    return fig


def feature_importance_chart():
    df = query("SELECT * FROM feature_importance ORDER BY importance ASC")
    feature_th_map = {
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
    df["feature_th"] = df["feature"].map(feature_th_map).fillna(df["feature"])
    fig = px.bar(df, x="importance", y="feature_th", orientation="h",
                  title="ความสำคัญของแต่ละ Feature (Random Forest)",
                  color="importance", color_continuous_scale=["#FFF5EB", "#FFB067"],
                  text="importance")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10),
                       font=THAI_FONT,
                       xaxis_title="ค่าความสำคัญ", yaxis_title="",
                       coloraxis_showscale=False)
    return fig


def per_class_table(metrics):
    best = metrics["best_model"]
    report = metrics["results"][best]["classification_report"]
    rows = []
    for tier in metrics["tier_order"]:
        if tier in report:
            rows.append({
                "ระดับ": TIER_TH[tier],
                "Precision": round(report[tier]["precision"], 3),
                "Recall": round(report[tier]["recall"], 3),
                "F1-Score": round(report[tier]["f1-score"], 3),
                "Support (ตัวอย่าง)": int(report[tier]["support"]),
            })
    df = pd.DataFrame(rows)
    return dbc.Table.from_dataframe(df, striped=True, bordered=True,
                                     hover=True, responsive=True)


def knn_tuning_section(m):
    """Display GridSearchCV results for KNN"""
    if "knn_tuning" not in m:
        return html.Div()
    t = m["knn_tuning"]
    baseline = t["baseline_cv_f1"]
    tuned = t["tuned_cv_f1"]
    improvement = t["improvement"]
    bp = t["best_params"]
    pct_improvement = (improvement / baseline) * 100

    # KPI cards
    kpi_row = dbc.Row([
        dbc.Col(indicator_card("Baseline KNN (k=5)",
                                f"{baseline:.3f}", color="secondary"), md=3),
        dbc.Col(indicator_card("Tuned KNN",
                                f"{tuned:.3f} ± {t['tuned_cv_f1_std']:.3f}",
                                color="success"), md=3),
        dbc.Col(indicator_card("Improvement",
                                f"{improvement:+.3f}",
                                color="success" if improvement > 0 else "danger"),
                 md=3),
        dbc.Col(indicator_card("% Gain",
                                f"{pct_improvement:+.1f}%",
                                color="success" if improvement > 0 else "danger"),
                 md=3),
    ], className="mb-3")

    # Best params table
    params_row = dbc.Row([
        dbc.Col([
            html.H6("⚙️ Best Hyperparameters"),
            dbc.Table([
                html.Tbody([
                    html.Tr([html.Td(html.Strong("n_neighbors")),
                              html.Td(str(bp.get("clf__n_neighbors", "?")))]),
                    html.Tr([html.Td(html.Strong("weights")),
                              html.Td(str(bp.get("clf__weights", "?")))]),
                    html.Tr([html.Td(html.Strong("metric")),
                              html.Td(str(bp.get("clf__metric", "?")))]),
                ])
            ], bordered=True, striped=True, hover=True, size="sm"),
        ], md=4),
        dbc.Col([
            html.H6("🔍 Search Space"),
            html.Div([
                html.Strong("n_neighbors: "),
                html.Code(str(t["search_space"]["n_neighbors"])),
                html.Br(),
                html.Strong("weights: "),
                html.Code(str(t["search_space"]["weights"])),
                html.Br(),
                html.Strong("metric: "),
                html.Code(str(t["search_space"]["metric"])),
                html.Br(),
                html.Small(f"Total: "
                            f"{len(t['search_space']['n_neighbors']) * len(t['search_space']['weights']) * len(t['search_space']['metric'])} "
                            "combinations × 5-fold CV",
                            className="text-muted"),
            ]),
        ], md=8),
    ], className="mb-3")

    # Top 5 candidates chart
    top5 = t["top5_candidates"]
    labels = []
    scores = []
    stds = []
    for c in top5:
        p = c["params"]
        label = (f"k={p.get('clf__n_neighbors','?')}, "
                  f"{p.get('clf__weights','?')[:4]}, "
                  f"{p.get('clf__metric','?')[:4]}")
        labels.append(label)
        scores.append(c["mean_test_score"])
        stds.append(c["std_test_score"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=scores,
                          error_y=dict(type="data", array=stds),
                          marker_color="#6CB4EE",
                          text=[f"{s:.3f}" for s in scores],
                          textposition="outside"))
    fig.add_hline(y=baseline, line_dash="dash", line_color="red",
                   annotation_text=f"Baseline = {baseline:.3f}",
                   annotation_position="right")
    fig.update_layout(title="Top-5 ผู้สมัครจาก GridSearchCV (CV F1 ± std)",
                       yaxis_title="CV F1 Score",
                       xaxis_title="Hyperparameter combination",
                       height=400, font=THAI_FONT,
                       margin=dict(l=10, r=10, t=50, b=10))

    return html.Div([
        kpi_row,
        params_row,
        dcc.Graph(figure=fig),
        dbc.Alert([
            html.Strong("ข้อสรุป: "),
            f"GridSearchCV ปรับ hyperparameter ของ KNN จาก baseline (k=5) "
            f"เป็น k={bp.get('clf__n_neighbors')}, "
            f"weights='{bp.get('clf__weights')}', "
            f"metric='{bp.get('clf__metric')}' "
            f"ทำให้ CV F1 เพิ่มจาก {baseline:.3f} เป็น {tuned:.3f} "
            f"({pct_improvement:+.1f}%)" if improvement > 0 else
            "GridSearchCV ไม่พบ hyperparameter ที่ดีกว่า baseline",
        ], color="success" if improvement > 0 else "warning"),
    ])


def layout():
    m = load_metrics()
    best = m["best_model"]
    best_metrics = m["results"][best]
    return dbc.Container([
        html.H2("Supervised Learning — ทำนายระดับความเข้าใจแบรนด์"),
        html.P([
            "งาน Multi-class Classification: ทำนาย ",
            html.Code("awareness_tier"),
            " (ต่ำ / กลาง / สูง) จากจำนวน ",
            html.Code("brand_correct"),
            " (จับแบรนด์ลูกของ Calvora ถูก จากทั้งหมด 7 แบรนด์) — "
            "Feature ปลอดจากการ leakage โดยไม่ใส่ brand_correct และ decoy_hit"
        ], className="text-muted"),

        dbc.Row([
            dbc.Col(indicator_card("โมเดลที่ดีที่สุด", best, "success"), md=3),
            dbc.Col(indicator_card(
                "CV F1 (5-fold)",
                f"{best_metrics['cv_f1_mean']:.3f} ± {best_metrics['cv_f1_std']:.3f}"),
                md=3),
            dbc.Col(indicator_card(
                "Test Accuracy", f"{best_metrics['test_accuracy']:.1%}"), md=3),
            dbc.Col(indicator_card(
                "Test F1 (weighted)", f"{best_metrics['test_f1']:.3f}"), md=3),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=model_comparison_chart()), md=12),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=confusion_heatmap(m)), md=6),
            dbc.Col(dcc.Graph(figure=feature_importance_chart()), md=6),
        ], className="mb-3"),

        html.H4(f"ประสิทธิภาพรายระดับ — {best}"),
        per_class_table(m),

        # === KNN GridSearchCV Tuning Section ===
        html.Hr(),
        html.H4("🔧 GridSearchCV — Hyperparameter Tuning ของ KNN",
                className="mt-4"),
        knn_tuning_section(m),

        html.Hr(),
        html.H4("เหตุผลการเลือกโมเดล (Model Justification)"),
        dbc.Alert([
            html.Strong(f"เลือก: {best}. "),
            f"Cross-val F1 = {best_metrics['cv_f1_mean']:.3f} ± "
            f"{best_metrics['cv_f1_std']:.3f}. ",
            "ด้วยตัวอย่างเพียง 124 คน และ 3 ระดับที่ไม่สมดุล "
            "(ต่ำ: 78, กลาง: 21, สูง: 25) — F1 แบบ weighted จาก stratified "
            "5-fold CV เป็น metric ที่น่าเชื่อถือที่สุด ทุกโมเดลใช้ "
            "class_weight='balanced' เพื่อลดอคติจากคลาสที่ไม่สมดุล"
        ], color="info"),
    ], fluid=True)
