"""หน้าทำนายผล — ผู้ใช้กรอกข้อมูล → ใช้โมเดล clustering + supervised → แสดงผล"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc

THAI_FONT = dict(family="Mali, Itim, Quicksand, sans-serif")
BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"

# Load models once at import
CLUSTERING = joblib.load(MODELS / "clustering.joblib")
SUPERVISED = joblib.load(MODELS / "supervised_best.joblib")

# Brand attribution mapping (จาก data_prep.py)
CALVORA_REAL = {"เอบินาริ", "เอบินาริ X", "แบ็กซ์",
                "บิบิป๊อป", "Jomona", "Veggie Snap", "ฟรูทร่า"}
CALVORA_DECOY = {"ฮานาโร", "พัฟโมริ", "สแน็คแบ๊ค"}
ALL_BRANDS = ["คาลโวร่า", "เอบินาริ", "เอบินาริ X", "ฮานาโร", "แบ็กซ์",
              "พัฟโมริ", "Jomona", "บิบิป๊อป", "สแน็คแบ๊ค",
              "Veggie Snap", "ฟรูทร่า"]

PERSONA_TH = {
    "Brand Advocate": "⭐ ผู้สนับสนุนแบรนด์",
    "Partial Connect": "💭 เชื่อมโยงบางส่วน",
    "Aware but Lost": "🤔 รู้จักแต่สับสน",
    "Disconnected": "👤 ไม่เชื่อมโยง",
}
PERSONA_COLOR = {
    "Brand Advocate": "#88D66C",
    "Partial Connect": "#6CB4EE",
    "Aware but Lost": "#FFD93D",
    "Disconnected": "#FF8E8E",
}
PERSONA_STRATEGY = {
    "Brand Advocate": ("เป็นทูตแบรนด์! กระตุ้นด้วย Loyalty Program, "
                        "Exclusive Preview รสชาติใหม่, Referral Reward"),
    "Partial Connect": ("รู้จักแบรนด์แต่ลองสินค้าน้อย — ดันด้วย "
                         "Portfolio Bundling + Discovery Sampling"),
    "Aware but Lost": ("ทานเยอะแต่ไม่เชื่อมโยง Calvora — Co-branded Packaging "
                        "'Powered by Calvora' + Corporate Brand Education"),
    "Disconnected": ("ยังไม่เห็นแบรนด์เลย — Mass Awareness + Trade Marketing "
                      "+ In-store Sampling"),
}
TIER_TH = {"Low": "ระดับต่ำ", "Mid": "ระดับกลาง", "High": "ระดับสูง"}
TIER_COLOR = {"Low": "#FF8E8E", "Mid": "#FFD93D", "High": "#88D66C"}


def tagline_alignment_score(text):
    """Replicate data_prep.tagline_alignment keyword scoring"""
    if not text or len(str(text).strip()) < 3:
        return 0.0
    t = str(text).lower()
    keywords = ["ธรรมชาติ", "เก็บเกี่ยว", "พลัง", "natural", "harvest",
                "power", "ออร์แกนิก", "สดใหม่", "วัตถุดิบ"]
    hits = sum(1 for k in keywords if k in t)
    return min(1.0, hits / 3.0)


# === FORM COMPONENTS ============================================

def _section(title, icon, children):
    return dbc.Card([
        dbc.CardHeader(html.H5(f"{icon}  {title}", className="mb-0")),
        dbc.CardBody(children),
    ], className="mb-3 shadow-sm")


def _slider(id_, label, mn, mx, value, step=1, marks=None):
    return html.Div([
        dbc.Label(label, className="small mb-1"),
        dcc.Slider(min=mn, max=mx, value=value, step=step,
                    id=id_, marks=marks or {i: str(i) for i in range(mn, mx+1)}),
    ], className="mb-3")


def form():
    return dbc.Container([
        dbc.Alert([
            html.Strong("ระบบทำนาย: "),
            "กรอกข้อมูลผู้บริโภคแล้วระบบจะทำนายว่าเป็นกลุ่มไหน + ระดับความเข้าใจแบรนด์ "
            "พร้อมข้อเสนอแนะแคมเปญที่เหมาะสม"
        ], color="info", className="mb-3"),

        # === Demographics ===
        _section("ข้อมูลส่วนตัว", "👤", [
            dbc.Row([
                dbc.Col([
                    dbc.Label("ช่วงอายุ"),
                    dcc.Dropdown(id="in-age", value=2,
                                  options=[{"label": "ต่ำกว่า 20 ปี", "value": 1},
                                           {"label": "20-29 ปี", "value": 2},
                                           {"label": "30-39 ปี", "value": 3},
                                           {"label": "40-49 ปี", "value": 4},
                                           {"label": "50 ปีขึ้นไป", "value": 5}]),
                ], md=6),
                dbc.Col([
                    dbc.Label("เพศ"),
                    dbc.RadioItems(id="in-gender", value="หญิง", inline=True,
                                    options=[{"label": " ชาย", "value": "ชาย"},
                                             {"label": " หญิง", "value": "หญิง"},
                                             {"label": " LGBTQ+", "value": "LGBTQ+"}]),
                ], md=6),
            ]),
        ]),

        # === Brand Attribution (Calvora portfolio quiz) ===
        _section("คำถาม: คุณคิดว่าแบรนด์ใดต่อไปนี้เป็นของ Calvora?",
                  "🛍️", [
            html.P("เลือกเฉพาะแบรนด์ที่คุณ คิดว่า อยู่ในเครือ Calvora",
                    className="small text-muted"),
            dbc.Checklist(
                id="in-attribution",
                options=[{"label": f" {b}", "value": b} for b in ALL_BRANDS],
                value=["คาลโวร่า"],  # everyone says yes to parent
                inline=False, switch=False,
            ),
        ]),

        # === Product Recognition & Trial ===
        _section("การรู้จัก / ทดลองสินค้า", "🍽️", [
            dbc.Row([
                dbc.Col([
                    dbc.Label("แบรนด์ลูกที่ คุณรู้จัก"),
                    dbc.Checklist(id="in-recog",
                                   options=[{"label": f" {b}", "value": b}
                                            for b in ALL_BRANDS],
                                   value=[]),
                ], md=6),
                dbc.Col([
                    dbc.Label("แบรนด์ลูกที่ คุณเคยทาน"),
                    dbc.Checklist(id="in-tried",
                                   options=[{"label": f" {b}", "value": b}
                                            for b in ALL_BRANDS],
                                   value=[]),
                ], md=6),
            ]),
        ]),

        # === Tagline ===
        _section("Tagline ของ Calvora", "📢", [
            dbc.Label('Tagline ของ Calvora คือ "Harvest the Power of Nature"'),
            html.Br(),
            dbc.Label("คุณเคยรู้จัก tagline นี้ไหม?"),
            dbc.RadioItems(id="in-tagline-known", value=0, inline=True,
                            options=[{"label": " รู้", "value": 1},
                                     {"label": " ไม่รู้", "value": 0}],
                            className="mb-3"),
            _slider("in-tagline-rating",
                     "ถ้ารู้จัก: tagline สะท้อนภาพแบรนด์ระดับใด (1-5)",
                     1, 5, 3),
            dbc.Label("ลองตีความ tagline นี้ในความคิดของคุณ (เช่น 'เก็บเกี่ยวพลังจากธรรมชาติ')"),
            dbc.Input(id="in-tagline-text", type="text",
                       placeholder="พิมพ์การตีความของคุณ (ไม่จำเป็น)",
                       className="mb-2"),
            html.Small("ระบบจะวิเคราะห์ keyword จากข้อความนี้",
                        className="text-muted"),
        ]),

        # === Ebisen ===
        _section("ความคุ้นเคยกับ Ebisen (เอบินาริ)", "🍤", [
            dbc.RadioItems(id="in-ebisen", value=1, inline=True,
                            options=[{"label": " เคยทาน", "value": 3},
                                     {"label": " รู้จัก แต่ไม่เคยทาน", "value": 2},
                                     {"label": " ไม่รู้จักเลย", "value": 1}]),
        ]),

        # === Purchase Factors (Likert) ===
        _section("ปัจจัยการตัดสินใจซื้อขนม (1 = ไม่มีผล, 5 = มีผลมาก)",
                  "⭐", [
            _slider("in-score-ingredient", "วัตถุดิบคุณภาพ", 1, 5, 4),
            _slider("in-score-taste", "รสชาติอร่อย", 1, 5, 5),
            _slider("in-score-variety", "ความหลากหลายของรสชาติ", 1, 5, 3),
            _slider("in-score-texture", "สัมผัสกรุบกรอบ เคี้ยวเพลิน", 1, 5, 4),
            _slider("in-score-health", "เพื่อสุขภาพดี", 1, 5, 3),
        ]),

        # Submit
        html.Div([
            dbc.Button("ทำนายผล", id="btn-predict",
                        color="primary", size="lg",
                        className="w-100 mb-4"),
        ]),

        # Results placeholder
        html.Div(id="prediction-results"),
    ], fluid=True)


# === PREDICTION CALLBACK =========================================

@callback(
    Output("prediction-results", "children"),
    Input("btn-predict", "n_clicks"),
    State("in-age", "value"),
    State("in-gender", "value"),
    State("in-attribution", "value"),
    State("in-recog", "value"),
    State("in-tried", "value"),
    State("in-tagline-known", "value"),
    State("in-tagline-rating", "value"),
    State("in-tagline-text", "value"),
    State("in-ebisen", "value"),
    State("in-score-ingredient", "value"),
    State("in-score-taste", "value"),
    State("in-score-variety", "value"),
    State("in-score-texture", "value"),
    State("in-score-health", "value"),
    prevent_initial_call=True,
)
def predict(n_clicks, age, gender, attribution, recog, tried,
             tagline_known, tagline_rating, tagline_text, ebisen,
             s_ing, s_taste, s_var, s_tex, s_health):
    if not n_clicks:
        return no_update

    attribution = attribution or []
    recog = recog or []
    tried = tried or []

    # === Compute derived features ===
    brand_correct = sum(1 for b in attribution if b in CALVORA_REAL)
    brand_decoy_hit = sum(1 for b in attribution if b in CALVORA_DECOY)
    products_recog_count = len(recog)
    products_tried_count = len(tried)
    tagline_alignment = tagline_alignment_score(tagline_text)
    if not tagline_known:
        tagline_rating = 0  # rule: people who don't know shouldn't rate

    # === Build feature vector for SUPERVISED model ===
    sup_features = SUPERVISED["features"]
    sup_input = {
        "tagline_known": tagline_known,
        "tagline_alignment": tagline_alignment,
        "tagline_rating": tagline_rating,
        "products_recognized_count": products_recog_count,
        "products_tried_count": products_tried_count,
        "score_ingredient": s_ing,
        "score_taste": s_taste,
        "score_variety": s_var,
        "score_texture": s_tex,
        "score_health": s_health,
        "ebisen_familiarity": ebisen,
        "age": age,
        "gender_ชาย": 1 if gender == "ชาย" else 0,
        "gender_หญิง": 1 if gender == "หญิง" else 0,
        "gender_LGBTQ+": 1 if gender == "LGBTQ+" else 0,
    }
    X_sup = np.array([[sup_input.get(f, 0) for f in sup_features]])

    # Predict awareness_tier
    sup_model = SUPERVISED["model"]
    le = SUPERVISED["label_encoder"]
    tier_pred_idx = sup_model.predict(X_sup)[0]
    tier_pred = le.inverse_transform([tier_pred_idx])[0]
    try:
        proba = sup_model.predict_proba(X_sup)[0]
        proba_dict = {le.inverse_transform([i])[0]: float(p)
                       for i, p in enumerate(proba)}
    except Exception:
        proba_dict = None

    # === Build feature vector for CLUSTERING model ===
    clu_features = CLUSTERING["features"]
    clu_input = {
        "brand_correct": brand_correct,
        "brand_decoy_hit": brand_decoy_hit,
        "products_recognized_count": products_recog_count,
        "products_tried_count": products_tried_count,
        "tagline_alignment": tagline_alignment,
        "tagline_known": tagline_known,
        "score_ingredient": s_ing,
        "score_taste": s_taste,
        "score_health": s_health,
        "ebisen_familiarity": ebisen,
        "age": age,
    }
    X_clu = np.array([[clu_input.get(f, 0) for f in clu_features]])
    X_scaled = CLUSTERING["scaler"].transform(X_clu)
    cluster_id = int(CLUSTERING["kmeans"].predict(X_scaled)[0])
    persona = CLUSTERING["persona_map"][cluster_id]
    pca_coord = CLUSTERING["pca"].transform(X_scaled)[0]

    # === Build results UI ===
    return _render_results(
        tier_pred, proba_dict, persona, pca_coord,
        brand_correct, brand_decoy_hit, products_recog_count,
        products_tried_count, tagline_alignment
    )


def _render_results(tier, proba, persona, pca_coord,
                     bc, dh, recog_n, tried_n, t_align):
    # Persona card
    persona_card = dbc.Card([
        dbc.CardHeader(html.H4("🎯 กลุ่มลูกค้าที่ทำนาย (Persona)",
                                 className="mb-0")),
        dbc.CardBody([
            html.Div([
                html.Span(PERSONA_TH[persona], className="display-6",
                           style={"color": PERSONA_COLOR[persona],
                                  "fontWeight": "bold"}),
            ], className="text-center mb-3"),
            html.H6("📋 กลยุทธ์การตลาดที่แนะนำ:"),
            html.P(PERSONA_STRATEGY[persona]),
        ]),
    ], className="mb-3 shadow")

    # Tier card with probability bar
    tier_color = TIER_COLOR.get(tier, "#888")
    proba_bars = []
    if proba:
        for t in ["Low", "Mid", "High"]:
            p = proba.get(t, 0)
            proba_bars.append(html.Div([
                html.Div([
                    html.Span(TIER_TH[t], className="small me-2"),
                    html.Span(f"{p*100:.0f}%",
                                className="small text-muted"),
                ], className="d-flex justify-content-between"),
                dbc.Progress(value=p*100, color={"Low": "danger",
                                                   "Mid": "warning",
                                                   "High": "success"}[t],
                              className="mb-2", style={"height": "20px"}),
            ]))

    tier_card = dbc.Card([
        dbc.CardHeader(html.H4("📊 ระดับความเข้าใจแบรนด์ (Awareness Tier)",
                                 className="mb-0")),
        dbc.CardBody([
            html.Div([
                html.Span(TIER_TH[tier], className="display-6",
                           style={"color": tier_color, "fontWeight": "bold"}),
            ], className="text-center mb-3"),
            html.Div(proba_bars) if proba_bars else html.Div(),
        ]),
    ], className="mb-3 shadow")

    # Stats summary
    stats_card = dbc.Card([
        dbc.CardHeader(html.H4("📈 สรุปคะแนน", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(_kpi("จับถูก (จาก 7)", f"{bc}/7"), md=3),
                dbc.Col(_kpi("สับสนคู่แข่ง", f"{dh}/3",
                              color="danger" if dh > 1 else "success"), md=3),
                dbc.Col(_kpi("รู้จัก (จาก 11)", f"{recog_n}/11"), md=3),
                dbc.Col(_kpi("เคยทาน (จาก 11)", f"{tried_n}/11"), md=3),
            ]),
            html.Hr(),
            html.Div([
                dbc.Label("คะแนนความเข้าใจ Tagline"),
                dbc.Progress(value=t_align*100,
                              label=f"{t_align*100:.0f}%",
                              color="info"),
            ]),
        ]),
    ], className="mb-3 shadow")

    # PCA scatter — show user's position
    pca_fig = _make_pca_chart(pca_coord, persona)

    return html.Div([
        html.Hr(),
        html.H3("✅ ผลการทำนาย", className="mb-3"),
        dbc.Row([
            dbc.Col(persona_card, md=6),
            dbc.Col(tier_card, md=6),
        ]),
        stats_card,
        dbc.Card([
            dbc.CardHeader(html.H4("📍 ตำแหน่งของคุณบน PCA Map",
                                     className="mb-0")),
            dbc.CardBody(dcc.Graph(figure=pca_fig)),
        ], className="mb-3 shadow"),
    ])


def _kpi(label, value, color="primary"):
    return html.Div([
        html.Div(label, className="text-muted small"),
        html.H4(value, className=f"text-{color}"),
    ], className="text-center")


def _make_pca_chart(user_xy, user_persona):
    """Plot existing 124 respondents + highlight user point"""
    from db import query
    df = query("SELECT pca_x, pca_y, persona FROM survey_responses")
    df["กลุ่ม"] = df["persona"].map({
        "Brand Advocate": "ผู้สนับสนุนแบรนด์",
        "Partial Connect": "เชื่อมโยงบางส่วน",
        "Aware but Lost": "รู้จักแต่สับสน",
        "Disconnected": "ไม่เชื่อมโยง",
    })
    color_map = {PERSONA_TH[k]: v for k, v in PERSONA_COLOR.items()}
    fig = px.scatter(df, x="pca_x", y="pca_y", color="กลุ่ม",
                      color_discrete_map=color_map,
                      title="ตำแหน่งของคุณ (★) เทียบกับผู้ตอบทั้ง 124 คน",
                      opacity=0.45)
    fig.update_traces(marker=dict(size=10))
    # Add user point
    fig.add_trace(go.Scatter(
        x=[user_xy[0]], y=[user_xy[1]], mode="markers+text",
        marker=dict(size=24, symbol="star",
                     color=PERSONA_COLOR[user_persona],
                     line=dict(width=3, color="black")),
        text=["คุณ"], textposition="top center",
        textfont=dict(size=14, color="black"),
        name="ตำแหน่งของคุณ", showlegend=True,
    ))
    fig.update_layout(height=550, margin=dict(l=10, r=10, t=50, b=10),
                       font=THAI_FONT, xaxis_title="PC1", yaxis_title="PC2")
    return fig


def layout():
    return dbc.Container([
        html.H2("ระบบทำนายผล (Predict)"),
        html.P("กรอกข้อมูลลูกค้า — ระบบจะใช้โมเดล Machine Learning "
               "ทำนาย Persona และระดับความเข้าใจแบรนด์ "
               "พร้อมข้อเสนอแนะแคมเปญ", className="text-muted"),
        form(),
    ], fluid=True)
