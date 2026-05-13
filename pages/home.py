"""หน้าแรก — สรุปกรณีศึกษาและภาพรวมข้อมูล"""
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db import query

THAI_FONT = dict(family="Mali, Itim, Quicksand, sans-serif")
PASTEL_COLORS = ["#88D66C", "#FFB067", "#B19CD9", "#FFD93D", "#6CB4EE"]


def stat_card(title, value, subtitle="", color="primary", icon=""):
    """Large hero-style KPI card"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={"fontSize": "3.5rem"}),
            ], className="text-center mb-2 floating") if icon else html.Div(),
            html.Div(title, className="text-muted text-center",
                      style={"fontSize": "1.1rem", "fontWeight": "600"}),
            html.H1(value, className="text-center my-3 stat-value",
                     style={"fontSize": "3.5rem", "fontWeight": "800"}),
            html.Div(subtitle, className="text-muted text-center",
                      style={"minHeight": "20px", "fontSize": "0.9rem"}),
        ], style={"padding": "2rem 1rem"})
    ], className="shadow h-100 border-0",
       style={"background": "linear-gradient(180deg, #FFFFFF 0%, #F9FFF9 100%)"})


def attribution_chart():
    """% ที่จับแบรนด์ลูกถูกว่าเป็นของ Calvora — แสดงช่องว่างการรับรู้"""
    df = query("SELECT * FROM survey_responses")
    counts = {}
    for col in df.columns:
        if col.startswith("recog_"):
            brand = col.replace("recog_", "")
            counts[brand] = int(df[col].sum())
    cdf = pd.DataFrame([{"แบรนด์": k, "จำนวนคน": v} for k, v in counts.items()])
    cdf = cdf.sort_values("จำนวนคน", ascending=True)
    fig = px.bar(cdf, x="จำนวนคน", y="แบรนด์", orientation="h",
                 title="การรู้จักแบรนด์ลูกในเครือ Calvora (จากผู้ตอบ 124 คน)",
                 color="จำนวนคน", color_continuous_scale=["#FFF5EB", "#FFB067"],
                 text="จำนวนคน")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10),
                      font=THAI_FONT, coloraxis_showscale=False)
    return fig


def tier_dist_chart():
    df = query("SELECT awareness_tier, COUNT(*) as n FROM survey_responses "
                "GROUP BY awareness_tier")
    label_map = {"Low": "ระดับต่ำ", "Mid": "ระดับกลาง", "High": "ระดับสูง"}
    df["label"] = df["awareness_tier"].map(label_map)
    order = {"ระดับต่ำ": 0, "ระดับกลาง": 1, "ระดับสูง": 2}
    df["order"] = df["label"].map(order)
    df = df.sort_values("order")
    fig = px.bar(df, x="label", y="n",
                 title="กลุ่มระดับความเข้าใจแบรนด์ Calvora",
                 color="label",
                 color_discrete_map={"ระดับต่ำ": "#FF8E8E",
                                      "ระดับกลาง": "#FFD93D",
                                      "ระดับสูง": "#88D66C"},
                 text="n")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=350, showlegend=False,
                      margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="", yaxis_title="จำนวนคน",
                      font=THAI_FONT)
    return fig


def understanding_histogram():
    """การกระจายตัวของคะแนนความเข้าใจในแบรนด์ (เต็ม 11 คะแนน)"""
    df = query("SELECT brand_understanding FROM survey_responses")
    fig = px.histogram(
        df, x="brand_understanding", nbins=11,
        title="การกระจายตัวของคะแนนความเข้าใจในแบรนด์ (เต็ม 11 คะแนน)",
        color_discrete_sequence=["#B19CD9"],
    )
    fig.update_traces(marker_line_color="#967BB6", marker_line_width=1)
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="คะแนนที่ได้รับ",
                      yaxis_title="จำนวนคน",
                      bargap=0.05,
                      font=THAI_FONT)
    return fig


def decoy_confusion_chart():
    """แบรนด์คู่แข่งที่คนสับสนว่าเป็น Calvora มากที่สุด"""
    df = pd.read_csv(Path(__file__).resolve().parent.parent /
                     "data" / "decoy_per_brand.csv")
    df = df.sort_values("wrong_count", ascending=False)
    fig = px.bar(df, x="wrong_count", y="brand", orientation="h",
                 title="แบรนด์คู่แข่งที่คนสับสนว่าเป็น Calvora มากที่สุด",
                 color="wrong_count",
                 color_continuous_scale=["#FFF0F0", "#FF8E8E"],
                 text="wrong_count")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="จำนวนคนที่ตอบผิด (คน)",
                      yaxis_title="ชื่อแบรนด์คู่แข่ง",
                      font=THAI_FONT, coloraxis_showscale=False,
                      yaxis=dict(autorange="reversed"))
    return fig


def layout():
    return dbc.Container([
        html.H2("กรณีศึกษา: ช่องว่างการรับรู้แบรนด์ Calvora (Corporate Brand Gap)",
                className="mb-3"),
        dbc.Alert([
            html.Strong("โจทย์: "),
            "บริษัท Calvora (คาลโวร่า) ผู้ผลิตขนมรายใหญ่จากญี่ปุ่น "
            "มีพอร์ตสินค้าหลายแบรนด์ในประเทศไทยที่ประสบความสำเร็จและเป็นที่รู้จักในวงกว้าง "
            "แต่กำลังเผชิญกับช่องว่างสำคัญในด้านการรับรู้ของผู้บริโภค (Consumer Perception Gap) — "
            "แต่ละแบรนด์ภายใต้บริษัทมีความแข็งแกร่งในตัวเอง แต่ผู้บริโภคส่วนใหญ่ "
            "ไม่เชื่อมโยงว่าอยู่ภายใต้บริษัทเดียวกัน  เป้าหมายคือยกระดับการสื่อสารจาก "
            "Product Brand → Corporate Brand"
        ], color="warning"),

        dbc.Row([
            dbc.Col(stat_card("ผู้ตอบแบบสอบถามทั้งหมด", "141",
                                "คน (ทั้งหมดในแบบสอบถาม)",
                                color="primary", icon="👥"), md=3),
            dbc.Col(stat_card("รู้จักแบรนด์ Calvora", "124",
                                "คน (88% ของผู้ตอบ)",
                                color="success", icon="✓"), md=3),
            dbc.Col(stat_card("รู้จัก Tagline", "3",
                                "คน (2.4% — ช่องว่างวิกฤต)",
                                color="danger", icon="⚠️"), md=3),
            dbc.Col(stat_card("แบรนด์ลูกในพอร์ตโฟลิโอ", "7",
                                "แบรนด์ ตามข้อมูลจริงจาก Sheet2",
                                color="warning", icon="🛍️"), md=3),
        ], className="mb-5"),

        html.Hr(),
        html.H4("ภาพรวม ML Pipeline"),
        html.Pre("""
ข้อมูลดิบจากแบบสอบถาม → Data Prep (NLP + Encoding) → ข้อมูลสะอาด (124 แถว × 65 คอลัมน์)
        ↓                                                ↓
SQLite Database ←──────────── Unsupervised (K-Means + Hierarchical, k=4)
        ↑                                                ↓
Dashboard ←──── Supervised (5 โมเดล, target = ระดับความเข้าใจแบรนด์)
        """, className="bg-light p-3 small"),
    ], fluid=True)
