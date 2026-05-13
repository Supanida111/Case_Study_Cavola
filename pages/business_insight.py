"""หน้า Business Insight — ข้อเสนอแนะเชิงกลยุทธ์"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from db import query

THAI_FONT = dict(family="Mali, Itim, Quicksand, sans-serif")

PERSONA_TH = {
    "Brand Advocate": "ผู้สนับสนุนแบรนด์",
    "Partial Connect": "เชื่อมโยงบางส่วน",
    "Aware but Lost": "รู้จักแต่สับสน",
    "Disconnected": "ไม่เชื่อมโยง",
}


def attribution_gap_chart():
    """% ที่จับแบรนด์ลูกถูก vs พลาด"""
    df = query("SELECT * FROM survey_responses")
    n = len(df)
    counts = {}
    for col in df.columns:
        if col.startswith("recog_"):
            brand = col.replace("recog_", "")
            counts[brand] = int(df[col].sum())
    cdf = pd.DataFrame([{"แบรนด์": k,
                          "รู้จัก (%)": 100 * v / n,
                          "ช่องว่าง (%)": 100 * (n - v) / n}
                         for k, v in counts.items()])
    cdf = cdf.sort_values("รู้จัก (%)", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="รู้จัก", y=cdf["แบรนด์"],
                          x=cdf["รู้จัก (%)"], orientation="h",
                          marker_color="#88D66C"))
    fig.add_trace(go.Bar(name="ช่องว่าง", y=cdf["แบรนด์"],
                          x=cdf["ช่องว่าง (%)"], orientation="h",
                          marker_color="#FF8E8E"))
    fig.update_layout(barmode="stack",
                       title="ช่องว่างการรู้จักแบรนด์ลูก (% ของ 124 คน)",
                       xaxis_title="% ของผู้ตอบ",
                       height=400, margin=dict(l=10, r=10, t=50, b=10),
                       font=THAI_FONT)
    return fig


def ebisen_funnel():
    df = query("SELECT ebisen_familiarity, COUNT(*) as n FROM survey_responses "
                "GROUP BY ebisen_familiarity ORDER BY ebisen_familiarity DESC")
    label_map = {3: "เคยทาน", 2: "รู้จักแต่ไม่เคยทาน",
                 1: "ไม่รู้จัก", 0: "ไม่ระบุ"}
    df["label"] = df["ebisen_familiarity"].map(label_map)
    fig = go.Figure(go.Funnel(
        y=df["label"], x=df["n"],
        marker=dict(color=["#88D66C", "#FFD93D", "#FF8E8E", "#D1D1D1"]),
        textinfo="value+percent total",
    ))
    fig.update_layout(title="Funnel การทดลอง Ebisen",
                       height=400, margin=dict(l=10, r=10, t=50, b=10),
                       font=THAI_FONT)
    return fig


def ebisen_flavors_chart():
    """รสชาติ/สินค้า Ebisen ที่เคยรับประทาน"""
    df = query("SELECT * FROM survey_responses")
    flavor_cols = [c for c in df.columns if c.startswith("report_flavor_")]
    counts = {c.replace("report_flavor_", ""): int(df[c].sum())
              for c in flavor_cols}
    cdf = pd.DataFrame([{"รสชาติ": k, "จำนวน": v} for k, v in counts.items()])
    cdf = cdf.sort_values("จำนวน", ascending=True)
    fig = px.bar(cdf, x="จำนวน", y="รสชาติ", orientation="h",
                  title="รสชาติ/สินค้า Ebisen ที่เคยรับประทาน",
                  color="จำนวน", color_continuous_scale=["#F0F9EE", "#88D66C"],
                  text="จำนวน")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10),
                       font=THAI_FONT,
                       xaxis_title="จำนวนคน (People)",
                       yaxis_title="รายการรสค้า",
                       coloraxis_showscale=False)
    return fig


def trust_factors_chart():
    """ทำไมถึงเชื่อมั่นในวัตถุดิบจากธรรมชาติ"""
    df = query("SELECT * FROM survey_responses")
    trust_cols = [c for c in df.columns
                  if c.startswith("trust_") and c != "trust_signal"]
    counts = {c.replace("trust_", ""): int(df[c].sum()) for c in trust_cols}
    cdf = pd.DataFrame([{"เหตุผลที่เลือก": k, "จำนวน": v}
                          for k, v in counts.items() if v > 0])
    cdf = cdf.sort_values("จำนวน", ascending=True)
    fig = px.bar(cdf, x="จำนวน", y="เหตุผลที่เลือก", orientation="h",
                  title="ทำไมถึงเชื่อมั่นในวัตถุดิบจากธรรมชาติ?",
                  color="จำนวน",
                  color_continuous_scale=["#FFF5EB", "#FFB067"],
                  text="จำนวน")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=50, b=10),
                       font=THAI_FONT,
                       xaxis_title="จำนวนคน (People)",
                       yaxis_title="เหตุผลที่เลือก",
                       coloraxis_showscale=False)
    return fig


def trial_intent_chart():
    df = query("SELECT trial_intent, persona, COUNT(*) as n "
                "FROM survey_responses WHERE trial_intent IS NOT NULL "
                "GROUP BY trial_intent, persona")
    df["intent_label"] = df["trial_intent"].map({1: "อยากลอง", 0: "ไม่อยากลอง"})
    df["กลุ่ม"] = df["persona"].map(PERSONA_TH).fillna(df["persona"])
    fig = px.bar(df, x="กลุ่ม", y="n", color="intent_label",
                  title="ความตั้งใจทดลอง แยกตามกลุ่มลูกค้า",
                  color_discrete_map={"อยากลอง": "#88D66C",
                                       "ไม่อยากลอง": "#FF8E8E"},
                  barmode="group", text="n")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10),
                       font=THAI_FONT,
                       yaxis_title="จำนวนคน",
                       legend_title_text="ความตั้งใจ")
    return fig


def belief_reason_chart():
    """ทำไมถึงเชื่อ/ไม่เชื่อว่า Ebisen ทำจากกุ้งแท้
    Stacked bar: believer vs disbeliever × reason category"""
    df = query("SELECT belief_reason, ebisen_belief, COUNT(*) as n "
                "FROM survey_responses WHERE belief_reason != 'no_response' "
                "GROUP BY belief_reason, ebisen_belief")
    reason_th = {
        "sensory": "รสชาติ / กลิ่น",
        "brand_image": "ภาพลักษณ์แบรนด์ / ญี่ปุ่น",
        "skeptical": "สงสัย (ผง / แต่ง)",
        "label_evidence": "ฉลาก / ส่วนผสม",
        "experience": "ประสบการณ์ตรง",
        "other": "อื่นๆ",
    }
    df["เหตุผล"] = df["belief_reason"].map(reason_th).fillna(df["belief_reason"])
    df["กลุ่ม"] = df["ebisen_belief"].map({1.0: "เชื่อ", 0.0: "ไม่เชื่อ"})
    df = df.dropna(subset=["กลุ่ม"])
    fig = px.bar(df, x="เหตุผล", y="n", color="กลุ่ม",
                  title="เหตุผลที่เชื่อ vs ไม่เชื่อว่า Ebisen ทำจากกุ้งแท้",
                  color_discrete_map={"เชื่อ": "#88D66C", "ไม่เชื่อ": "#FF8E8E"},
                  barmode="stack", text="n")
    fig.update_traces(textposition="inside")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10),
                       font=THAI_FONT, yaxis_title="จำนวนคน",
                       xaxis_title="เหตุผล",
                       legend_title_text="ผู้บริโภค")
    return fig


def tagline_alignment_chart():
    df = query("SELECT tagline_alignment FROM survey_responses")
    fig = px.histogram(df, x="tagline_alignment", nbins=10,
                        title="คะแนนความเข้าใจ Tagline (0–1)",
                        color_discrete_sequence=["#B19CD9"])
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10),
                       font=THAI_FONT,
                       xaxis_title="ระดับการตีความที่ตรงกับเจตนาของแบรนด์",
                       yaxis_title="จำนวนคน")
    return fig


def insight_card(title, text, color):
    return dbc.Card([
        dbc.CardHeader(title, style={"backgroundColor": color, "color": "white"}),
        dbc.CardBody(text),
    ], className="mb-3")


def layout():
    n_aware = query("SELECT COUNT(*) as n FROM survey_responses").iloc[0]["n"]
    tagline_known = query(
        "SELECT SUM(tagline_known) as n FROM survey_responses").iloc[0]["n"]
    intense_yes = query(
        "SELECT COUNT(*) as n FROM survey_responses WHERE trial_intent = 1"
    ).iloc[0]["n"]
    intense_total = query(
        "SELECT COUNT(*) as n FROM survey_responses WHERE trial_intent IS NOT NULL"
    ).iloc[0]["n"]

    return dbc.Container([
        html.H2("ข้อเสนอแนะเชิงธุรกิจ (Business Insight)"),
        html.P("ข้อเสนอแนะเชิงกลยุทธ์และโอกาสทางการตลาด จากผลการแบ่งกลุ่ม "
               "และการทำนายระดับความเข้าใจแบรนด์",
               className="text-muted"),

        dbc.Row([
            dbc.Col(insight_card(
                "ช่องว่างวิกฤต: Tagline",
                f"มีเพียง {tagline_known}/{n_aware} คน "
                f"(≈{100*tagline_known/n_aware:.1f}%) ที่รู้จัก Tagline "
                "'Harvest the Power of Nature' — เป็น Corporate Asset "
                "ที่ใช้งานน้อยที่สุด ควรเสริมการสื่อสาร Tagline ลงบน "
                "บรรจุภัณฑ์และโฆษณาของทุกแบรนด์ลูก",
                "#FF8E8E"), md=4),
            dbc.Col(insight_card(
                "ช่องว่าง: Sub-brand Attribution",
                "แบรนด์ลูกถูกระบุว่าเป็นของ Calvora เพียง 23–44% "
                "เท่านั้น มีเพียง 'คาลโวร่า' เองที่คนรู้ครบถ้วน — "
                "ควรเปิดตัวบรรจุภัณฑ์ co-branded 'Powered by Calvora'",
                "#FFB067"), md=4),
            dbc.Col(insight_card(
                "โอกาส: Trial Conversion",
                f"จากผู้ที่ทาน Ebisen แล้ว มีคนที่อยากลองรสชาติเข้มข้นถึง "
                f"{intense_yes}/{intense_total} คน "
                f"({100*intense_yes/intense_total:.0f}%) — "
                "เป็นโอกาสในการขยาย product line",
                "#88D66C"), md=4),
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=trust_factors_chart()), md=7),
            dbc.Col(dcc.Graph(figure=ebisen_flavors_chart()), md=5),
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=attribution_gap_chart()), md=6),
            dbc.Col(dcc.Graph(figure=ebisen_funnel()), md=6),
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=belief_reason_chart()), md=12),
        ]),
    ], fluid=True)
