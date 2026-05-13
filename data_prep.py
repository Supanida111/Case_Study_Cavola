"""
Calvora — Data Preparation
Reads snack.xlsx, encodes Likert/binary, parses multi-select, runs lightweight Thai NLP,
and produces calvora_clean.csv (124 aware respondents)
"""
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import re as _re

BASE = Path(__file__).parent
RAW = BASE / "data" / "snack.xlsx"
CLEAN = BASE / "data" / "calvora_clean.csv"

# Sheet2 ground truth: 7 real Calvora brands
CALVORA_BRANDS = {"แบ็กซ์", "บิบิป๊อป", "Jomona",
                  "เอบินาริ", "Veggie Snap", "ฟรูทร่า", "เอบินาริ X"}

LIKERT_MAP = {
    "ไม่มีผลเลย": 1, "ไม่มีผล": 2, "มีผลน้อย": 2,
    "ปานกลาง": 3, "มีผลมาก": 4, "มีผลมากที่สุด": 5,
}

AGE_MAP = {"ต่ำกว่า 20ปี": 1, "20-29ปี": 2, "30-39ปี": 3, "40-49ปี": 4, "50ปี ขึ้นไป": 5}


def load_raw():
    df = pd.read_excel(RAW, sheet_name="Sheet1", header=1)
    df.columns = [c.strip() for c in df.columns]
    return df


def encode_likert(df, idx_cols):
    out = {}
    for label, idx in idx_cols.items():
        out[label] = df.iloc[:, idx].map(LIKERT_MAP)
    return pd.DataFrame(out)


def parse_multiselect(series, all_options):
    """Convert comma-separated string column into binary dummies."""
    out = pd.DataFrame(index=series.index)
    for opt in all_options:
        out[f"has_{opt}"] = series.fillna("").astype(str).apply(
            lambda s: 1 if opt in s else 0
        )
    out["count"] = out.sum(axis=1)
    return out


def attribution_features(df):
    """Cols 12-22 are Yes/No attribution to Calvora for 11 brands."""
    attr_cols = list(df.columns[12:23])
    real_idx, decoy_idx = [], []
    for c in attr_cols:
        # Extract brand name in [brackets]
        m = re.search(r"\[(.+?)\]", c)
        brand = m.group(1).strip() if m else c
        is_real = any(rb.lower() in brand.lower() or brand.lower() in rb.lower()
                      for rb in CALVORA_BRANDS)
        # Trivially exclude คาลโวร่า itself (parent name = always yes)
        if brand == "คาลโวร่า":
            continue
        if is_real:
            real_idx.append(c)
        else:
            decoy_idx.append(c)

    yes = "ใช่ เป็นของ Calvora"
    real_correct = df[real_idx].apply(lambda col: (col == yes).astype(int)).sum(axis=1)
    decoy_hit = df[decoy_idx].apply(lambda col: (col == yes).astype(int)).sum(axis=1)
    return real_correct, decoy_hit, real_idx, decoy_idx


EATING_OCCASION_THEMES = {
    # Multi-label: 1 คำตอบ match ได้หลาย theme เช่น "ทำงาน กับเบียร์" → work + leisure
    "work_study": ["ทำงาน", "ออฟฟิศ", "เรียน", "work", "ประชุม"],
    "leisure":    ["ดู", "อนิเมะ", "หนัง", "เกม", "netflix", "chill",
                    "พัก", "สังสรรค์", "เบียร์", "relax"],
    "late_night": ["ก่อนนอน", "กลางคืน", "ดึก", "หิว", "โหย"],
    "between_meals": ["บ่าย", "เช้า", "ว่าง", "หลังอาหาร", "มื้อ", "ระหว่างมื้อ"],
}


def eating_occasion_flags(text):
    """Multi-label classifier — returns dict of binary flags (each theme 0/1).
    1 คำตอบสามารถ match หลาย theme ได้พร้อมกัน เช่น 'ทำงาน กับเบียร์'
    จะ match occ_work_study + occ_leisure ทั้งคู่
    ถ้าไม่ match อะไรเลย → occ_other = 1"""
    flags = {f"occ_{k}": 0 for k in EATING_OCCASION_THEMES}
    flags["occ_other"] = 0
    if pd.isna(text):
        flags["occ_other"] = 1
        return flags
    t = str(text).lower()
    matched_any = False
    for theme, keywords in EATING_OCCASION_THEMES.items():
        for kw in keywords:
            if kw.lower() in t:
                flags[f"occ_{theme}"] = 1
                matched_any = True
                break
    if not matched_any:
        flags["occ_other"] = 1
    return flags


TOP_OF_MIND_THEMES = {
    # 1 คำตอบสามารถ match ได้หลาย theme — multi-label binary flags
    "subbrand": ["ฮานาโร", "แบ็กซ์", "แบ๊กซ์", "แบ็ก", "แบ๊ก",
                 "bibipop", "บิบิป๊อป", "jomona", "โจโมน่า",
                 "เอบินาริ", "veggie", "ฟรูทร่า", "พัฟโมริ", "สแน็ค"],
    "parent_brand": ["คาลโวร่า", "calvora"],
    "shrimp_cracker": ["ข้าวเกรียบ", "ข้าวเกียบ", "ข้าวเกรีบ",
                        "ข้าวเกนียบ", "ข้างเกรียบ"],
    "shrimp": ["กุ้ง"],
    "potato": ["มันฝรั่ง", "มันแท่ง", "มัน ", "potato"],
    "snack_generic": ["ขนม", "snack", "ขบเคี้ยว"],
    "japan": ["ญี่ปุ่น", "japan"],
    "natural_health": ["ธรรมชาติ", "natural", "สุขภาพ", "ออร์แกนิก"],
    "positive_attr": ["อร่อย", "กรอบ", "เพลิน", "คุณภาพ", "รสชาติดี"],
    "visual_packaging": ["สีแดง", "ถุง", "ตัวหนังสือ", "โลโก้"],
}


def extract_top_of_mind_flags(text):
    """Multi-label classifier — returns dict of binary flags (each theme 0/1).
    1 คำตอบสามารถ match หลาย theme ได้พร้อมกัน เช่น 'ขนมกรอบจากญี่ปุ่น'
    จะ match snack_generic + japan + positive_attr ทั้งหมด"""
    flags = {f"topmind_{k}": 0 for k in TOP_OF_MIND_THEMES}
    if pd.isna(text):
        return flags
    t = str(text).lower()
    for theme, keywords in TOP_OF_MIND_THEMES.items():
        for kw in keywords:
            if kw.lower() in t:
                flags[f"topmind_{theme}"] = 1
                break
    return flags


def tagline_alignment(text):
    """Col 25: how close is the interpretation to 'Harvest the Power of Nature'?
    Lightweight keyword scoring (0-1)."""
    if pd.isna(text):
        return 0.0
    t = str(text).lower()
    keywords = ["ธรรมชาติ", "เก็บเกี่ยว", "พลัง", "natural", "harvest", "power", "ออร์แกนิก",
                "สดใหม่", "วัตถุดิบ"]
    hits = sum(1 for k in keywords if k in t)
    if "-" == t.strip() or len(t) < 3:
        return 0.0
    # Normalize 0-1 with diminishing returns
    return min(1.0, hits / 3.0)


REASON_KEYWORDS = {
    # คำตอบ "ไม่เชื่อ"/"ไม่จริง" คือ belief เอง (col 32/37) ไม่ใช่ "เหตุผล"
    # → ลบออกจาก keywords ของ skeptical
    "skeptical":      ["แต่ง", "ผง", "แป้ง", "เคลม", "ปรุงแต่ง", "claim",
                        "ไม่ใช่กุ้งจริง", "สังเคราะห์", "ฟอก", "เคมี"],
    "sensory":        ["รสชาติ", "กลิ่น", "อร่อย", "เค็ม", "สัมผัส", "หวาน"],
    "brand_image":    ["ยี่ห้อ", "แบรนด์", "ญี่ปุ่น", "โลโก้", "ภาพ",
                        "เชื่อมั่น", "trust", "ภาพลักษณ์"],
    "label_evidence": ["ส่วนผสม", "ส่วนประกอบ", "ฉลาก", "วัตถุดิบ",
                        "หลังซอง", "ingredients", "ระบุ", "เขียน"],
    "experience":     ["เคยทาน", "เคยกิน", "เคยลอง", "เคยซื้อ", "ทานบ่อย"],
}


def classify_belief_reason(text):
    """Cols 33/38: จัดหมวด 'เหตุผล' ว่าทำไมเชื่อ/ไม่เชื่อ
    (ไม่ใช่จัดหมวด 'เชื่อ vs ไม่เชื่อ' — อันนั้นเก็บใน ebisen_belief แยกแล้ว)"""
    if pd.isna(text):
        return "no_response"
    t = str(text).lower()
    for label, keywords in REASON_KEYWORDS.items():
        if any(k in t for k in keywords):
            return label
    return "other"


def main():
    print("Loading raw data...")
    df = load_raw()
    print(f"  shape: {df.shape}")

    # === Awareness gate ===
    # Drop the 17 "never heard of Calvora" respondents — they only answered
    # demographics and provide no signal for brand-perception modeling.
    aware_col = df.columns[1]
    df["aware"] = (df[aware_col] == "เคย").astype(int)
    aware = df[df["aware"] == 1].copy().reset_index(drop=True)
    print(f"  aware (kept): {len(aware)} | dropped non-aware: "
          f"{(df['aware'] == 0).sum()}")

    # === Likert purchase factors (cols 6-10) ===
    print("Encoding Likert purchase factors...")
    likert = encode_likert(aware, {
        "score_ingredient": 6, "score_taste": 7, "score_variety": 8,
        "score_texture": 9, "score_health": 10,
    })

    # === Brand attribution (cols 12-22) ===
    print("Computing brand attribution features...")
    brand_correct, decoy_hit, real_cols, decoy_cols = attribution_features(aware)
    print(f"  real Calvora cols: {len(real_cols)}, decoy cols: {len(decoy_cols)}")
    print(f"  brand_correct range: {brand_correct.min()} - {brand_correct.max()}")
    print(f"  decoy_hit range: {decoy_hit.min()} - {decoy_hit.max()}")

    # Per-decoy hit counts (for "which decoy is most confused as Calvora?")
    decoy_per_brand = {}
    yes = "ใช่ เป็นของ Calvora"
    for c in decoy_cols:
        m = _re.search(r"\[(.+?)\]", c)
        brand = m.group(1).strip() if m else c
        decoy_per_brand[brand] = int((aware[c] == yes).sum())
    print(f"  decoy hits per brand: {decoy_per_brand}")

    # Brand understanding score (out of 11) — total correct answers across all 11 attribution Qs
    # = 1 (trivial Calvora) + brand_correct + (n_decoys - decoy_hit)
    n_decoys = len(decoy_cols)
    brand_understanding = 1 + brand_correct + (n_decoys - decoy_hit)
    print(f"  brand_understanding range: {brand_understanding.min()} - {brand_understanding.max()} (max = {1+7+n_decoys})")

    # === Multi-select: products recognized & tried ===
    print("Parsing multi-select brand recognition...")
    all_brands = ["คาลโวร่า", "เอบินาริ", "เอบินาริ X", "ฮานาโร", "แบ็กซ์",
                  "พัฟโมริ", "Jomona", "บิบิป๊อป", "สแน็คแบ๊ค", "Veggie Snap", "ฟรูทร่า"]
    recog = parse_multiselect(aware.iloc[:, 4], all_brands)
    tried = parse_multiselect(aware.iloc[:, 5], all_brands)

    # === Multi-select: Calvora strengths (col 23), trust factors (col 27), Ebisen flavors (col 30) ===
    print("Parsing strengths, trust factors, Ebisen flavors...")
    strength_options = [
        "ใช้วัตถุดิบจากธรรมชาติ (Use natural ingredients)",
        "ทำจากเนื้อสัตว์แท้ (From real meat)",
        "มีรสชาติอร่อย (Tasty)",
        "มีคุณภาพดี (Good quality)",
        "เป็นแบรนด์ญี่ปุ่น (Japan brand)",
        "เพื่อสุขภาพที่ดี (For healthy lifestyles)",
    ]
    strengths = parse_multiselect(aware.iloc[:, 23], strength_options)

    trust_options = [
        "การรับรองมาตรฐานการผลิตจากญี่ปุ่น",
        "แหล่งที่มาของวัตถุดิบ",
        "เป็นผลิตภัณฑ์นำเข้าจากญี่ปุ่น",
        "การรับรองจากหน่วยงานด้านอาหาร",
        "ข้อมูลเกี่ยวกับกระบวนการผลิต",
        "เห็นขั้นตอนของการเก็บเกี่ยววัตถุดิบ",
    ]
    trust = parse_multiselect(aware.iloc[:, 27], trust_options)

    # NOTE: flavor columns ใช้สำหรับ "Business Insight chart" เท่านั้น
    # ไม่ใส่ใน clustering/supervised features เนื่องจากมีคนตอบเพียง 57/124
    # (เฉพาะคนที่เคยทาน Ebisen) — ใช้ prefix `report_` เพื่อให้ ML scripts
    # filter ออกได้ง่าย
    flavor_options = ["Original", "ต้มยำกุ้ง", "Extra BBQ", "หมึกย่างสาหร่าย"]
    flavors = parse_multiselect(aware.iloc[:, 30], flavor_options)

    # === Tagline ===
    tagline_known = (aware.iloc[:, 24] == "รู้").astype(int)
    tagline_rating = pd.to_numeric(aware.iloc[:, 26], errors="coerce")
    tagline_align = aware.iloc[:, 25].apply(tagline_alignment)

    # === NLP-derived features ===
    print("Running lightweight Thai NLP...")
    # Top-of-mind = multi-label binary flags
    topmind_records = aware.iloc[:, 2].apply(extract_top_of_mind_flags)
    topmind_df = pd.DataFrame(list(topmind_records))
    # Eating occasion = multi-label binary flags (occ_*)
    occ_records = aware.iloc[:, 11].apply(eating_occasion_flags)
    occ_df = pd.DataFrame(list(occ_records))

    # === Ebisen sub-section ===
    ebisen_familiarity_map = {
        "รู้จัก และเคยทาน": 3, "รู้จัก แต่ไม่เคยทาน": 2, "ไม่รู้จักเลย": 1,
    }
    ebisen_familiarity = aware.iloc[:, 29].map(ebisen_familiarity_map).fillna(0).astype(int)

    # Ebisen belief unified (col 32 OR 37) — สำคัญ: ใช้ก่อน classify reason
    ebisen_belief = aware.iloc[:, 32].fillna(aware.iloc[:, 37])
    ebisen_belief_bin = ebisen_belief.map({"เชื่อ": 1, "ไม่เชื่อ": 0})

    # Belief reason (เหตุผลที่เชื่อ/ไม่เชื่อ จาก cols 33 OR 38)
    belief_reasons_txt = aware.iloc[:, 33].fillna(aware.iloc[:, 38])
    belief_reason = belief_reasons_txt.apply(classify_belief_reason)

    # Cross belief × reason → "believer_sensory", "disbeliever_skeptical" etc.
    def _typed(belief, reason):
        if pd.isna(belief) or reason == "no_response":
            return "no_response"
        prefix = "believer" if belief == 1 else "disbeliever"
        return f"{prefix}_{reason}"
    belief_reason_typed = pd.Series(
        [_typed(b, r) for b, r in zip(ebisen_belief_bin, belief_reason)],
        index=aware.index
    )

    # Trial intent: try new flavor (col 34) OR intense flavor (col 39)
    try_new = aware.iloc[:, 34].map({"ลอง": 1, "ไม่ลอง": 0})
    try_intense = aware.iloc[:, 39].map({"ชอบรสเข้มข้น อยากลอง": 1, "ไม่อยากลอง": 0})
    trial_intent = try_new.fillna(try_intense)

    # === Demographics ===
    age_ord = aware.iloc[:, 44].map(AGE_MAP).fillna(0).astype(int)
    gender = aware.iloc[:, 45].fillna("unknown")
    gender_dummies = pd.get_dummies(gender, prefix="gender").astype(int)

    # === Awareness tier (TARGET for supervised) ===
    def to_tier(x):
        if x <= 2:
            return "Low"
        elif x <= 4:
            return "Mid"
        else:
            return "High"
    awareness_tier = brand_correct.apply(to_tier)

    # === Assemble final dataframe ===
    clean = pd.DataFrame({
        "respondent_id": range(1, len(aware) + 1),
        "aware": 1,
        "brand_correct": brand_correct,
        "brand_decoy_hit": decoy_hit,
        "brand_understanding": brand_understanding,
        "awareness_tier": awareness_tier,
        "products_recognized_count": recog["count"],
        "products_tried_count": tried["count"],
        "tagline_known": tagline_known,
        "tagline_rating": tagline_rating,
        "tagline_alignment": tagline_align,
        "ebisen_familiarity": ebisen_familiarity,
        "ebisen_belief": ebisen_belief_bin,
        "belief_reason": belief_reason,
        "belief_reason_typed": belief_reason_typed,
        "trial_intent": trial_intent,
        "age": age_ord,
        "gender": gender,
    })
    clean = pd.concat([clean, likert.reset_index(drop=True),
                       gender_dummies.reset_index(drop=True),
                       topmind_df.reset_index(drop=True),
                       occ_df.reset_index(drop=True)], axis=1)

    # Add per-brand recognition flags (for dashboard insight)
    for col in recog.columns:
        if col != "count":
            clean[f"recog_{col[4:]}"] = recog[col]
        if col != "count" and col in tried.columns:
            clean[f"tried_{col[4:]}"] = tried[col]

    # Add strengths, trust factors, flavors as binary cols
    for col in strengths.columns:
        if col != "count":
            clean[f"strength_{col[4:]}"] = strengths[col]
    for col in trust.columns:
        if col != "count":
            clean[f"trust_{col[4:]}"] = trust[col]
    # flavor_* — REPORTING ONLY (n=57). ใช้ prefix `report_flavor_` ให้ชัดเจน
    for col in flavors.columns:
        if col != "count":
            clean[f"report_flavor_{col[4:]}"] = flavors[col]

    # Save per-decoy hit counts as a separate small CSV for dashboard
    decoy_df = pd.DataFrame([{"brand": k, "wrong_count": v}
                              for k, v in decoy_per_brand.items()])
    decoy_df.to_csv(BASE / "data" / "decoy_per_brand.csv", index=False,
                     encoding="utf-8-sig")

    # Impute Likert NaN with median (only 3-4 cells affected)
    for c in ["score_ingredient", "score_taste", "score_variety",
              "score_texture", "score_health"]:
        clean[c] = clean[c].fillna(clean[c].median())

    # Tagline rating NaN → 0 (means didn't answer)
    clean["tagline_rating"] = clean["tagline_rating"].fillna(0)

    print(f"\nFinal clean shape: {clean.shape}")
    print("Awareness tier distribution:")
    print(clean["awareness_tier"].value_counts())
    print(f"\nSaving to {CLEAN}")
    clean.to_csv(CLEAN, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
