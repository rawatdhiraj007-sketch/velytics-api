from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io, re, warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Velytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Smart Clean Engine ────────────────────────────────────────────────────────
def smart_clean(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    fixes = []
    df = df_raw.copy()

    def header_score(row):
        score = 0
        for val in row:
            v = str(val).strip()
            if v and v.lower() not in ["nan","none",""]:
                if not re.match(r"^[\d,.\-\+\(\)₹$%\s]+$", v):
                    score += 1
        return score

    best_row = 0
    for i in range(1, min(10, len(df))):
        if header_score(df.iloc[i]) > header_score(df.iloc[best_row]):
            best_row = i

    df.columns = df.iloc[best_row].astype(str).str.strip()
    df = df.iloc[best_row + 1:].reset_index(drop=True)

    before = len(df)
    df.dropna(how="all", inplace=True)
    df.drop_duplicates(inplace=True)
    removed = before - len(df)
    if removed: fixes.append(f"Removed {removed} blank/duplicate rows")

    for col in df.select_dtypes(include="object").columns:
        try:
            df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            df[col] = df[col].replace({"nan": np.nan, "none": np.nan, "None": np.nan,
                                        "NULL": np.nan, "N/A": np.nan, "": np.nan})
            nuniq = df[col].nunique()
            if 0 < nuniq < 30:
                df[col] = df[col].str.title()
        except Exception:
            pass

    num_converted = 0
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(30)
            if len(sample) == 0: continue
            cleaned = sample.astype(str).str.replace(r"[₹$€£,\s%\(\)]", "", regex=True)
            if pd.to_numeric(cleaned, errors="coerce").notna().mean() >= 0.8:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r"[₹$€£,\s%]", "", regex=True),
                    errors="coerce"
                )
                num_converted += 1

    if num_converted: fixes.append(f"Converted {num_converted} columns to numeric")

    filled = 0
    for col in df.select_dtypes(include="number").columns:
        miss = df[col].isnull().sum()
        if miss:
            df[col] = df[col].fillna(df[col].median())
            filled += miss
    if filled: fixes.append(f"Filled {filled} missing values with median")

    df.reset_index(drop=True, inplace=True)
    return df, fixes


# ── Column Detector ───────────────────────────────────────────────────────────
def detect(df: pd.DataFrame, *keywords: str):
    for kw in keywords:
        col = next((c for c in df.columns if kw.lower() in c.lower()), None)
        if col: return col
    return None


# ── Sales Analysis ────────────────────────────────────────────────────────────
def analyze_sales(df: pd.DataFrame, region: str, category: str, period: str) -> dict:
    date_col    = detect(df, "date")
    rev_col     = detect(df, "total revenue", "revenue")
    region_col  = detect(df, "region")
    product_col = detect(df, "product name", "product")
    person_col  = detect(df, "salesperson", "sales person")
    qty_col     = detect(df, "quantity", "qty")
    target_col  = detect(df, "target")
    cat_col     = detect(df, "category")
    ctype_col   = detect(df, "customer type")
    payment_col = detect(df, "payment")

    # Apply filters
    dff = df.copy()
    if region != "All Regions" and region_col:
        dff = dff[dff[region_col].astype(str).str.title() == region]
    if category != "All Categories" and cat_col:
        dff = dff[dff[cat_col].astype(str).str.title() == category]

    if date_col:
        dff[date_col] = pd.to_datetime(dff[date_col], dayfirst=True, errors="coerce")
        if period == "Last 6 Months" and dff[date_col].notna().any():
            cutoff = dff[date_col].max() - pd.DateOffset(months=6)
            dff = dff[dff[date_col] >= cutoff]
        elif period == "Last 3 Months" and dff[date_col].notna().any():
            cutoff = dff[date_col].max() - pd.DateOffset(months=3)
            dff = dff[dff[date_col] >= cutoff]
        elif period == "This Month" and dff[date_col].notna().any():
            latest = dff[date_col].max()
            dff = dff[(dff[date_col].dt.month == latest.month) & (dff[date_col].dt.year == latest.year)]

    total_rev    = float(pd.to_numeric(dff[rev_col], errors="coerce").sum()) if rev_col else 0
    total_qty    = float(pd.to_numeric(dff[qty_col], errors="coerce").sum()) if qty_col else 0
    total_target = float(pd.to_numeric(dff[target_col], errors="coerce").sum()) if target_col else 0
    achievement  = round(total_rev / total_target * 100, 1) if total_target > 0 else 0
    aov          = round(total_rev / len(dff), 0) if len(dff) > 0 else 0

    # Monthly trend
    monthly_trend = []
    if date_col and rev_col:
        dff2 = dff.copy()
        dff2["_month"] = dff2[date_col].dt.to_period("M")
        mt = dff2.groupby("_month")[rev_col].sum().reset_index()
        mt.columns = ["month", "revenue"]
        monthly_trend = [{"month": str(r["month"]), "revenue": round(float(r["revenue"]), 0)} for _, r in mt.iterrows()]

    # By region
    by_region = []
    if region_col and rev_col:
        rr = dff.groupby(region_col)[rev_col].sum().sort_values(ascending=False).reset_index()
        total_r = rr[rev_col].sum()
        by_region = [{"region": str(r[region_col]), "revenue": round(float(r[rev_col]), 0),
                      "pct": round(float(r[rev_col]) / total_r * 100, 1) if total_r > 0 else 0}
                     for _, r in rr.iterrows()]

    # By category
    by_category = []
    if cat_col and rev_col:
        cc = dff.groupby(cat_col)[rev_col].sum().sort_values(ascending=False).reset_index()
        by_category = [{"category": str(r[cat_col]), "revenue": round(float(r[rev_col]), 0)}
                       for _, r in cc.iterrows()]

    # Top products
    top_products = []
    if product_col and rev_col:
        pp = dff.groupby(product_col)[rev_col].sum().sort_values(ascending=False).head(10).reset_index()
        max_rev = pp[rev_col].max()
        top_products = [{"product": str(r[product_col]),
                         "revenue": round(float(r[rev_col]), 0),
                         "pct": round(float(r[rev_col]) / max_rev * 100, 1) if max_rev > 0 else 0}
                        for _, r in pp.iterrows()]

    # Salesperson performance
    salesperson_perf = []
    if person_col and rev_col:
        sp_rev = dff.groupby(person_col)[rev_col].sum()
        sp_target = dff.groupby(person_col)[target_col].sum() if target_col else None
        sp_df = sp_rev.reset_index()
        sp_df.columns = [person_col, "revenue"]
        if sp_target is not None:
            sp_df = sp_df.merge(sp_target.reset_index().rename(columns={target_col: "target"}), on=person_col, how="left")
            sp_df["achievement"] = (sp_df["revenue"] / sp_df["target"] * 100).round(1)
        else:
            sp_df["achievement"] = 0
        sp_df = sp_df.sort_values("achievement" if "achievement" in sp_df.columns else "revenue", ascending=False)
        salesperson_perf = [{"name": str(r[person_col]),
                              "revenue": round(float(r["revenue"]), 0),
                              "achievement": round(float(r.get("achievement", 0)), 1)}
                            for _, r in sp_df.iterrows()]

    # Payment breakdown
    payment_breakdown = []
    if payment_col and rev_col:
        pm = dff.groupby(payment_col)[rev_col].sum().sort_values(ascending=False).reset_index()
        total_pm = pm[rev_col].sum()
        payment_breakdown = [{"method": str(r[payment_col]),
                               "revenue": round(float(r[rev_col]), 0),
                               "pct": round(float(r[rev_col]) / total_pm * 100, 1) if total_pm > 0 else 0}
                             for _, r in pm.iterrows()]

    # Customer type
    customer_type = []
    if ctype_col and rev_col:
        ct = dff.groupby(ctype_col)[rev_col].sum().sort_values(ascending=False).reset_index()
        total_ct = ct[rev_col].sum()
        customer_type = [{"type": str(r[ctype_col]),
                           "revenue": round(float(r[rev_col]), 0),
                           "pct": round(float(r[rev_col]) / total_ct * 100, 1) if total_ct > 0 else 0}
                        for _, r in ct.iterrows()]

    # Smart alerts
    alerts = []
    if by_region and target_col and rev_col:
        avg_rev = total_rev / max(len(by_region), 1)
        for r in by_region:
            if r["revenue"] < avg_rev * 0.75:
                alerts.append({"type": "Critical", "text": f"{r['region']} region is significantly below average ({_fmt(r['revenue'])} vs avg {_fmt(avg_rev)}). Review strategy."})
    if salesperson_perf:
        low = [s for s in salesperson_perf if s["achievement"] > 0 and s["achievement"] < 75]
        if low:
            alerts.append({"type": "Critical", "text": f"{len(low)} salesperson(s) below 75% of target: {', '.join(s['name'] for s in low[:3])}. Schedule 1:1 coaching."})
        high = [s for s in salesperson_perf if s["achievement"] >= 110]
        if high:
            alerts.append({"type": "Opportunity", "text": f"{high[0]['name']} exceeded target by {high[0]['achievement']-100:.0f}%. Document and replicate this approach."})
    if achievement < 80 and achievement > 0:
        alerts.append({"type": "Warning", "text": f"Overall target achievement is {achievement}% — below the 80% threshold. Review pipeline urgently."})
    if achievement >= 100:
        alerts.append({"type": "Opportunity", "text": f"Team achieved {achievement}% of target. Consider raising Q2 targets by 10-15%."})
    if not alerts:
        alerts.append({"type": "Info", "text": "No critical alerts. Business is performing within normal parameters."})

    # Forecast (simple linear)
    forecast = []
    if monthly_trend and len(monthly_trend) >= 3:
        from sklearn.linear_model import LinearRegression
        vals = [m["revenue"] for m in monthly_trend[-6:]]
        X = np.array(range(len(vals))).reshape(-1, 1)
        y = np.array(vals)
        model = LinearRegression().fit(X, y)
        for i in range(1, 4):
            pred = float(model.predict([[len(vals) + i - 1]])[0])
            forecast.append({"month": f"M+{i}", "revenue": round(max(pred, 0), 0), "forecast": True})

    return {
        "kpis": {
            "total_revenue": round(total_rev, 0),
            "total_revenue_fmt": _fmt(total_rev),
            "target_achievement": achievement,
            "units_sold": int(total_qty),
            "avg_order_value": round(aov, 0),
            "avg_order_value_fmt": _fmt(aov),
            "total_rows": len(dff),
        },
        "monthly_trend": monthly_trend,
        "by_region": by_region,
        "by_category": by_category,
        "top_products": top_products,
        "salesperson_perf": salesperson_perf,
        "payment_breakdown": payment_breakdown,
        "customer_type": customer_type,
        "alerts": alerts,
        "forecast": forecast,
        "columns_detected": {
            "date": date_col, "revenue": rev_col, "region": region_col,
            "product": product_col, "salesperson": person_col,
            "quantity": qty_col, "target": target_col, "category": cat_col,
        }
    }


def _fmt(v: float) -> str:
    """Format number in Indian style: Cr, L, K"""
    v = abs(v)
    if v >= 1e7:   return f"₹{v/1e7:.2f}Cr"
    if v >= 1e5:   return f"₹{v/1e5:.1f}L"
    if v >= 1e3:   return f"₹{v/1e3:.0f}K"
    return f"₹{v:.0f}"


# ── API Routes ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "product": "Velytics API", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    module: str = Form(default="Sales Intelligence"),
    region: str = Form(default="All Regions"),
    category: str = Form(default="All Categories"),
    period: str = Form(default="Last 12 Months"),
):
    # Validate file type
    name = file.filename.lower() if file.filename else ""
    if not any(name.endswith(ext) for ext in [".xlsx", ".csv", ".json"]):
        raise HTTPException(400, "Unsupported file. Please upload .xlsx, .csv, or .json")

    # Read file
    contents = await file.read()
    try:
        if name.endswith(".csv"):
            df_raw = pd.read_csv(io.BytesIO(contents), low_memory=False)
        elif name.endswith(".json"):
            df_raw = pd.read_json(io.BytesIO(contents))
        else:
            df_raw = pd.read_excel(io.BytesIO(contents), header=None)
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {str(e)}")

    # Clean
    try:
        df, fixes = smart_clean(df_raw)
    except Exception as e:
        raise HTTPException(500, f"Cleaning failed: {str(e)}")

    # Analyse
    try:
        if "Sales" in module:
            result = analyze_sales(df, region, category, period)
        else:
            # Other modules: return basic profile for now
            result = {
                "kpis": {"total_rows": len(df), "total_columns": len(df.columns)},
                "columns": list(df.columns),
                "preview": df.head(5).to_dict(orient="records"),
            }
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

    return {
        "module": module,
        "file": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "fixes": fixes,
        "filters": {"region": region, "category": category, "period": period},
        "result": result,
    }
