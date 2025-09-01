"""
Simple Dash dashboard for company 106 (product K3AG1).

Visualizes market share, failure rates, survival curves, mileage at failure,
and a data table (now same width as other cards). Single file for easy review.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dash
from dash import Dash, Input, Output, dcc, html
from dash import dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# -------------------- Theme / constants --------------------

TEXT_DARK = "#0f172a"
CORPORATE_LIGHT_BLUE = "#e6f0ff"
CORPORATE_BLUE = "#1d6fff"
CORPORATE_BLUE_SOFT = "#5aa3ff"

CARD_BG = "#ffffff"  # cards are pure white
CARD_BORDER = "rgba(0,0,0,0.08)"
MAC_SHADOW = "0 12px 30px rgba(0,0,0,0.08)"

DATA_FILE = "final_dataset_group_23.csv"

EXTERNAL_STYLESHEETS = [
    "https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700;900&display=swap"
]

PX_TEMPLATE = "plotly_white"

# keep 106 blue; others warm-ish
BASE_COLOR_MAP: Dict[str, str] = {
    "106": "#1d6fff",
    "105": "#f2cc8f",
    "107": "#f4b183",
    "108": "#f7d9a8",
}
WARM_FALLBACK: List[str] = [
    "#f2cc8f", "#f4b183", "#f7d9a8", "#eddcc8", "#ffe0b2", "#ffe5c4"
]

# parts palette
PART_COLOR_MAP: Dict[str, str] = {
    "K3AG1": "#7fb3ff",
    "K3SG1": "#ffd6a5",
}

TABLE_ROW_CAP = 10_000


# -------------------- Data helpers --------------------

def load_dataset_for_kpi(path: Path) -> pd.DataFrame:
    """Load minimal cols for the slogan calc."""
    return pd.read_csv(
        path,
        usecols=["gearbox_manufacturer_id"],
        dtype={"gearbox_manufacturer_id": "category"},
    )


def load_dataset_for_aggregations(path: Path) -> pd.DataFrame:
    """Load all fields needed for charts + table."""
    df = pd.read_csv(
        path,
        usecols=[
            "gearbox_id",
            "vehicle_id",
            "vehicle_type",
            "gearbox_manufacturer_id",
            "gearbox_plant_id",
            "gearbox_type",
            "gearbox_production_date",
            "gearbox_defective_flag",
            "gearbox_defective_date",
            "gearbox_defective_mileage",
        ],
        dtype={
            "gearbox_id": "string",
            "vehicle_id": "string",
            "vehicle_type": "category",
            "gearbox_manufacturer_id": "category",
            "gearbox_plant_id": "category",
            "gearbox_type": "category",
            "gearbox_defective_flag": "int8",
            "gearbox_defective_mileage": "Int64",
        },
        parse_dates=["gearbox_production_date", "gearbox_defective_date"],
    )
    return df


def compute_x_for_slogan(df: pd.DataFrame) -> Optional[int]:
    """Return x in 'A component from 106 is found in every x gearbox'."""
    total = len(df)
    count_106 = (df["gearbox_manufacturer_id"] == "106").sum()
    if total == 0 or count_106 == 0:
        return None
    return max(1, int(total / count_106 + 0.5))


def preaggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly produced/defective by manufacturer and type."""
    month = df["gearbox_production_date"].dt.to_period("M").dt.to_timestamp("M")
    grp = (
        df.assign(month=month)
        .groupby(["month", "gearbox_manufacturer_id", "gearbox_type"], observed=True)
        .agg(
            produced_count=("gearbox_type", "size"),
            defective_count=("gearbox_defective_flag", "sum"),
        )
        .reset_index()
        .sort_values("month")
    )
    grp["gearbox_manufacturer_id"] = grp["gearbox_manufacturer_id"].astype("category")
    grp["gearbox_type"] = grp["gearbox_type"].astype("category")
    return grp


def get_color_map_for_manus(manus: List[str]) -> Dict[str, str]:
    """Build a color map: 106 fixed blue, rest warm."""
    cmap: Dict[str, str] = {}
    used = set()
    for m in manus:
        if m in BASE_COLOR_MAP:
            cmap[m] = BASE_COLOR_MAP[m]
            used.add(BASE_COLOR_MAP[m])
    for m in manus:
        if m not in cmap:
            for c in WARM_FALLBACK:
                if c not in used:
                    cmap[m] = c
                    used.add(c)
                    break
            else:
                cmap[m] = WARM_FALLBACK[-1]
    return cmap


# -------------------- KM survival (compact helper) --------------------

def km_curve(times: pd.Series, events: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Small Kaplan–Meier with Greenwood SE."""
    df = pd.DataFrame({"t": times.astype(int), "event": events.astype(bool)}).sort_values("t")
    if df.empty:
        return pd.Series([0]), pd.Series([1.0]), pd.Series([0.0])

    di = df.groupby("t")["event"].sum().astype(int)
    vc = df["t"].value_counts().sort_index()
    at_risk_from_t = vc[::-1].cumsum()[::-1]
    ni = at_risk_from_t.reindex(di.index).astype(int)

    valid = di[di > 0]
    if valid.empty:
        return pd.Series([0]), pd.Series([1.0]), pd.Series([0.0])
    ni = ni.loc[valid.index]

    S_vals: List[float] = []
    se_vals: List[float] = []
    S_curr = 1.0
    var_sum = 0.0
    for n_i, d_i in zip(ni.tolist(), valid.tolist()):
        if n_i <= 0 or d_i > n_i:
            continue
        S_curr *= (1.0 - d_i / n_i)
        S_vals.append(S_curr)
        if n_i > d_i:
            var_sum += d_i / (n_i * (n_i - d_i))
        var_S = (S_curr ** 2) * var_sum
        se_vals.append(var_S ** 0.5)

    t = pd.Series([0] + valid.index.tolist(), dtype=int)
    S = pd.Series([1.0] + S_vals, dtype=float)
    se = pd.Series([0.0] + se_vals, dtype=float)
    return t, S, se


# -------------------- App --------------------

def create_app() -> Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=EXTERNAL_STYLESHEETS,
        suppress_callback_exceptions=True,
        title="106 • K3AG1 Quality Dashboard",
        assets_folder="www"
    )

    # Global CSS (mac-like gradient; white cards)
    app.index_string = f"""
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <style>
                :root {{
                    --card-bg: {CARD_BG};
                    --card-border: {CARD_BORDER};
                    --text-dark: {TEXT_DARK};
                    --sidebar-w: 340px;
                    --gap-lg: 28px;
                    --blue: {CORPORATE_BLUE};
                    --blue-soft: {CORPORATE_BLUE_SOFT};
                    --blue-verylight: {CORPORATE_LIGHT_BLUE};
                }}
                html, body {{
                    height: 100%;
                    background: linear-gradient(180deg, #eef5ff 0%, #eaf2ff 35%, #e6f0ff 65%, #edf3ff 100%);
                    color: var(--text-dark);
                    font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont,
                                 "Segoe UI", Roboto, "Helvetica Neue", Arial, "Apple Color Emoji",
                                 "Segoe UI Emoji", "Segoe UI Symbol";
                }}

                .page {{ display: flex; flex-direction: row; align-items: flex-start; }}

                .sidebar {{
                    position: fixed;
                    top: 0; left: 0;
                    width: var(--sidebar-w);
                    height: 100vh;
                    overflow-y: auto;
                    padding: 22px 18px 30px 18px;
                    box-sizing: border-box;
                    backdrop-filter: saturate(180%) blur(16px);
                    background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(250,250,255,0.76));
                    border-right: 1px solid var(--card-border);
                    box-shadow: 0 12px 34px rgba(0,0,0,0.06);
                }}
                /* make the embedded svg scale nicely */
                .logo svg {{
                    height: 32px;
                    width: auto;
                    display: block;
                }}
                .sidebar h1 {{
                    font-size: 20px; margin: 12px 0 6px 0; font-weight: 900; line-height: 1.2;
                    color: #0b2239;
                }}
                .sidebar .sub {{ font-size: 13px; opacity: 0.95; line-height: 1.55; margin: 8px 0 8px; }}
                .sidebar .author {{ margin-top: 2px; font-size: 12px; opacity: 0.9; color: #334155; }}

                .controls {{
                    margin-top: 12px;
                    background: var(--card-bg);
                    border: 1px solid var(--card-border);
                    border-radius: 16px;
                    padding: 14px 14px;
                    box-shadow: {MAC_SHADOW};
                }}
                .controls .block {{ margin-bottom: 16px; }}
                .controls .label {{ display: block; font-size: 13px; font-weight: 900; margin-bottom: 6px; color: #0b2239; }}
                .controls .hint {{ font-size: 11px; opacity: 0.9; margin-top: 4px; color: #3f4d63; }}

                .DateRangePickerInput {{
                    background: var(--blue-verylight);
                    border: 1px solid #c7ddff;
                    border-radius: 10px;
                    padding: 6px 8px;
                }}
                .DateInput_input {{ background: transparent; color: #0b2239; font-weight: 700; }}
                .CalendarDay__selected_span,
                .CalendarDay__selected,
                .CalendarDay__hovered_span {{
                    background: var(--blue) !important;
                    border: 1px solid var(--blue) !important;
                    color: #fff !important;
                }}
                .CalendarDay__default {{ border: 1px solid #e2e8f0; color: #0b2239; }}
                .DayPickerKeyboardShortcuts_buttonReset {{ display: none !important; }}

                .right {{
                    margin-left: calc(var(--sidebar-w) + var(--gap-lg));
                    padding: 28px var(--gap-lg) 48px var(--gap-lg);
                    box-sizing: border-box;
                    width: calc(100vw - var(--sidebar-w) - var(--gap-lg)*2);
                }}

                .grid-1 {{ display: grid; grid-template-columns: 1fr; gap: var(--gap-lg); margin-bottom: var(--gap-lg); }}
                .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: var(--gap-lg); margin-bottom: var(--gap-lg); }}
                .grid-2-1 {{ display: grid; grid-template-columns: 2fr 1fr; gap: var(--gap-lg); margin-bottom: var(--gap-lg); }}

                .card {{
                    background: var(--card-bg);
                    border: 1px solid var(--card-border);
                    border-radius: 18px;
                    padding: 22px 24px;
                    box-shadow: {MAC_SHADOW};
                }}
                .card h3 {{ margin: 0 0 10px 0; font-size: 18px; font-weight: 900; color: #0b2239; }}
                .card .note {{ font-size: 12px; opacity: 0.85; margin-top: 8px; color: #3f4d63; }}

                .slogan {{
                    display: flex; align-items: center; justify-content: center; text-align: center;
                    border-radius: 18px; padding: 24px 22px;
                    font-size: 26px; font-weight: 900; letter-spacing: 0.3px; color: #0b2239;
                    background: var(--card-bg);
                    border: 1px solid var(--card-border);
                    box-shadow: {MAC_SHADOW};
                }}

                .btn {{
                    appearance: none; border: 1px solid #d1d9e6; border-radius: 12px;
                    padding: 9px 14px; background: #ffffff;
                    font-weight: 800; color: #0b2239; cursor: pointer;
                    box-shadow: 0 6px 16px rgba(0,0,0,0.06);
                    transition: all .15s ease;
                }}
                .btn:hover {{ transform: translateY(-1px); box-shadow: 0 10px 20px rgba(0,0,0,0.08); }}
                .btn:active {{ transform: translateY(0); box-shadow: 0 6px 16px rgba(0,0,0,0.06); }}
                .btn.blue {{ border-color: #c7ddff; background: var(--blue-verylight); color: #0b2239; }}

                .dash-table-container .dash-spreadsheet-container th,
                .dash-table-container .dash-spreadsheet-container td {{
                    font-family: "Source Sans Pro", Arial, sans-serif;
                    font-size: 14px;
                }}
                .dash-table-container .dash-spreadsheet-container th {{
                    background: #f8fafc; color: #0b2239; font-weight: 900;
                }}

                .graph-gap {{ margin-top: 12px; }}

                @media (max-width: 1280px) {{
                    .grid-2, .grid-2-1 {{ grid-template-columns: 1fr; }}
                    .right {{ width: calc(100% - var(--sidebar-w) - var(--gap-lg)); }}
                }}
                @media (max-width: 900px) {{
                    :root {{ --sidebar-w: 280px; }}
                    .right {{ width: calc(100% - var(--sidebar-w) - var(--gap-lg)); }}
                }}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
        </body>
    </html>
    """

    # Slogan KPI
    data_error: Optional[str] = None
    df_kpi = load_dataset_for_kpi(Path(DATA_FILE))
    x_val = compute_x_for_slogan(df_kpi)

    slogan_text = f"There is a part of 106 in every {x_val} gearbox"

    # Shared frames for callbacks
    market_error: Optional[str] = None
    manu_options: List[str] = []
    default_type_selection: List[str] = ["K3AG1", "K3SG1"]
    try:
        df_all = load_dataset_for_aggregations(Path(DATA_FILE))
        df_monthly = preaggregate_monthly(df_all)
        app.server.df_all = df_all          # type: ignore[attr-defined]
        app.server.df_monthly = df_monthly  # type: ignore[attr-defined]

        manu_options = sorted(df_all["gearbox_manufacturer_id"].astype(str).unique())
        min_date = pd.to_datetime(df_all["gearbox_production_date"].min())
        max_date = pd.to_datetime(df_all["gearbox_production_date"].max())
    except Exception as exc:  # pragma: no cover
        market_error = str(exc)
        min_date = pd.Timestamp("2008-11-12")
        max_date = pd.Timestamp("2016-11-15")

    if not manu_options:
        manu_options = ["105", "106", "107", "108"]

    # -------------------- Layout --------------------

    app.layout = html.Div(
        className="page",
        children=[
            # Sidebar with embedded SVG logo
            html.Div(
                className="sidebar",
                children=[
                    dcc.Markdown(
                        r"""
                        <svg width="124" height="69" viewBox="0 0 124 69" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M13.7143 55.2857H20.5714V62.1429H13.7143V55.2857ZM13.7143 62.1429H20.5714V69H13.7143V62.1429ZM6.85714 62.1429H13.7143V69H6.85714V62.1429ZM6.85714 55.2857H13.7143V62.1429H6.85714V55.2857ZM6.85714 48.4286H13.7143V55.2857H6.85714V48.4286ZM6.85714 41.5714H13.7143V48.4286H6.85714V41.5714ZM6.85714 34.7143H13.7143V41.5714H6.85714V34.7143ZM6.85714 27.8571H13.7143V34.7143H6.85714V27.8571ZM6.85714 21H13.7143V27.8571H6.85714V21ZM6.85714 14.1429H13.7143V21H6.85714V14.1429ZM13.7143 21H20.5714V27.8571H13.7143V21ZM13.7143 27.8571H20.5714V34.7143H13.7143V27.8571ZM13.7143 34.7143H20.5714V41.5714H13.7143V34.7143ZM13.7143 41.5714H20.5714V48.4286H13.7143V41.5714ZM13.7143 48.4286H20.5714V55.2857H13.7143V48.4286ZM13.7143 14.1429H20.5714V21H13.7143V14.1429ZM13.7143 7.28571H20.5714V14.1429H13.7143V7.28571ZM6.85714 0.428574H13.7143V7.28571H6.85714V0.428574ZM13.7143 0.428574H20.5714V7.28571H13.7143V0.428574ZM6.85714 7.28571H13.7143V14.1429H6.85714V7.28571ZM0 0.428574H6.85714V7.28571H0V0.428574ZM0 7.28571H6.85714V14.1429H0V7.28571ZM41.0893 0.428574H47.9464V7.28571H41.0893V0.428574ZM34.2321 0.428574H41.0893V7.28571H34.2321V0.428574ZM34.2321 7.28571H41.0893V14.1429H34.2321V7.28571ZM41.0893 7.28571H47.9464V14.1429H41.0893V7.28571ZM47.9464 7.28571H54.8036V14.1429H47.9464V7.28571ZM47.9464 0.428574H54.8036V7.28571H47.9464V0.428574ZM54.8036 0.428574H61.6607V7.28571H54.8036V0.428574ZM61.6607 7.28571H68.5179V14.1429H61.6607V7.28571ZM54.8036 7.28571H61.6607V14.1429H54.8036V7.28571ZM61.6607 14.1429H68.5179V21H61.6607V14.1429ZM61.6607 55.2857H68.5179V62.1429H61.6607V55.2857ZM61.6607 48.4286H68.5179V55.2857H61.6607V48.4286ZM61.6607 41.5714H68.5179V48.4286H61.6607V41.5714ZM61.6607 34.7143H68.5179V41.5714H61.6607V34.7143ZM61.6607 27.8571H68.5179V34.7143H61.6607V27.8571ZM61.6607 21H68.5179V27.8571H61.6607V21ZM54.8036 55.2857H61.6607V62.1429H54.8036V55.2857ZM54.8036 62.1429H61.6607V69H54.8036V62.1429ZM47.9464 62.1429H54.8036V69H47.9464V62.1429ZM47.9464 55.2857H54.8036V62.1429H47.9464V55.2857ZM41.0893 62.1429H47.9464V69H41.0893V62.1429ZM34.2321 62.1429H41.0893V69H34.2321V62.1429ZM34.2321 55.2857H41.0893V62.1429H34.2321V55.2857ZM41.0893 55.2857H47.9464V62.1429H41.0893V55.2857ZM34.2321 48.4286H41.0893V55.2857H34.2321V48.4286ZM27.375 55.2857H34.2321V62.1429H27.375V55.2857ZM27.375 48.4286H34.2321V55.2857H27.375V48.4286ZM27.375 41.5714H34.2321V48.4286H27.375V41.5714ZM27.375 34.7143H34.2321V41.5714H27.375V34.7143ZM27.375 27.8571H34.2321V34.7143H27.375V27.8571ZM27.375 21H34.2321V27.8571H27.375V21ZM27.375 14.1429H34.2321V21H27.375V14.1429ZM34.2321 14.1429H41.0893V21H34.2321V14.1429ZM27.375 7.28571H34.2321V14.1429H27.375V7.28571ZM34.2321 21H41.0893V27.8571H34.2321V21ZM34.2321 27.8571H41.0893V34.7143H34.2321V27.8571ZM34.2321 34.7143H41.0893V41.5714H34.2321V34.7143ZM34.2321 41.5714H41.0893V48.4286H34.2321V41.5714ZM54.8036 14.1429H61.6607V21H54.8036V14.1429ZM54.8036 21H61.6607V27.8571H54.8036V21ZM54.8036 27.8571H61.6607V34.7143H54.8036V27.8571ZM54.8036 34.7143H61.6607V41.5714H54.8036V34.7143ZM54.8036 41.5714H61.6607V48.4286H54.8036V41.5714ZM54.8036 48.4286H61.6607V55.2857H54.8036V48.4286ZM82.2321 0.428574H89.0893V7.28571H82.2321V0.428574ZM89.0893 0.428574H95.9464V7.28571H89.0893V0.428574ZM102.804 0.428574H109.661V7.28571H102.804V0.428574ZM109.661 0.428574H116.518V7.28571H109.661V0.428574ZM109.661 7.28571H116.518V14.1429H109.661V7.28571ZM109.661 14.1429H116.518V21H109.661V14.1429ZM116.518 14.1429H123.375V21H116.518V14.1429ZM116.518 7.28571H123.375V14.1429H116.518V7.28571ZM102.804 7.28571H109.661V14.1429H102.804V7.28571ZM89.0893 7.28571H95.9464V14.1429H89.0893V7.28571ZM82.2321 7.28571H89.0893V14.1429H82.2321V7.28571ZM75.375 7.28571H82.2321V14.1429H75.375V7.28571ZM75.375 14.1429H82.2321V21H75.375V14.1429ZM82.2321 14.1429H89.0893V21H82.2321V14.1429ZM82.2321 21H89.0893V27.8571H82.2321V21ZM75.375 21H82.2321V27.8571H75.375V21ZM75.375 27.8571H82.2321V34.7143H75.375V27.8571ZM75.375 34.7143H82.2321V41.5714H75.375V34.7143ZM82.2321 34.7143H89.0893V41.5714H82.2321V34.7143ZM82.2321 27.8571H89.0893V34.7143H82.2321V27.8571ZM89.0893 27.8571H95.9464V34.7143H89.0893V27.8571ZM89.0893 34.7143H95.9464V41.5714H89.0893V34.7143ZM102.804 34.7143H109.661V41.5714H102.804V34.7143ZM102.804 27.8571H109.661V34.7143H102.804V27.8571ZM109.661 27.8571H116.518V34.7143H109.661V27.8571ZM109.661 34.7143H116.518V41.5714H109.661V34.7143ZM116.518 34.7143H123.375V41.5714H116.518V34.7143ZM116.518 41.5714H123.375V48.4286H116.518V41.5714ZM116.518 48.4286H123.375V55.2857H116.518V48.4286ZM109.661 48.4286H116.518V55.2857H109.661V48.4286ZM109.661 41.5714H116.518V48.4286H109.661V41.5714ZM116.518 55.2857H123.375V62.1429H116.518V55.2857ZM109.661 55.2857H116.518V62.1429H109.661V55.2857ZM109.661 62.1429H116.518V69H109.661V62.1429ZM102.804 62.1429H109.661V69H102.804V62.1429ZM102.804 55.2857H109.661V62.1429H102.804V55.2857ZM89.0893 55.2857H95.9464V62.1429H89.0893V55.2857ZM89.0893 62.1429H95.9464V69H89.0893V62.1429ZM82.2321 62.1429H89.0893V69H82.2321V62.1429ZM82.2321 55.2857H89.0893V62.1429H82.2321V55.2857ZM75.375 55.2857H82.2321V62.1429H75.375V55.2857ZM75.375 48.4286H82.2321V55.2857H75.375V48.4286ZM75.375 41.5714H82.2321V48.4286H75.375V41.5714ZM82.2321 41.5714H89.0893V48.4286H82.2321V41.5714ZM82.2321 48.4286H89.0893V55.2857H82.2321V48.4286ZM95.9464 7.28571H102.804V14.1429H95.9464V7.28571ZM95.9464 0.428574H102.804V7.28571H95.9464V0.428574ZM95.9464 27.8571H102.804V34.7143H95.9464V27.8571ZM95.9464 34.7143H102.804V41.5714H95.9464V34.7143ZM95.9464 55.2857H102.804V62.1429H95.9464V55.2857ZM95.9464 62.1429H102.804V69H95.9464V62.1429Z" fill="#1D6FFF"/>
                        </svg>
                        """,
                            className="logo",
                            dangerously_allow_html=True,
                        ),
                    html.H1("K3 Series Gearboxes — Quality & Market Insights"),
                    html.Div(
                        "Manufacturer 106 produces the K3AG1 gearbox for vehicle types Type11 and Type12. "
                        "This dashboard explores production, defects, and reliability across the market.",
                        className="sub",
                    ),
                    html.Div("Author: group23", className="author"),
                    html.Div(
                        className="controls",
                        children=[
                            html.Div(
                                className="block",
                                children=[
                                    html.Span("Display mode", className="label"),
                                    dcc.RadioItems(
                                        id="ms-mode",
                                        options=[
                                            {"label": "Absolute", "value": "abs"},
                                            {"label": "Percentage", "value": "pct"},
                                        ],
                                        value="abs",
                                        inputStyle={"marginRight": "6px"},
                                        labelStyle={"marginRight": "12px"},
                                    ),
                                    html.Div("Applies to market share and error frequency.", className="hint"),
                                ],
                            ),
                            html.Div(
                                className="block",
                                children=[
                                    html.Span("Gearbox types", className="label"),
                                    dcc.Checklist(
                                        id="ms-types",
                                        options=[
                                            {"label": "K3AG1(us)", "value": "K3AG1"},
                                            {"label": "K3SG1", "value": "K3SG1"},
                                        ],
                                        value=default_type_selection,
                                        inputStyle={"marginRight": "6px"},
                                        labelStyle={"marginRight": "12px"},
                                    ),
                                ],
                            ),
                            html.Div(
                                className="block",
                                children=[
                                    html.Span("Manufacturers", className="label"),
                                    dcc.Checklist(
                                        id="ms-manus",
                                        options=[
                                            {"label": ("106 (us)" if m == "106" else m), "value": m}
                                            for m in manu_options
                                        ],
                                        value=manu_options,
                                        inputStyle={"marginRight": "6px"},
                                        labelStyle={"marginRight": "12px"},
                                    ),
                                ],
                            ),
                            html.Div(
                                className="block",
                                children=[
                                    html.Span("Production date range", className="label"),
                                    dcc.DatePickerRange(
                                        id="ms-daterange",
                                        minimum_nights=0,
                                        min_date_allowed=min_date,
                                        max_date_allowed=max_date,
                                        start_date=min_date,
                                        end_date=max_date,
                                        display_format="YYYY-MM-DD",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # Right content
            html.Div(
                className="right",
                children=[
                    # Market share + Slogan
                    html.Div(
                        className="grid-1",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.H3("Market share over time"),
                                    html.Div(
                                        dcc.Graph(
                                            id="market-share-chart",
                                            config={"displayModeBar": True},
                                            figure=go.Figure(),
                                            style={"height": "460px"},
                                        ),
                                        className="graph-gap",
                                    ),
                                    html.Div(
                                        "106 is emphasized in corporate blue; other manufacturers use muted warm tones. "
                                        "Switch between units and share to adjust the y-axis.",
                                        className="note",
                                    ),
                                ],
                            ),
                            html.Div(className="slogan", children=[slogan_text]),
                        ],
                    ),

                    # Failure rate + Error frequency
                    html.Div(
                        className="grid-2-1",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.H3("Failure rate over time"),
                                    html.Div(
                                        dcc.Graph(
                                            id="failure-rate-line",
                                            config={"displayModeBar": True},
                                            figure=go.Figure(),
                                            style={"height": "420px"},
                                        ),
                                        className="graph-gap",
                                    ),
                                    html.Div(
                                        "Monthly failure rate = defects / produced for the active selection. "
                                        "Filter by manufacturer, type and dates to focus the trend.",
                                        className="note",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="card",
                                children=[
                                    html.H3("Error frequency"),
                                    html.Div(
                                        dcc.Graph(
                                            id="error-frequency-chart",
                                            config={"displayModeBar": True},
                                            figure=go.Figure(),
                                            style={"height": "420px"},
                                        ),
                                        className="graph-gap",
                                    ),
                                    html.Div(
                                        "View defective unit counts (absolute) or defect rates (percentage) per bucket. "
                                        "The non-106 bucket name adapts to your manufacturer selection.",
                                        className="note",
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # Survival + Mileage
                    html.Div(
                        className="grid-2",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.H3("Survival (Kaplan–Meier)"),
                                    html.Div(
                                        dcc.Graph(
                                            id="km-survival",
                                            config={"displayModeBar": True},
                                            figure=go.Figure(),
                                            style={"height": "440px"},
                                        ),
                                        className="graph-gap",
                                    ),
                                    html.Div(
                                        "S(t) is the probability a unit has not failed by time t. "
                                        "Right-censoring is applied at the selected observation end.",
                                        className="note",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="card",
                                children=[
                                    html.H3("Mileage distribution at failure"),
                                    html.Div(
                                        dcc.Graph(
                                            id="mileage-hist",
                                            config={"displayModeBar": True},
                                            figure=go.Figure(),
                                            style={"height": "440px"},
                                        ),
                                        className="graph-gap",
                                    ),
                                    html.Div(
                                        "Histogram of reported mileage at failure, stacked by manufacturer.",
                                        className="note",
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # Dataset table (same width as other cards)
                    html.Div(
                        className="grid-1",
                        children=[
                            html.Div(
                                className="card",
                                style={"width": "100%", "overflowX": "auto"},
                                children=[
                                    html.H3("dataset"),
                                    html.Div(
                                        style={"margin": "8px 0 12px 0", "width": "100%", "overflowX": "auto"},
                                        children=[
                                            html.Button("Export CSV", id="btn-export", className="btn blue"),
                                            dcc.Download(id="download-table"),
                                        ],
                                    ),
                                    dash_table.DataTable(
                                        id="support-table",
                                        columns=[
                                            {"name": "gearbox_id", "id": "gearbox_id", "type": "text"},
                                            {"name": "vehicle_id", "id": "vehicle_id", "type": "text"},
                                            {"name": "vehicle_type", "id": "vehicle_type", "type": "text"},
                                            {"name": "gearbox_manufacturer_id", "id": "gearbox_manufacturer_id", "type": "text"},
                                            {"name": "gearbox_plant_id", "id": "gearbox_plant_id", "type": "text"},
                                            {"name": "gearbox_type", "id": "gearbox_type", "type": "text"},
                                            {"name": "gearbox_production_date", "id": "gearbox_production_date", "type": "datetime"},
                                            {"name": "gearbox_defective_flag", "id": "gearbox_defective_flag", "type": "numeric"},
                                            {"name": "gearbox_defective_date", "id": "gearbox_defective_date", "type": "datetime"},
                                            {"name": "gearbox_defective_mileage", "id": "gearbox_defective_mileage", "type": "numeric"},
                                        ],
                                        data=[],
                                        page_current=0,
                                        page_size=10,
                                        page_action="native",
                                        sort_action="native",
                                        filter_action="native",
                                        style_table={"overflowX": "auto",  "width": "100%", "maxWidth": "100%", "box-sizing": "border-box"},
                                        style_cell={
                                            "fontFamily": "Source Sans Pro, Arial, sans-serif",
                                            "fontSize": "14px",
                                            "padding": "10px",
                                            "whiteSpace": "nowrap",
                                            "textOverflow": "ellipsis",
                                            "maxWidth": 240,
                                        },
                                        style_header={
                                            "fontWeight": "900",
                                            "backgroundColor": "#f8fafc",
                                            "borderBottom": "1px solid #e2e8f0",
                                        },
                                        style_data={"borderBottom": "1px solid #e2e8f0"},
                                        export_format=None,
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # Error panels
                    html.Div(
                        className="card",
                        style={"marginBottom": "24px", "display": "block" if data_error else "none", "borderColor": "#fecaca"},
                        children=[html.H3("Dataset load warning (KPI)"),
                                  html.Div(data_error or "", style={"fontSize": "14px", "whiteSpace": "pre-wrap"})],
                    ),
                    html.Div(
                        className="card",
                        style={"marginBottom": "24px", "display": "block" if market_error else "none", "borderColor": "#fecaca"},
                        children=[html.H3("Dataset load warning (Aggregations)"),
                                  html.Div(market_error or "", style={"fontSize": "14px", "whiteSpace": "pre-wrap"})],
                    ),
                ],
            ),
        ],
    )

    # -------------------- Callbacks --------------------

    @app.callback(
        Output("market-share-chart", "figure"),
        Input("ms-mode", "value"),
        Input("ms-types", "value"),
        Input("ms-manus", "value"),
        Input("ms-daterange", "start_date"),
        Input("ms-daterange", "end_date"),
        prevent_initial_call=False,
    )
    def update_market_share_chart(mode, selected_types, selected_manus, start_date, end_date):
        """Stacked bars of produced units or share by month."""
        dfm: Optional[pd.DataFrame] = getattr(app.server, "df_monthly", None)  # type: ignore[attr-defined]
        if dfm is None or not selected_types or not selected_manus:
            fig = go.Figure()
            fig.update_layout(template=PX_TEMPLATE, title=None, margin=dict(l=6, r=6, t=34, b=6))
            return fig

        filt = dfm[
            (dfm["gearbox_type"].isin(selected_types))
            & (dfm["gearbox_manufacturer_id"].isin(selected_manus))
        ].copy()
        if start_date:
            filt = filt[filt["month"] >= pd.to_datetime(start_date)]
        if end_date:
            filt = filt[filt["month"] <= pd.to_datetime(end_date)]
        if filt.empty:
            fig = go.Figure()
            fig.update_layout(template=PX_TEMPLATE, title=None, margin=dict(l=6, r=6, t=34, b=6))
            return fig

        if mode == "pct":
            totals = filt.groupby("month", as_index=False)["produced_count"].sum()
            totals = totals.rename(columns={"produced_count": "total"})
            merged = filt.merge(totals, on="month", how="left")
            merged["value"] = merged["produced_count"] / merged["total"]
            y_title = "Share"
        else:
            merged = filt.copy()
            merged["value"] = merged["produced_count"]
            y_title = "Units"

        manus_unique = merged["gearbox_manufacturer_id"].astype(str).unique().tolist()
        manus_order = (["106"] if "106" in manus_unique else []) + [
            m for m in sorted(manus_unique) if m != "106"
        ]
        color_map = get_color_map_for_manus(manus_order)

        fig = px.bar(
            merged.sort_values("month"),
            x="month",
            y="value",
            color="gearbox_manufacturer_id",
            barmode="stack",
            category_orders={"gearbox_manufacturer_id": manus_order},
            color_discrete_map=color_map,
            labels={"month": "Production month", "value": y_title, "gearbox_manufacturer_id": "Manufacturer"},
            template=PX_TEMPLATE,
        )
        fig.update_layout(
            title=None,
            legend_title_text="Manufacturer",
            margin=dict(l=6, r=6, t=34, b=6),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        if mode == "pct":
            fig.update_yaxes(tickformat=".0%")
            hover_tmpl = "Month: %{x|%Y-%m}<br>Manufacturer: %{customdata[0]}<br>Share: %{y:.1%}<extra></extra>"
        else:
            fig.update_yaxes(separatethousands=True)
            hover_tmpl = "Month: %{x|%Y-%m}<br>Manufacturer: %{customdata[0]}<br>Units: %{y:,}<extra></extra>"
        fig.update_traces(hovertemplate=hover_tmpl, customdata=merged[["gearbox_manufacturer_id"]])
        return fig

    @app.callback(
        Output("error-frequency-chart", "figure"),
        Input("ms-mode", "value"),
        Input("ms-types", "value"),
        Input("ms-manus", "value"),
        Input("ms-daterange", "start_date"),
        Input("ms-daterange", "end_date"),
        prevent_initial_call=False,
    )
    def update_error_frequency_chart(mode, selected_types, selected_manus, start_date, end_date):
        """Stacked parts by bucket (106 vs one brand or 'Others')."""
        dfm: Optional[pd.DataFrame] = getattr(app.server, "df_monthly", None)  # type: ignore[attr-defined]
        if dfm is None or not selected_types or not selected_manus:
            fig = go.Figure()
            fig.update_layout(template=PX_TEMPLATE, title=None, margin=dict(l=6, r=6, t=34, b=6))
            return fig

        filt = dfm[
            (dfm["gearbox_type"].isin(selected_types))
            & (dfm["gearbox_manufacturer_id"].isin(selected_manus))
        ].copy()
        if start_date:
            filt = filt[filt["month"] >= pd.to_datetime(start_date)]
        if end_date:
            filt = filt[filt["month"] <= pd.to_datetime(end_date)]
        if filt.empty:
            fig = go.Figure()
            fig.update_layout(template=PX_TEMPLATE, title=None, margin=dict(l=6, r=6, t=34, b=6))
            return fig

        non_106 = sorted([m for m in set(selected_manus) if str(m) != "106"])
        other_label = non_106[0] if len(non_106) == 1 else "Others"

        manu_series = filt["gearbox_manufacturer_id"].astype(str)
        filt["manufacturer_bucket"] = manu_series.where(manu_series == "106", other_label)

        defects = (
            filt.groupby(["manufacturer_bucket", "gearbox_type"], observed=True)["defective_count"]
            .sum()
            .reset_index()
            .rename(columns={"defective_count": "defects"})
        )
        denom = (
            filt.groupby(["manufacturer_bucket"], observed=True)["produced_count"]
            .sum()
            .reset_index()
            .rename(columns={"produced_count": "bucket_produced"})
        )
        merged = defects.merge(denom, on="manufacturer_bucket", how="left")

        if mode == "pct":
            denom_safe = merged["bucket_produced"].replace(0, pd.NA)
            merged["value"] = (merged["defects"] / denom_safe).fillna(0.0)
            y_title = "Defect rate"
        else:
            merged["value"] = merged["defects"]
            y_title = "Defective units"

        bucket_order = ["106"] + ([other_label] if other_label != "106" else [])
        merged["manufacturer_bucket"] = pd.Categorical(
            merged["manufacturer_bucket"], categories=bucket_order, ordered=True
        )

        fig = px.bar(
            merged.sort_values(["manufacturer_bucket", "gearbox_type"]),
            x="manufacturer_bucket",
            y="value",
            color="gearbox_type",
            barmode="stack",
            category_orders={
                "manufacturer_bucket": bucket_order,
                "gearbox_type": sorted(merged["gearbox_type"].astype(str).unique()),
            },
            color_discrete_map=PART_COLOR_MAP,
            labels={"manufacturer_bucket": "Manufacturer bucket", "value": y_title, "gearbox_type": "Part"},
            template=PX_TEMPLATE,
            pattern_shape=None,
        )
        fig.update_layout(
            title=None,
            legend_title_text="Component (gearbox type)",
            margin=dict(l=6, r=6, t=34, b=6),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        if mode == "pct":
            fig.update_yaxes(tickformat=".0%")
            hover_tmpl = "Bucket: %{x}<br>Component: %{customdata[0]}<br>Defect rate: %{y:.2%}<extra></extra>"
        else:
            fig.update_yaxes(separatethousands=True)
            hover_tmpl = "Bucket: %{x}<br>Component: %{customdata[0]}<br>Defects: %{y:,}<extra></extra>"
        fig.update_traces(hovertemplate=hover_tmpl, customdata=merged[["gearbox_type"]])
        return fig

    @app.callback(
        Output("failure-rate-line", "figure"),
        Input("ms-types", "value"),
        Input("ms-manus", "value"),
        Input("ms-daterange", "start_date"),
        Input("ms-daterange", "end_date"),
        prevent_initial_call=False,
    )
    def update_failure_rate_line(selected_types, selected_manus, start_date, end_date):
        """Monthly failure rate by manufacturer (lines)."""
        dfm: Optional[pd.DataFrame] = getattr(app.server, "df_monthly", None)  # type: ignore[attr-defined]
        if dfm is None or not selected_types or not selected_manus:
            fig = go.Figure()
            fig.update_layout(template=PX_TEMPLATE, title=None, margin=dict(l=6, r=6, t=34, b=6))
            return fig

        filt = dfm[
            (dfm["gearbox_type"].isin(selected_types))
            & (dfm["gearbox_manufacturer_id"].isin(selected_manus))
        ].copy()
        if start_date:
            filt = filt[filt["month"] >= pd.to_datetime(start_date)]
        if end_date:
            filt = filt[filt["month"] <= pd.to_datetime(end_date)]
        if filt.empty:
            fig = go.Figure()
            fig.update_layout(template=PX_TEMPLATE, title=None, margin=dict(l=6, r=6, t=34, b=6))
            return fig

        monthly = (
            filt.groupby(["month", "gearbox_manufacturer_id"], observed=True)[
                ["produced_count", "defective_count"]
            ]
            .sum()
            .reset_index()
        )
        denom = monthly["produced_count"].replace(0, pd.NA)
        monthly["failure_rate"] = (monthly["defective_count"] / denom).fillna(0.0)

        manus_unique = monthly["gearbox_manufacturer_id"].astype(str).unique().tolist()
        manus_order = (["106"] if "106" in manus_unique else []) + [
            m for m in sorted(manus_unique) if m != "106"
        ]
        color_map = get_color_map_for_manus(manus_order)

        fig = px.line(
            monthly.sort_values("month"),
            x="month",
            y="failure_rate",
            color="gearbox_manufacturer_id",
            category_orders={"gearbox_manufacturer_id": manus_order},
            color_discrete_map=color_map,
            labels={
                "month": "Production month",
                "failure_rate": "Failure rate",
                "gearbox_manufacturer_id": "Manufacturer",
            },
            markers=True,
            template=PX_TEMPLATE,
        )
        fig.update_layout(
            title=None,
            legend_title_text="Manufacturer",
            margin=dict(l=6, r=6, t=34, b=6),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        fig.update_yaxes(tickformat=".1%")
        fig.update_traces(
            hovertemplate=(
                "Month: %{x|%Y-%m}<br>"
                "Manufacturer: %{customdata[0]}<br>"
                "Failure rate: %{y:.2%}<br>"
                "Produced: %{customdata[1]:,}<br>"
                "Defects: %{customdata[2]:,}<extra></extra>"
            ),
            customdata=monthly[["gearbox_manufacturer_id", "produced_count", "defective_count"]],
        )
        return fig

    @app.callback(
        Output("km-survival", "figure"),
        Input("ms-types", "value"),
        Input("ms-manus", "value"),
        Input("ms-daterange", "start_date"),
        Input("ms-daterange", "end_date"),
        prevent_initial_call=False,
    )
    def update_km_survival(selected_types, selected_manus, start_date, end_date):
        """KM survival per (manufacturer, gearbox_type)."""
        dfa: Optional[pd.DataFrame] = getattr(app.server, "df_all", None)  # type: ignore[attr-defined]
        if dfa is None or not selected_types or not selected_manus:
            fig = go.Figure()
            fig.update_layout(template=PX_TEMPLATE, title=None, margin=dict(l=6, r=6, t=34, b=6))
            return fig

        df = dfa[
            (dfa["gearbox_type"].isin(selected_types))
            & (dfa["gearbox_manufacturer_id"].isin(selected_manus))
        ].copy()

        if start_date:
            df = df[df["gearbox_production_date"] >= pd.to_datetime(start_date)]
        obs_end = pd.to_datetime(end_date) if end_date else pd.to_datetime(df["gearbox_production_date"].max())

        if df.empty:
            fig = go.Figure()
            fig.update_layout(template=PX_TEMPLATE, title=None, margin=dict(l=6, r=6, t=34, b=6))
            return fig

        prod_dt = df["gearbox_production_date"]
        fail_dt = df["gearbox_defective_date"]
        has_event = (df["gearbox_defective_flag"] == 1) & fail_dt.notna() & (fail_dt <= obs_end)

        end_ts = pd.Series(obs_end, index=df.index)
        end_ts.loc[has_event] = fail_dt.loc[has_event]
        times_days = (end_ts - prod_dt).dt.days.clip(lower=0)

        df["km_time_days"] = times_days
        df["km_event"] = has_event
        df = df.dropna(subset=["km_time_days"])
        df = df[df["km_time_days"] >= 0]
        if df.empty:
            fig = go.Figure()
            fig.update_layout(template=PX_TEMPLATE, title=None, margin=dict(l=6, r=6, t=34, b=6))
            return fig

        manus_unique = df["gearbox_manufacturer_id"].astype(str).unique().tolist()
        manus_order = (["106"] if "106" in manus_unique else []) + [
            m for m in sorted(manus_unique) if m != "106"
        ]
        color_map = get_color_map_for_manus(manus_order)
        dash_map = {"K3AG1": "solid", "K3SG1": "dot"}

        fig = go.Figure()
        for (manu, gtype), gdf in df.groupby(["gearbox_manufacturer_id", "gearbox_type"], observed=True):
            if gdf.empty:
                continue
            t, S, _ = km_curve(gdf["km_time_days"], gdf["km_event"])
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=S,
                    mode="lines",
                    line=dict(color=color_map.get(str(manu), "#999999"), dash=dash_map.get(str(gtype), "solid")),
                    name=f"{manu} • {gtype}",
                    hovertemplate="t (days): %{x}<br>S(t): %{y:.2%}<extra></extra>",
                )
            )

        fig.update_layout(
            template=PX_TEMPLATE,
            title=None,
            xaxis_title="Lifetime (days)",
            yaxis_title="Survival S(t)",
            margin=dict(l=6, r=6, t=34, b=6),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend_title_text="Manufacturer • Gear Box",
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        return fig

    @app.callback(
        Output("mileage-hist", "figure"),
        Input("ms-types", "value"),
        Input("ms-manus", "value"),
        Input("ms-daterange", "start_date"),
        Input("ms-daterange", "end_date"),
        prevent_initial_call=False,
    )
    def update_mileage_hist(selected_types, selected_manus, start_date, end_date):
        """Histogram of failure mileage (stacked by manufacturer)."""
        dfa: Optional[pd.DataFrame] = getattr(app.server, "df_all", None)  # type: ignore[attr-defined]
        if dfa is None or not selected_types or not selected_manus:
            fig = go.Figure()
            fig.update_layout(template=PX_TEMPLATE, title=None, margin=dict(l=6, r=6, t=34, b=6))
            return fig

        df = dfa[
            (dfa["gearbox_defective_flag"] == 1)
            & (dfa["gearbox_defective_mileage"].notna())
            & (dfa["gearbox_type"].isin(selected_types))
            & (dfa["gearbox_manufacturer_id"].isin(selected_manus))
        ].copy()

        if start_date:
            df = df[df["gearbox_production_date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["gearbox_production_date"] <= pd.to_datetime(end_date)]
        if df.empty:
            fig = go.Figure()
            fig.update_layout(template=PX_TEMPLATE, title=None, margin=dict(l=6, r=6, t=34, b=6))
            return fig

        df = df[df["gearbox_defective_mileage"] >= 0]

        manus_unique = df["gearbox_manufacturer_id"].astype(str).unique().tolist()
        manus_order = (["106"] if "106" in manus_unique else []) + [
            m for m in sorted(manus_unique) if m != "106"
        ]
        color_map = get_color_map_for_manus(manus_order)

        fig = px.histogram(
            df,
            x="gearbox_defective_mileage",
            color="gearbox_manufacturer_id",
            nbins=40,
            barmode="stack",
            histnorm=None,
            category_orders={"gearbox_manufacturer_id": manus_order},
            color_discrete_map=color_map,
            labels={"gearbox_defective_mileage": "Failure mileage(km)", "gearbox_manufacturer_id": "Manufacturer"},
            template=PX_TEMPLATE,
        )
        fig.update_layout(
            title=None,
            margin=dict(l=6, r=6, t=34, b=6),
            bargap=0.02,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend_title_text="Manufacturer",
        )
        fig.update_yaxes(title_text="Units", separatethousands=True)
        return fig

    @app.callback(
        Output("support-table", "data"),
        Input("ms-types", "value"),
        Input("ms-manus", "value"),
        Input("ms-daterange", "start_date"),
        Input("ms-daterange", "end_date"),
        prevent_initial_call=False,
    )
    def update_support_table(selected_types, selected_manus, start_date, end_date):
        """Filtered rows for the table; includes vehicle_id and vehicle_type."""
        dfa: Optional[pd.DataFrame] = getattr(app.server, "df_all", None)  # type: ignore[attr-defined]
        if dfa is None or not selected_types or not selected_manus:
            return []

        df = dfa[
            (dfa["gearbox_type"].isin(selected_types))
            & (dfa["gearbox_manufacturer_id"].isin(selected_manus))
        ].copy()

        if start_date:
            df = df[df["gearbox_production_date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["gearbox_production_date"] <= pd.to_datetime(end_date)]

        cols = [
            "gearbox_id",
            "vehicle_id",
            "vehicle_type",
            "gearbox_manufacturer_id",
            "gearbox_plant_id",
            "gearbox_type",
            "gearbox_production_date",
            "gearbox_defective_flag",
            "gearbox_defective_date",
            "gearbox_defective_mileage",
        ]
        df = df.loc[:, cols].copy().sort_values("gearbox_production_date", ascending=False)

        if len(df) > TABLE_ROW_CAP:
            df = df.head(TABLE_ROW_CAP)

        for c in ["gearbox_production_date", "gearbox_defective_date"]:
            if c in df.columns:
                df[c] = df[c].dt.strftime("%Y-%m-%d")

        app.server.filtered_table_df = df  # type: ignore[attr-defined]
        return df.where(pd.notna(df), None).to_dict("records")

    @app.callback(
        Output("download-table", "data"),
        Input("btn-export", "n_clicks"),
        prevent_initial_call=True,
    )
    def export_filtered_table(n_clicks):
        """Export the current filtered table as CSV."""
        dff: Optional[pd.DataFrame] = getattr(app.server, "filtered_table_df", None)  # type: ignore[attr-defined]
        if dff is None or dff.empty:
            return dash.no_update
        return dcc.send_data_frame(dff.to_csv, "k3_series_filtered.csv", index=False)

    return app


app = create_app()


if __name__ == "__main__":
    # dev server with hot reload
    app.run(
        host="127.0.0.1",
        port=8050,
        debug=True,
       
    )
