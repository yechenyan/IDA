"""
Dash app for company '106' — product 'K3AG1'

Now includes:
1) Stacked market share over time (bar) with filters (mode abs/pct, types, manufacturers, date).
   - '106' always stacked at the bottom.
   - Colors: 106 vivid blue; others low-saturation warm tones.
2) Error frequency of parts (bar) comparing '106' vs 'Others' (uses defective_flag).
   - Absolute: defective counts; Percentage: defect rate per bucket stacked by part.
3) NEW: Failure rate over time (line) — interactive overview comparing 106 vs competitors.
   - Monthly failure rate = defective_sum / produced_sum
   - Filters: date range, gearbox types, manufacturers.

Other:
- Corporate light-blue theme; font: Source Sans Pro; given logo.
- PEP 8; debug=True; hot reload; app.run(...).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List

import dash
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# ---------- Constants & Theme ----------

# Corporate palette (light-blue centric)
CORPORATE_LIGHT_BLUE = "#d9ecff"          # page background
CORPORATE_LIGHT_BLUE_ACCENT = "#5aa3ff"   # header gradient start
CORPORATE_BLUE_DEEP = "#1d6fff"           # header gradient end
TEXT_DARK = "#0f172a"
CARD_BG = "#ffffff"
CARD_BORDER = "#cbd5e1"

DATA_FILE = "final_dataset_group_23.csv"

EXTERNAL_STYLESHEETS = [
    "https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap"  # noqa: E501
]

LOGO_URL = "https://dev.dappso.com/_next/static/media/dappSoLogoText.05aa8b23.png"
PX_TEMPLATE = "plotly_white"

# Manufacturer colors (market share chart): 106 vivid blue; others low-saturation warm tones
BASE_COLOR_MAP: Dict[str, str] = {
    "106": "#1d6fff",  # vivid blue for our company
    "105": "#f2cc8f",  # light amber
    "107": "#f4b183",  # light orange
    "108": "#f7d9a8",  # light sand
}
WARM_FALLBACK: List[str] = [
    "#f2cc8f", "#f4b183", "#f7d9a8", "#eddcc8", "#ffe0b2", "#ffe5c4"
]

# Part colors (error chart): soft & distinct
PART_COLOR_MAP: Dict[str, str] = {
    "K3AG1": "#7fb3ff",  # soft blue
    "K3SG1": "#ffd6a5",  # soft warm
}


# ---------- Data Loading & KPI ----------

def load_dataset_for_kpi(path: Path) -> pd.DataFrame:
    """Load minimal columns for computing the slogan KPI (x)."""
    df = pd.read_csv(
        path,
        usecols=["gearbox_manufacturer_id"],
        dtype={"gearbox_manufacturer_id": "category"},
    )
    return df


def load_dataset_for_aggregations(path: Path) -> pd.DataFrame:
    """
    Load columns needed by all charts.

    Columns:
      - gearbox_manufacturer_id
      - gearbox_type
      - gearbox_production_date
      - gearbox_defective_flag
    """
    df = pd.read_csv(
        path,
        usecols=[
            "gearbox_manufacturer_id",
            "gearbox_type",
            "gearbox_production_date",
            "gearbox_defective_flag",
        ],
        dtype={
            "gearbox_manufacturer_id": "category",
            "gearbox_type": "category",
            "gearbox_defective_flag": "int8",
        },
        parse_dates=["gearbox_production_date"],
        infer_datetime_format=True,
    )
    return df


def compute_x_for_slogan(df: pd.DataFrame) -> Optional[int]:
    """
    Compute 'x' for the slogan:
    'there is a part of 106 in every x gearbox'

    x = round(total_gearboxes / gearboxes_from_106)
    Return None if denominator is zero.
    """
    total = len(df)
    count_106 = (df["gearbox_manufacturer_id"] == "106").sum()
    if count_106 == 0 or total == 0:
        return None
    x_float = total / count_106
    return max(1, int(x_float + 0.5))


def preaggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-aggregate to monthly counts for faster callbacks.

    Output columns:
      - month (Timestamp at month-end)
      - gearbox_manufacturer_id (e.g., '105', '106', '107', '108')
      - gearbox_type ('K3AG1' / 'K3SG1')
      - produced_count (units produced that month)
      - defective_count (among those produced that month, defective_flag==1)
    """
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
    """
    Build a color map ensuring '106' vivid blue, others warm and low-saturation.
    Any unseen manufacturer will be assigned from the fallback list.
    """
    cmap = {}
    used = set()

    for m in manus:
        if m in BASE_COLOR_MAP:
            cmap[m] = BASE_COLOR_MAP[m]
            used.add(BASE_COLOR_MAP[m])

    fallback_iter = (c for c in WARM_FALLBACK if c not in used)
    for m in manus:
        if m not in cmap:
            try:
                cmap[m] = next(fallback_iter)
            except StopIteration:
                cmap[m] = WARM_FALLBACK[-1]

    return cmap


# ---------- App Factory ----------

def create_app() -> Dash:
    """Create and configure the Dash app."""
    app = dash.Dash(
        __name__,
        external_stylesheets=EXTERNAL_STYLESHEETS,
        suppress_callback_exceptions=True,
        title="106 • K3AG1 Quality Dashboard",
    )

    # Custom index string to enforce global font and base colors
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
                    --page-bg: {CORPORATE_LIGHT_BLUE};
                    --card-bg: {CARD_BG};
                    --card-border: {CARD_BORDER};
                    --text-dark: {TEXT_DARK};
                    --accent-start: {CORPORATE_LIGHT_BLUE_ACCENT};
                    --accent-end: {CORPORATE_BLUE_DEEP};
                }}
                html, body {{
                    height: 100%;
                    background: var(--page-bg);
                    color: var(--text-dark);
                    font-family: "Source Sans Pro", system-ui, -apple-system,
                                 "Segoe UI", Roboto, "Helvetica Neue", Arial,
                                 "Noto Sans", "Apple Color Emoji",
                                 "Segoe UI Emoji", "Segoe UI Symbol",
                                 "Noto Color Emoji";
                }}
                .card {{
                    background: var(--card-bg);
                    border: 1px solid var(--card-border);
                    border-radius: 16px;
                    padding: 16px 20px;
                    box-shadow: 0 8px 24px rgba(2, 6, 23, 0.06);
                }}
                .header {{
                    background: linear-gradient(90deg, var(--accent-start), var(--accent-end));
                    color: white;
                    padding: 18px 22px;
                    border-radius: 16px;
                    display: flex;
                    align-items: center;
                    gap: 16px;
                }}
                .header h1 {{
                    font-size: 28px;
                    font-weight: 700;
                    margin: 0;
                    letter-spacing: 0.2px;
                }}
                .header-sub {{
                    font-size: 15px;
                    opacity: 0.92;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 24px auto 56px auto;
                    padding: 0 16px;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: 1fr;
                    gap: 16px;
                }}
                .pill {{
                    display: inline-block;
                    padding: 6px 10px;
                    border-radius: 999px;
                    background: rgba(255,255,255,0.18);
                    color: #ffffff;
                    font-size: 12px;
                }}
                .slogan {{
                    font-size: 20px;
                    font-weight: 600;
                    margin: 4px 0 0 0;
                }}
                .logo {{
                    height: 32px;
                    width: auto;
                    filter: brightness(0) invert(1);
                }}
                .toolbar {{
                    display: grid;
                    grid-template-columns: auto auto auto 1fr;
                    gap: 14px 18px;
                    align-items: center;
                    margin-bottom: 10px;
                }}
                .toolbar .label {{
                    font-weight: 600;
                    font-size: 14px;
                    margin-right: 4px;
                }}
                .toolbar .group {{
                    display: flex;
                    flex-wrap: wrap;
                    align-items: center;
                    gap: 10px 14px;
                }}
                @media (max-width: 900px) {{
                    .toolbar {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    """

    # Load data for slogan
    data_error: Optional[str] = None
    x_value: Optional[int] = None
    try:
        df_kpi = load_dataset_for_kpi(Path(DATA_FILE))
        x_value = compute_x_for_slogan(df_kpi)
    except Exception as exc:  # pragma: no cover
        data_error = str(exc)

    slogan_text = (
        "there is a part of 106 in every x gearbox"
        if x_value is None
        else f"there is a part of 106 in every {x_value} gearbox"
    )

    # Preload + preaggregate data for charts
    market_error: Optional[str] = None
    manu_options: List[str] = []
    default_type_selection: List[str] = ["K3AG1"]  # default ONLY K3AG1
    min_date = None
    max_date = None
    try:
        df_all = load_dataset_for_aggregations(Path(DATA_FILE))
        df_monthly = preaggregate_monthly(df_all)
        app.server.df_monthly = df_monthly  # type: ignore[attr-defined]

        manu_options = sorted(df_all["gearbox_manufacturer_id"].astype(str).unique())
        min_date = pd.to_datetime(df_all["gearbox_production_date"].min())
        max_date = pd.to_datetime(df_all["gearbox_production_date"].max())
    except Exception as exc:  # pragma: no cover
        market_error = str(exc)

    # Fallbacks if loading failed
    manu_options = manu_options or ["105", "106", "107", "108"]
    if min_date is None or pd.isna(min_date):
        min_date = pd.Timestamp("2008-11-12")
    if max_date is None or pd.isna(max_date):
        max_date = pd.Timestamp("2016-11-15")

    # ---------- Layout ----------

    app.layout = html.Div(
        className="container",
        children=[
            # Header with logo and title
            html.Div(
                className="header",
                children=[
                    html.Img(
                        src=LOGO_URL,
                        alt="106 — Quality Dashboard",
                        className="logo",
                        title="106 • Quality Dashboard",
                    ),
                    html.Div(
                        children=[
                            html.H1("K3AG1 Quality & Market Intelligence"),
                            html.Div("OEM1 • Types: Type11 / Type12", className="header-sub"),
                            html.Span("Internal dashboard prototype", className="pill"),
                        ]
                    ),
                ],
            ),

            # Controls + Market share chart
            html.Div(
                className="grid",
                children=[
                    html.Div(
                        className="card",
                        children=[
                            html.Div(
                                className="toolbar",
                                children=[
                                    html.Div(
                                        className="group",
                                        children=[
                                            html.Span("Display mode:", className="label"),
                                            dcc.RadioItems(
                                                id="ms-mode",
                                                options=[
                                                    {"label": "Absolute", "value": "abs"},
                                                    {"label": "Percentage", "value": "pct"},
                                                ],
                                                value="abs",
                                                inputStyle={"marginRight": "6px"},
                                                labelStyle={"marginRight": "14px"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="group",
                                        children=[
                                            html.Span("Gearbox types:", className="label"),
                                            dcc.Checklist(
                                                id="ms-types",
                                                options=[
                                                    {"label": "K3AG1 (our auto)", "value": "K3AG1"},
                                                    {"label": "K3SG1 (manual/other)", "value": "K3SG1"},
                                                ],
                                                value=default_type_selection,
                                                inputStyle={"marginRight": "6px"},
                                                labelStyle={"marginRight": "14px"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="group",
                                        children=[
                                            html.Span("Manufacturers:", className="label"),
                                            dcc.Checklist(
                                                id="ms-manus",
                                                options=[
                                                    {"label": ("106 (us)" if m == "106" else m), "value": m}
                                                    for m in manu_options
                                                ],
                                                value=manu_options,  # default select all
                                                inputStyle={"marginRight": "6px"},
                                                labelStyle={"marginRight": "12px"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="group",
                                        children=[
                                            html.Span("Date range:", className="label"),
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
                            dcc.Graph(
                                id="market-share-chart",
                                config={"displayModeBar": True},
                                figure=px.bar(title="Market share (stacked) — loading..."),
                                style={"height": "480px"},
                            ),
                            html.Br(),
                            html.Div(
                                [
                                    html.Span(
                                        "Advertising slogan:",
                                        style={"fontSize": "14px", "opacity": 0.9, "marginRight": "8px"},
                                    ),
                                    html.Div(slogan_text, className="slogan"),
                                ]
                            ),
                        ],
                    ),
                ],
            ),

            # Error frequency (defect rate composition) chart
            html.Div(
                className="grid",
                children=[
                    html.Div(
                        className="card",
                        children=[
                            html.Div(
                                "Error frequency of parts — 106 vs Others",
                                style={"fontWeight": 700, "marginBottom": "6px"},
                            ),
                            dcc.Graph(
                                id="error-frequency-chart",
                                config={"displayModeBar": True},
                                figure=px.bar(title="Error frequency — loading..."),
                                style={"height": "420px"},
                            ),
                            html.Div(
                                "Absolute uses defective counts (flag==1). Percentage shows defect rate per bucket.",
                                style={"fontSize": "12px", "opacity": 0.75, "marginTop": "6px"},
                            ),
                        ],
                    ),
                ],
            ),

            # NEW: Failure rate over time (line) — 106 vs competitors
            html.Div(
                className="grid",
                children=[
                    html.Div(
                        className="card",
                        children=[
                            html.Div(
                                "Failure rate over time — 106 vs competitors",
                                style={"fontWeight": 700, "marginBottom": "6px"},
                            ),
                            dcc.Graph(
                                id="failure-rate-line",
                                config={"displayModeBar": True},
                                figure=px.line(title="Failure rate — loading..."),
                                style={"height": "440px"},
                            ),
                            html.Div(
                                "Monthly failure rate = defective_sum / produced_sum within selected filters.",
                                style={"fontSize": "12px", "opacity": 0.75, "marginTop": "6px"},
                            ),
                        ],
                    ),
                ],
            ),

            # Data error surfaces
            html.Div(
                className="card",
                style={
                    "marginTop": "16px",
                    "display": "block" if data_error else "none",
                    "borderColor": "#fecaca",
                },
                children=[
                    html.Div(
                        "Dataset load warning (KPI)",
                        style={"fontWeight": 700, "marginBottom": "8px", "color": "#b91c1c"},
                    ),
                    html.Div(data_error or "", style={"fontSize": "14px", "whiteSpace": "pre-wrap"}),
                ],
            ),
            html.Div(
                className="card",
                style={
                    "marginTop": "16px",
                    "display": "block" if market_error else "none",
                    "borderColor": "#fecaca",
                },
                children=[
                    html.Div(
                        "Dataset load warning (Aggregations)",
                        style={"fontWeight": 700, "marginBottom": "8px", "color": "#b91c1c"},
                    ),
                    html.Div(market_error or "", style={"fontSize": "14px", "whiteSpace": "pre-wrap"}),
                ],
            ),
        ],
    )

    # ---------- Callbacks ----------

    @app.callback(
        Output("market-share-chart", "figure"),
        Input("ms-mode", "value"),
        Input("ms-types", "value"),
        Input("ms-manus", "value"),
        Input("ms-daterange", "start_date"),
        Input("ms-daterange", "end_date"),
        prevent_initial_call=False,
    )
    def update_market_share_chart(
        mode: str,
        selected_types: list[str],
        selected_manus: list[str],
        start_date: str,
        end_date: str,
    ):
        """Stacked market share over time; '106' always at the bottom."""
        dfm: Optional[pd.DataFrame] = getattr(app.server, "df_monthly", None)  # type: ignore[attr-defined]
        if dfm is None or not selected_types or not selected_manus:
            fig = px.bar(title="Market share (stacked)")
            fig.update_layout(template=PX_TEMPLATE)
            return fig

        # Filter
        filt = dfm[
            (dfm["gearbox_type"].isin(selected_types))
            & (dfm["gearbox_manufacturer_id"].isin(selected_manus))
        ].copy()
        if start_date:
            filt = filt[filt["month"] >= pd.to_datetime(start_date)]
        if end_date:
            filt = filt[filt["month"] <= pd.to_datetime(end_date)]

        if len(filt) == 0:
            fig = px.bar(title="Market share (stacked) — no data for current filters")
            fig.update_layout(template=PX_TEMPLATE)
            return fig

        # Value
        if mode == "pct":
            totals = (
                filt.groupby("month", as_index=False)["produced_count"]
                .sum()
                .rename(columns={"produced_count": "total"})
            )
            merged = filt.merge(totals, on="month", how="left")
            merged["value"] = merged["produced_count"] / merged["total"]
            y_title = "Share (%)"
        else:
            merged = filt.copy()
            merged["value"] = merged["produced_count"]
            y_title = "Units"

        # Order & colors
        manus_unique = merged["gearbox_manufacturer_id"].astype(str).unique().tolist()
        manus_order = (["106"] if "106" in manus_unique else []) + [
            m for m in sorted(manus_unique) if m != "106"
        ]
        color_map = get_color_map_for_manus(manus_order)

        # Figure
        fig = px.bar(
            merged.sort_values("month"),
            x="month",
            y="value",
            color="gearbox_manufacturer_id",
            barmode="stack",
            category_orders={"gearbox_manufacturer_id": manus_order},
            color_discrete_map=color_map,
            labels={
                "month": "Production month",
                "value": y_title,
                "gearbox_manufacturer_id": "Manufacturer",
            },
            template=PX_TEMPLATE,
        )

        fig.update_layout(
            title=("Market share (stacked) — " + ("percentage" if mode == "pct" else "absolute")),
            legend_title_text="Manufacturer",
            margin=dict(l=10, r=10, t=48, b=10),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        if mode == "pct":
            fig.update_yaxes(tickformat=".0%")
            hover_tmpl = (
                "Month: %{x|%Y-%m}<br>Manufacturer: %{customdata[0]}<br>Share: %{y:.1%}<extra></extra>"
            )
        else:
            fig.update_yaxes(separatethousands=True)
            hover_tmpl = (
                "Month: %{x|%Y-%m}<br>Manufacturer: %{customdata[0]}<br>Units: %{y:,}<extra></extra>"
            )

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
    def update_error_frequency_chart(
        mode: str,
        selected_types: list[str],
        selected_manus: list[str],
        start_date: str,
        end_date: str,
    ):
        """
        Error frequency (defect rate composition) chart: 106 vs Others.

        Absolute:
            y = defective_count_part  (sum where gearbox_defective_flag == 1)
        Percentage (defect rate):
            y_part = defective_count_part / produced_count_bucket_total
        """
        dfm: Optional[pd.DataFrame] = getattr(app.server, "df_monthly", None)  # type: ignore[attr-defined]
        if dfm is None or not selected_types or not selected_manus:
            fig = px.bar(title="Error frequency (parts)")
            fig.update_layout(template=PX_TEMPLATE)
            return fig

        # Filter by controls
        filt = dfm[
            (dfm["gearbox_type"].isin(selected_types))
            & (dfm["gearbox_manufacturer_id"].isin(selected_manus))
        ].copy()
        if start_date:
            filt = filt[filt["month"] >= pd.to_datetime(start_date)]
        if end_date:
            filt = filt[filt["month"] <= pd.to_datetime(end_date)]

        if len(filt) == 0:
            fig = px.bar(title="Error frequency (parts) — no data for current filters")
            fig.update_layout(template=PX_TEMPLATE)
            return fig

        # Manufacturer bucket: '106' vs 'Others'
        manu_series = filt["gearbox_manufacturer_id"].astype(str)
        filt["manufacturer_bucket"] = manu_series.where(manu_series == "106", "Others")

        # Numerator: defects per (bucket, part)
        defects = (
            filt.groupby(["manufacturer_bucket", "gearbox_type"], observed=True)["defective_count"]
            .sum()
            .reset_index()
            .rename(columns={"defective_count": "defects"})
        )

        # Denominator: total produced per bucket (across selected parts)
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
            y_title = "Defect rate (%)"
        else:
            merged["value"] = merged["defects"]
            y_title = "Defective units"

        bucket_order = ["106", "Others"]
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
            labels={
                "manufacturer_bucket": "Manufacturer bucket",
                "value": y_title,
                "gearbox_type": "Part (gearbox type)",
            },
            template=PX_TEMPLATE,
        )

        fig.update_layout(
            title=("Error frequency of parts — " + ("percentage (defect rate)" if mode == "pct" else "absolute")),
            legend_title_text="Part (gearbox type)",
            margin=dict(l=10, r=10, t=48, b=10),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )

        if mode == "pct":
            fig.update_yaxes(tickformat=".0%")
            hover_tmpl = "Bucket: %{x}<br>Part: %{customdata[0]}<br>Defect rate: %{y:.2%}<extra></extra>"
        else:
            fig.update_yaxes(separatethousands=True)
            hover_tmpl = "Bucket: %{x}<br>Part: %{customdata[0]}<br>Defects: %{y:,}<extra></extra>"

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
    def update_failure_rate_line(
        selected_types: list[str],
        selected_manus: list[str],
        start_date: str,
        end_date: str,
    ):
        """
        Failure rate over time (monthly time-series line):
        failure_rate(month, manufacturer) = defective_sum / produced_sum
        """
        dfm: Optional[pd.DataFrame] = getattr(app.server, "df_monthly", None)  # type: ignore[attr-defined]
        if dfm is None or not selected_types or not selected_manus:
            fig = px.line(title="Failure rate over time")
            fig.update_layout(template=PX_TEMPLATE)
            return fig

        # Filter by controls
        filt = dfm[
            (dfm["gearbox_type"].isin(selected_types))
            & (dfm["gearbox_manufacturer_id"].isin(selected_manus))
        ].copy()
        if start_date:
            filt = filt[filt["month"] >= pd.to_datetime(start_date)]
        if end_date:
            filt = filt[filt["month"] <= pd.to_datetime(end_date)]

        if len(filt) == 0:
            fig = px.line(title="Failure rate over time — no data for current filters")
            fig.update_layout(template=PX_TEMPLATE)
            return fig

        # Aggregate to month x manufacturer
        monthly = (
            filt.groupby(["month", "gearbox_manufacturer_id"], observed=True)[
                ["produced_count", "defective_count"]
            ]
            .sum()
            .reset_index()
        )
        # Compute failure rate safely
        denom = monthly["produced_count"].replace(0, pd.NA)
        monthly["failure_rate"] = (monthly["defective_count"] / denom).fillna(0.0)

        # Order & colors (ensure 106 is first)
        manus_unique = monthly["gearbox_manufacturer_id"].astype(str).unique().tolist()
        manus_order = (["106"] if "106" in manus_unique else []) + [
            m for m in sorted(manus_unique) if m != "106"
        ]
        color_map = get_color_map_for_manus(manus_order)

        # Line chart
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
            title="Failure rate over time — monthly (106 vs competitors)",
            legend_title_text="Manufacturer",
            margin=dict(l=10, r=10, t=48, b=10),
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

    return app


# ---------- Entrypoint ----------

app = create_app()

if __name__ == "__main__":
    # Debugging enabled and hot reloading active.
    # Using app.run(...) (not app.run_server) per request.
    app.run(
        host="0.0.0.0",
        port=8050,
        debug=True,
        dev_tools_hot_reload=True,
        dev_tools_silence_routes_logging=True,
        use_reloader=True,
    )
