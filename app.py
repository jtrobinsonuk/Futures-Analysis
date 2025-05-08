import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import cot_reports as cot
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.decomposition import PCA

DEFAULT_REPO = 0.05
FUTURES_SYMBOLS = {
    '2Y': 'ZT=F',
    '5Y': 'ZF=F',
    '7Y': 'ZN=F',  # You may want to update this symbol if CME changes it for 7Y
    '10Y': 'TN=F',
    '30Y': 'ZB=F'
}

# ---- HARDCODED CONTRACT NAMES ----
CONTRACT_NAME_FILTERS = {
    '2Y': [
        '2-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE',
        'UST 2Y NOTE - CHICAGO BOARD OF TRADE'
    ],
    '5Y': [
        '5-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE',
        'UST 5Y NOTE - CHICAGO BOARD OF TRADE'
    ],
    '7Y': [
        '10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE',
        'UST 10Y NOTE - CHICAGO BOARD OF TRADE'
    ],
    '10Y': [
        'ULTRA 10-YEAR U.S. T-NOTES - CHICAGO BOARD OF TRADE',
        'ULTRA UST 10Y - CHICAGO BOARD OF TRADE'
    ],
    '30Y': [
        'U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE',
        'LONG-TERM U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE',
        'ULTRA U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE',
        'ULTRA UST BOND - CHICAGO BOARD OF TRADE'
    ]
}

def fetch_cot_data():
    df = cot.cot_all(cot_report_type='traders_in_financial_futures_fut')
    df['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(
        df['Report_Date_as_YYYY-MM-DD'], errors='coerce'
    )
    return df.sort_values('Report_Date_as_YYYY-MM-DD')

def fetch_futures_prices():
    data = {}
    for k, v in FUTURES_SYMBOLS.items():
        try:
            ticker = yf.Ticker(v)
            hist = ticker.history(period="7d")
            data[k] = hist['Close'].iloc[-1] if not hist.empty else np.nan
        except Exception:
            data[k] = np.nan
    return data

def fetch_price_series(contract, period="5y"):
    try:
        ticker = yf.Ticker(FUTURES_SYMBOLS[contract])
        hist = ticker.history(period=period)
        if not hist.empty:
            return hist['Close'].resample('W').last()
    except Exception:
        pass
    return pd.Series([], dtype='float64')

def compute_zscores(series, window=52):
    if series.isnull().all() or len(series) < window:
        return pd.Series([np.nan]*len(series), index=series.index)
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def compute_signals(df, contract_names):
    signals = {}
    for label, name_list in contract_names.items():
        sub = df[df['Market_and_Exchange_Names'].isin(name_list)].sort_values('Report_Date_as_YYYY-MM-DD')
        # --- AGGREGATE BY DATE ---
        sub_grouped = sub.groupby('Report_Date_as_YYYY-MM-DD').agg({
            'Lev_Money_Positions_Long_All': 'sum',
            'Lev_Money_Positions_Short_All': 'sum'
        }).reset_index()
        if sub_grouped.empty or len(sub_grouped) < 30:
            signals[label] = pd.DataFrame({
                'date': [],
                'net_lev': [],
                'zscore': [],
                'adaptive_z': [],
                'momentum': [],
                'cs_zscore': []
            })
            continue
        net_lev = sub_grouped['Lev_Money_Positions_Long_All'] - sub_grouped['Lev_Money_Positions_Short_All']
        dates = sub_grouped['Report_Date_as_YYYY-MM-DD']
        z = compute_zscores(net_lev)
        returns = net_lev.pct_change().rolling(52).std().fillna(0)
        window = (52 * (1 + returns)).clip(lower=26, upper=104).fillna(26).astype(int)
        adaptive_z = []
        for i in range(len(net_lev)):
            win = window.iloc[i] if i < len(window) else 26
            win = int(win) if win >= 1 else 26
            if i >= win - 1:
                window_slice = net_lev.iloc[i - win + 1:i + 1]
                std = window_slice.std()
                adaptive_z.append((window_slice.iloc[-1] - window_slice.mean()) / std if std > 0 else np.nan)
            else:
                adaptive_z.append(np.nan)
        momentum = net_lev.diff(4)
        signals[label] = pd.DataFrame({
            'date': dates.values,
            'net_lev': net_lev.values,
            'zscore': z.values,
            'adaptive_z': adaptive_z,
            'momentum': momentum.values
        })
    # Cross-sectional z-score
    all_dates = sorted(set().union(*[set(signals[k]['date']) for k in signals if not signals[k].empty]))
    for k in signals:
        if signals[k].empty:
            continue
        sig = signals[k]
        cs_z = []
        for d in sig['date']:
            vals = []
            for kk in signals:
                if signals[kk].empty:
                    continue
                match_rows = signals[kk][signals[kk]['date'] == d]
                if not match_rows.empty:
                    vals.append(match_rows['net_lev'].values[0])
            if vals:
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                val = sig[sig['date'] == d]['net_lev'].values[0]
                cs_z.append((val - mean) / std if std > 0 else np.nan)
            else:
                cs_z.append(np.nan)
        signals[k]['cs_zscore'] = cs_z
    return signals

def compute_corr_heatmap(signals):
    df_corr = pd.DataFrame({k: signals[k]['net_lev'] for k in signals if not signals[k].empty})
    return df_corr.corr()

def enhanced_trading_insight(contract, sig, price_series):
    if sig.empty or len(sig) == 0:
        return html.Div("No data available for analysis.", style={'color': 'gray', 'background': '#f8f8f8', 'padding': '10px', 'borderRadius': '6px', 'marginBottom': '30px'})
    recent_data = sig.iloc[-8:] if len(sig) >= 8 else sig
    z = sig['adaptive_z'].iloc[-1] if not pd.isna(sig['adaptive_z'].iloc[-1]) else 0
    net = sig['net_lev'].iloc[-1] if not pd.isna(sig['net_lev'].iloc[-1]) else 0
    cs_z = sig['cs_zscore'].iloc[-1] if not pd.isna(sig['cs_zscore'].iloc[-1]) else 0
    mom = sig['momentum'].iloc[-1] if not pd.isna(sig['momentum'].iloc[-1]) else 0
    if len(recent_data) >= 2:
        recent_trend = np.polyfit(range(len(recent_data)), recent_data['net_lev'].values, 1)[0]
    else:
        recent_trend = 0
    price = price_series.iloc[-1] if not price_series.empty else None
    if z < -0.8 and recent_trend < -500:
        color = '#ffcccc'
        msg = (f"Bearish Signal for {contract}: Short-term positioning trend is accelerating (z={z:.2f}). "
               f"Net position decreasing by {abs(recent_trend):.0f} contracts per week. "
               f"Watch for potential short-covering rally above ${price*1.005:.2f}." if price else "")
    elif z > 0.8 and recent_trend > 500:
        color = '#ccffcc'
        msg = (f"Bullish Signal for {contract}: Short-term positioning trend is strengthening (z={z:.2f}). "
               f"Net position increasing by {recent_trend:.0f} contracts per week. "
               f"Be cautious of profit-taking below ${price*0.995:.2f}." if price else "")
    elif recent_trend > 1000:
        color = '#e6ffe6'
        msg = (f"Momentum Signal for {contract}: Rapid positioning change detected. "
               f"Net position trending up by {recent_trend:.0f} contracts per week. "
               f"Potential tactical long opportunity forming.")
    elif recent_trend < -1000:
        color = '#ffe6e6'
        msg = (f"Momentum Signal for {contract}: Rapid positioning change detected. "
               f"Net position trending down by {abs(recent_trend):.0f} contracts per week. "
               f"Potential tactical short opportunity forming.")
    elif abs(z) > 1.2:
        color = '#fff9e6'
        msg = (f"Extreme Positioning Alert for {contract}: Current z-score ({z:.2f}) suggests "
               f"{'net short' if z < 0 else 'net long'} positioning is reaching an extreme. "
               f"Historical patterns suggest potential for mean reversion.")
    else:
        color = '#ffffcc'
        msg = (f"No Strong Signal for {contract}. Current positioning (z={z:.2f}, net={net:,.0f}) shows "
               f"no significant short-term trend. 2-month momentum is {recent_trend:.0f} contracts per week.")
    return html.Div(msg, style={'background': color, 'padding': '10px', 'borderRadius': '6px', 'marginBottom': '30px'})

def fair_value_roll(front_price, back_price, spot, repo, days, cf_front, cf_back):
    front_fwd = spot * (1 + repo * days / 360)
    back_fwd = spot * (1 + repo * (days + 90) / 360)
    fair_front = front_fwd / cf_front if cf_front != 0 else 0
    fair_back = back_fwd / cf_back if cf_back != 0 else 0
    fair_roll = fair_front - fair_back
    market_roll = front_price - back_price
    richness = market_roll - fair_roll
    return {
        'Market Roll': market_roll,
        'Fair Roll': fair_roll,
        'Richness': richness,
        'Implied Repo': repo
    }

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # Define server here for Render
app.title = "US Treasury Futures Positioning Dashboard"

try:
    print("Initializing data...")
    cot_df = fetch_cot_data()
    signals = compute_signals(cot_df, CONTRACT_NAME_FILTERS)
    futures_prices = fetch_futures_prices()
    price_series = {k: fetch_price_series(k) for k in CONTRACT_NAME_FILTERS}
    print("Data initialization complete.")
except Exception as e:
    print(f"Error during data initialization: {e}")
    signals = {k: pd.DataFrame({'date': [], 'net_lev': [], 'zscore': [], 'adaptive_z': [], 'momentum': [], 'cs_zscore': []}) for k in CONTRACT_NAME_FILTERS}
    futures_prices = {k: 0 for k in CONTRACT_NAME_FILTERS}
    price_series = {k: pd.Series([], dtype='float64') for k in CONTRACT_NAME_FILTERS}

def contract_analysis_section(contract):
    sig = signals.get(contract, pd.DataFrame())
    if sig.empty:
        sig = pd.DataFrame({'date': [], 'net_lev': [], 'zscore': [], 'adaptive_z': [], 'momentum': [], 'cs_zscore': []})
    insight = enhanced_trading_insight(contract, sig, price_series.get(contract, pd.Series([])))
    table_data = [
        {"Metric": "Latest Net Leveraged", "Value": f"{sig['net_lev'].iloc[-1]:,.0f}" if not sig.empty else "N/A"},
        {"Metric": "Z-Score", "Value": f"{sig['zscore'].iloc[-1]:.2f}" if not sig.empty else "N/A"},
        {"Metric": "Adaptive Z", "Value": f"{sig['adaptive_z'].iloc[-1]:.2f}" if not sig.empty else "N/A"},
        {"Metric": "Cross-Sectional Z", "Value": f"{sig['cs_zscore'].iloc[-1]:.2f}" if not sig.empty else "N/A"},
        {"Metric": "Momentum", "Value": f"{sig['momentum'].iloc[-1]:.0f}" if not sig.empty else "N/A"},
        {"Metric": "Latest Price", "Value": f"{price_series[contract].iloc[-1]:.2f}" if contract in price_series and not price_series[contract].empty else "N/A"}
    ]
    table = dash_table.DataTable(
        data=table_data,
        columns=[{"name": "Metric", "id": "Metric"}, {"name": "Value", "id": "Value"}],
        style_table={'width': '50%', 'marginBottom': '10px'},
        style_cell={'textAlign': 'left'}
    )
    return html.Div([
        html.H4(f"{contract} Treasury Analysis", style={'fontWeight': 'bold', 'marginTop': '20px', 'color': '#2c3e50'}),
        table,
        html.Label("Select metrics to display:", style={'fontWeight': 'bold', 'marginTop': '10px'}),
        dcc.Checklist(
            id=f'{contract}-metrics-checklist',
            options=[
                {'label': 'Net Leveraged Position', 'value': 'net_lev'},
                {'label': 'Z-Score', 'value': 'zscore'},
                {'label': 'Adaptive Z', 'value': 'adaptive_z'},
                {'label': 'Cross-Sectional Z', 'value': 'cs_zscore'},
                {'label': 'Momentum', 'value': 'momentum'},
                {'label': 'Price', 'value': 'price'}
            ],
            value=['net_lev', 'zscore', 'adaptive_z'],
            inline=True,
            style={'marginBottom': '10px'}
        ),
        dcc.Graph(id=f'{contract}-positioning-graph', style={'height': '650px'}),
        html.Div([
            html.Strong("Trading Insight:", style={'fontSize': '16px'}),
            html.Div(insight)
        ])
    ], style={'marginBottom': '40px', 'border': '1px solid #ddd', 'padding': '15px', 'borderRadius': '5px'})

app.layout = html.Div([
    html.H1("US Treasury Futures Positioning Analysis", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
    dcc.Tabs([
        dcc.Tab(label='Quantitative Analytics', children=[
            html.Div([
                html.H3("Contract-by-Contract Analysis", style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px', 'color': '#2c3e50'}),
                contract_analysis_section('2Y'),
                contract_analysis_section('5Y'),
                contract_analysis_section('7Y'),
                contract_analysis_section('10Y'),
                contract_analysis_section('30Y'),
                html.H3("Cross-Contract Analysis", style={'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '20px', 'color': '#2c3e50'}),
                dcc.Graph(
                    id='all-contracts-graph',
                    figure=go.Figure([
                        go.Scatter(x=signals[k]['date'], y=signals[k]['net_lev'], mode='lines', name=f"{k} Net Position")
                        for k in signals if not signals[k].empty
                    ]).update_layout(
                        title="Net Leveraged Position - All Contracts",
                        xaxis=dict(title='', type='date'),
                        yaxis=dict(title='Net Position'),
                        template="plotly_white",
                        height=650,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                ),
                dcc.Graph(
                    id='corr-heatmap',
                    figure=px.imshow(
                        compute_corr_heatmap(signals),
                        text_auto=True,
                        title="Correlation Matrix of Net Positions"
                    ).update_layout(height=500)
                ),
                html.Div([
                    html.Hr(),
                    html.P("© Jacob Robinson 2025. All rights reserved.", 
                    style={'textAlign': 'center', 'color': '#7f7f7f', 'fontSize': '12px', 'marginTop': '20px'})
                ])
            ], style={'width': '95%', 'margin': 'auto', 'padding': '20px'})
        ]),
        dcc.Tab(label='Futures Roll Analysis', children=[
            html.Div([
                html.H3("Treasury Futures Roll Calculator", style={'marginTop': '20px', 'color': '#2c3e50'}),
                html.P("This calculator helps determine fair value of calendar rolls based on financing rates and conversion factors."),
                html.Div([
                    html.Div([
                        html.Label("Select Contract:"),
                        dcc.Dropdown(
                            id='roll-contract-dropdown',
                            options=[{'label': k, 'value': k} for k in CONTRACT_NAME_FILTERS.keys()],
                            value='10Y',
                            style={'width': '100%'}
                        )
                    ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '20px'}),
                    html.Div([
                        html.Label("Front Month Price:"),
                        dcc.Input(id='front-price', type='number', value=0, step=0.01, style={'width': '100%'})
                    ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '20px'}),
                    html.Div([
                        html.Label("Back Month Price:"),
                        dcc.Input(id='back-price', type='number', value=0, step=0.01, style={'width': '100%'})
                    ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '20px'}),
                    html.Div([
                        html.Label("Spot Price:"),
                        dcc.Input(id='spot-price', type='number', value=100, step=0.01, style={'width': '100%'})
                    ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '20px'}),
                    html.Div([
                        html.Label("Days to Delivery:"),
                        dcc.Input(id='days-delivery', type='number', value=30, step=1, style={'width': '100%'})
                    ], style={'width': '15%', 'display': 'inline-block'})
                ], style={'marginBottom': '20px'}),
                html.Div([
                    html.Div([
                        html.Label("Front Conversion Factor:"),
                        dcc.Input(id='cf-front', type='number', value=1.0, step=0.0001, style={'width': '100%'})
                    ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '20px'}),
                    html.Div([
                        html.Label("Back Conversion Factor:"),
                        dcc.Input(id='cf-back', type='number', value=1.0, step=0.0001, style={'width': '100%'})
                    ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '20px'}),
                    html.Div([
                        html.Label("Repo Rate (%):"),
                        dcc.Input(id='repo-rate', value=DEFAULT_REPO*100, type='number', step=0.01, style={'width': '100%'})
                    ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '20px'}),
                    html.Div([
                        html.Button("Calculate Roll", id='roll-btn', n_clicks=0, 
                                   style={'backgroundColor': '#2c3e50', 'color': 'white', 'border': 'none', 
                                          'borderRadius': '5px', 'padding': '10px 15px', 'marginTop': '22px'})
                    ])
                ], style={'marginBottom': '30px'}),
                dash_table.DataTable(
                    id='roll-table',
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'Richness', 'filter_query': '{Richness} > 0'},
                            'backgroundColor': '#e6ffe6',
                            'color': 'green'
                        },
                        {
                            'if': {'column_id': 'Richness', 'filter_query': '{Richness} < 0'},
                            'backgroundColor': '#ffe6e6',
                            'color': 'red'
                        }
                    ]
                )
            ], style={'width': '90%', 'margin': 'auto', 'padding': '20px'})
        ])
    ])
])

for contract in CONTRACT_NAME_FILTERS.keys():
    @app.callback(
        Output(f'{contract}-positioning-graph', 'figure'),
        Input(f'{contract}-metrics-checklist', 'value')
    )
    def update_contract_chart(selected_metrics, contract=contract):
        sig = signals.get(contract, pd.DataFrame())
        if sig.empty:
            sig = pd.DataFrame({'date': [], 'net_lev': [], 'zscore': [], 'adaptive_z': [], 'momentum': [], 'cs_zscore': []})
        fig = go.Figure()
        color_map = {
            'net_lev': 'blue', 'zscore': 'orange', 'adaptive_z': 'green',
            'cs_zscore': 'purple', 'momentum': 'brown', 'price': 'crimson'
        }
        yaxis2_metrics = {'zscore', 'adaptive_z', 'cs_zscore'}
        has_data = False
        for metric in selected_metrics:
            if metric == 'price':
                price_data = price_series.get(contract, pd.Series([]))
                if not price_data.empty:
                    fig.add_trace(go.Scatter(
                        x=price_data.index, 
                        y=price_data.values, 
                        name='Price',
                        yaxis='y3', 
                        mode='lines',
                        line=dict(color='crimson', width=3, dash='dash')
                    ))
                    has_data = True
            elif not sig.empty and metric in sig.columns:
                fig.add_trace(go.Scatter(
                    x=sig['date'], 
                    y=sig[metric], 
                    name=metric.replace('_', ' ').title(),
                    yaxis='y2' if metric in yaxis2_metrics else 'y1',
                    mode='lines',
                    line=dict(color=color_map.get(metric, 'gray'), width=1.5)
                ))
                has_data = True
        fig.update_layout(
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(230, 230, 230, 0.5)',
                zeroline=True,
                zerolinecolor='rgba(0,0,0,0.3)'
            ),
            yaxis=dict(
                title='Net Leveraged / Momentum',
                showgrid=True,
                gridcolor='rgba(230, 230, 230, 0.5)',
                zeroline=True,
                zerolinecolor='rgba(0,0,0,0.3)'
            ),
            yaxis2=dict(
                title='Z-Scores',
                overlaying='y',
                side='right',
                showgrid=False,
                zeroline=True,
                zerolinecolor='rgba(0,0,0,0.3)'
            ),
            yaxis3=dict(
                title='Price',
                overlaying='y',
                side='right',
                position=0.95,
                showgrid=False,
                zeroline=False
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            height=650,
            template='plotly_white'
        )
        if not has_data:
            fig.add_annotation(
                text="No data available for this contract/metric selection.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
        return fig

@app.callback(
    Output('roll-table', 'data'),
    Output('roll-table', 'columns'),
    Input('roll-btn', 'n_clicks'),
    State('front-price', 'value'),
    State('back-price', 'value'),
    State('spot-price', 'value'),
    State('repo-rate', 'value'),
    State('days-delivery', 'value'),
    State('cf-front', 'value'),
    State('cf-back', 'value')
)
def roll_analysis(n_clicks, front, back, spot, repo, days, cf_front, cf_back):
    if n_clicks == 0 or not all([front, back, spot, repo, days, cf_front, cf_back]):
        return [], []
    result = fair_value_roll(front, back, spot, repo/100, days, cf_front, cf_back)
    data = [{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in result.items()}]
    columns = [{"name": k, "id": k} for k in result.keys()]
    return data, columns

if __name__ == '__main__':
    app.run(debug=True)
