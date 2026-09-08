# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go

st.set_page_config(page_title="Retail Demand Control Plane", page_icon="🛍️", layout="wide")

st.markdown("""
    <style>
    .metric-card { background-color: #f8f9fa; border-radius: 8px; padding: 16px; border-left: 5px solid #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    raw_hist = pd.read_parquet("data/daily_transactions.parquet")
    with open("artifacts/evaluation_metrics.json") as f:
        metrics = json.load(f)
    test_preds = pd.read_parquet("artifacts/test_predictions.parquet")
    future_fc = pd.read_parquet("artifacts/future_forecast.parquet")
    return raw_hist, metrics, test_preds, future_fc

raw_hist, metrics, test_preds, future_fc = load_data()

st.sidebar.title("🎛️ Control Panel")
forecast_days = st.sidebar.selectbox("Future Horizon", options=[7, 14, 30, 90], index=2)
confidence_level = st.sidebar.select_slider("Confidence Band", options=["80%", "95%"], value="95%")
overlay_y2 = st.sidebar.toggle("Overlay Year 2 Ground Truth", value=True)

tab_screen1, tab_screen2 = st.tabs(["📈 Executive Forecast Viewer", "🔬 Model Performance & Residuals"])

with tab_screen1:
    st.subheader("LightGBM Demand Overview")
    subset_future = future_fc.head(forecast_days)
    next_7_vol = int(future_fc.head(7)["yhat"].sum())
    peak_row = subset_future.loc[subset_future["yhat"].idxmax()]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Next 7 Days Volume", f"{next_7_vol:,} txns")
    col2.metric("Model WMAPE (Year 2)", f"{metrics['test']['WMAPE']}%", f"Accuracy: {metrics['test']['Accuracy']}%", delta_color="normal")
    col3.metric(f"Expected Peak (Next {forecast_days}d)", peak_row["t_dat"].strftime("%b %d"), f"{int(peak_row['yhat']):,} txns")
        
    st.markdown("---")
    fig = go.Figure()
    
    train_df = raw_hist[raw_hist['t_dat'] < test_preds['t_dat'].min()]
    fig.add_trace(go.Scatter(x=train_df["t_dat"], y=train_df["transaction_count"], mode="lines", name="Year 1 (Train)", line=dict(color="#6c757d", width=1.5)))

    if overlay_y2:
        fig.add_trace(go.Scatter(x=test_preds["t_dat"], y=test_preds["transaction_count"], mode="lines", name="Year 2 Actuals", line=dict(color="#1f77b4", width=1.5)))
        fig.add_trace(go.Scatter(x=test_preds["t_dat"], y=test_preds["pred"], mode="lines", name="LightGBM Predictions", line=dict(color="#2ca02c", width=1.5, dash="dot")))

    band_suffix = "95" if confidence_level == "95%" else "80"
    fig.add_trace(go.Scatter(x=subset_future["t_dat"], y=subset_future["yhat"], mode="lines+markers", name="Future Forecast", line=dict(color="#d62728", width=2.5)))
    fig.add_trace(go.Scatter(x=subset_future["t_dat"], y=subset_future[f"yhat_upper_{band_suffix}"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=subset_future["t_dat"], y=subset_future[f"yhat_lower_{band_suffix}"], mode="lines", fill="tonexty", fillcolor="rgba(214, 39, 40, 0.15)", name=f"{confidence_level} CI", line=dict(width=0)))

    fig.update_layout(title="Retail Demand Trajectory (Spike-Aware)", hovermode="x unified", height=550, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tab_screen2:
    st.subheader("Model Diagnostic & Out-Of-Time Evaluation")
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        metrics_df = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "MAPE / WMAPE", "R² Score"],
            "Train (Yr 1)": [metrics['train']['MAE'], metrics['train']['RMSE'], f"{metrics['train']['MAPE']}%", metrics['train']['R2']],
            "Test (Yr 2)": [metrics['test']['MAE'], metrics['test']['RMSE'], f"{metrics['test']['WMAPE']}%", metrics['test']['R2']]
        })
        st.table(metrics_df)

    with col_right:
        residual_fig = go.Figure()
        residual_fig.add_trace(go.Bar(x=test_preds["t_dat"], y=test_preds["residual"], name="Residuals", marker_color=np.where(test_preds["residual"] >= 0, "#2ca02c", "#d62728")))
        residual_fig.update_layout(title="Year 2 Prediction Residuals", height=300, template="plotly_white")
        st.plotly_chart(residual_fig, use_container_width=True)
        
    st.markdown("#### Detected Spikes & Anomalies")
    anomalies = test_preds[test_preds["z_score"].abs() >= 2.0].copy()
    if not anomalies.empty:
        st.dataframe(anomalies[["t_dat", "transaction_count", "pred", "residual", "z_score"]].sort_values("z_score", ascending=False), use_container_width=True)
