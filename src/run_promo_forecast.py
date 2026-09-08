import os
import json
import duckdb
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. Dynamic Floating Holiday Helpers
# ---------------------------------------------------------------------------

def _thanksgiving(year: int) -> pd.Timestamp:
    """Return the fourth Thursday of November for the given year."""
    # November 1st
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    # Day-of-week for Nov 1 (Monday=0 … Sunday=6)
    dow_nov1 = nov1.dayofweek
    # First Thursday offset
    first_thu = (3 - dow_nov1) % 7  # Thursday = 3
    # Fourth Thursday
    fourth_thu = nov1 + pd.Timedelta(days=first_thu + 21)
    return fourth_thu


def _black_friday_window(year: int):
    """Wednesday before Thanksgiving through Cyber Monday (6-day window)."""
    tg = _thanksgiving(year)
    start = tg - pd.Timedelta(days=1)   # Wednesday
    end   = tg + pd.Timedelta(days=4)   # Cyber Monday
    return start, end


def _apply_promo_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all promotional event flags dynamically based on date logic."""
    df = df.copy()
    month = df['t_dat'].dt.month
    day   = df['t_dat'].dt.day

    # --- Black Friday / Cyber Week (dynamic per year) ---
    df['is_black_friday_week'] = 0
    for year in df['t_dat'].dt.year.unique():
        bf_start, bf_end = _black_friday_window(year)
        mask = (df['t_dat'] >= bf_start) & (df['t_dat'] <= bf_end)
        df.loc[mask, 'is_black_friday_week'] = 1

    # --- Summer Sale: last week of June into early July ---
    df['is_june_summer_sale'] = (
        ((month == 6) & (day >= 24)) | ((month == 7) & (day <= 2))
    ).astype(int)

    # --- September Payday Sale: last week of September ---
    df['is_september_payday_sale'] = (
        (month == 9) & (day >= 24)
    ).astype(int)

    # --- Spring Payday Sale: late April / late May ---
    df['is_spring_payday_sale'] = (
        ((month == 4) & (day >= 24) & (day <= 28)) |
        ((month == 5) & (day >= 23) & (day <= 27))
    ).astype(int)

    # --- Combined promo intensity (useful gradient signal) ---
    df['promo_intensity'] = (
        df['is_black_friday_week'] * 3 +
        df['is_june_summer_sale'] * 2 +
        df['is_september_payday_sale'] +
        df['is_spring_payday_sale']
    )

    return df

# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------

def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, cyclical, and interaction features."""
    df = df.copy()
    df['month']     = df['t_dat'].dt.month
    df['day']       = df['t_dat'].dt.day
    df['dayofweek'] = df['t_dat'].dt.dayofweek
    df['dayofyear'] = df['t_dat'].dt.dayofyear
    df['weekofyear'] = df['t_dat'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_end'] = (df['day'] >= 28).astype(int)

    # Fourier harmonics for yearly seasonality
    for K in range(1, 6):
        df[f'sin_{K}'] = np.sin(2 * np.pi * K * df['dayofyear'] / 365.25)
        df[f'cos_{K}'] = np.cos(2 * np.pi * K * df['dayofyear'] / 365.25)

    # Weekly cyclical encoding
    df['sin_dow'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['cos_dow'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    return df


def _add_lag_features(df: pd.DataFrame, target_col: str = 'transaction_count') -> pd.DataFrame:
    """Add autoregressive lag and rolling window features (shifted to prevent leakage)."""
    df = df.copy()
    df = df.sort_values('t_dat').reset_index(drop=True)

    # Near-term lags (critical for capturing recent momentum)
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    # Rolling window statistics (shifted by 1 to prevent same-day leakage)
    shifted = df[target_col].shift(1)
    df['rolling_mean_7']  = shifted.rolling(window=7,  min_periods=3).mean()
    df['rolling_std_7']   = shifted.rolling(window=7,  min_periods=3).std()
    df['rolling_mean_14'] = shifted.rolling(window=14, min_periods=5).mean()
    df['rolling_mean_28'] = shifted.rolling(window=28, min_periods=7).mean()
    df['rolling_max_7']   = shifted.rolling(window=7,  min_periods=3).max()   # spike detector
    df['expanding_mean']  = shifted.expanding(min_periods=7).mean()            # global trend

    # Momentum: ratio of short-term to long-term rolling mean
    df['momentum_7_28'] = df['rolling_mean_7'] / df['rolling_mean_28'].replace(0, np.nan)

    # Lag difference (acceleration signal)
    df['lag_diff_1_7'] = df['lag_1'] - df['lag_7']

    return df

# ---------------------------------------------------------------------------
# 3. Main Pipeline
# ---------------------------------------------------------------------------

NON_FEATURE_COLS = frozenset([
    't_dat', 'transaction_count', 'transaction_count_log',
    'promo_event_name', 'week_no', 'pred', 'pred_log',
    'error', 'residual', 'z_score'
])


def run_pipeline():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    print("=" * 60)
    print("Running Spike-Aware Promo Forecast Pipeline v2.0")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Data Ingestion (synthetic mock data for CI/CD)
    # ------------------------------------------------------------------
    mock_data_path = "data/transactions_train.parquet"
    if not os.path.exists(mock_data_path):
        print("[Data] Generating synthetic training data with promo spikes...")
        np.random.seed(42)
        dates = pd.date_range(start="2018-09-20", end="2020-09-22", freq="D")
        base = np.random.normal(5000, 200, len(dates)).astype(int)
        mock_df = pd.DataFrame({"t_dat": dates, "transaction_count": base})

        # Black Friday spikes (dynamic)
        for year in [2018, 2019]:
            bf_start, bf_end = _black_friday_window(year)
            mask = (mock_df['t_dat'] >= bf_start) & (mock_df['t_dat'] <= bf_end)
            mock_df.loc[mask, 'transaction_count'] += 15000

        # Summer sale spikes
        summer_mask = (
            ((mock_df['t_dat'].dt.month == 6) & (mock_df['t_dat'].dt.day >= 24)) |
            ((mock_df['t_dat'].dt.month == 7) & (mock_df['t_dat'].dt.day <= 2))
        )
        mock_df.loc[summer_mask, 'transaction_count'] += 12000

        # September payday mini-spikes
        sep_mask = (mock_df['t_dat'].dt.month == 9) & (mock_df['t_dat'].dt.day >= 24)
        mock_df.loc[sep_mask, 'transaction_count'] += 5000

        # Spring payday mini-spikes
        spring_mask = (
            ((mock_df['t_dat'].dt.month == 4) & (mock_df['t_dat'].dt.day >= 24) & (mock_df['t_dat'].dt.day <= 28)) |
            ((mock_df['t_dat'].dt.month == 5) & (mock_df['t_dat'].dt.day >= 23) & (mock_df['t_dat'].dt.day <= 27))
        )
        mock_df.loc[spring_mask, 'transaction_count'] += 4000

        mock_df.to_parquet(mock_data_path, index=False)
        print(f"[Data] Saved {len(mock_df)} rows -> {mock_data_path}")

    # ------------------------------------------------------------------
    # Aggregate daily via DuckDB
    # ------------------------------------------------------------------
    con = duckdb.connect()
    df_daily = con.execute(
        f"SELECT CAST(t_dat AS DATE) as t_dat, "
        f"SUM(transaction_count) as transaction_count "
        f"FROM '{mock_data_path}' GROUP BY 1 ORDER BY 1"
    ).df()
    df_daily['t_dat'] = pd.to_datetime(df_daily['t_dat'])
    df_daily.to_parquet("data/daily_transactions.parquet", index=False)
    print(f"[Data] Daily aggregation: {len(df_daily)} days")

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------
    d = df_daily.copy()
    d = _add_calendar_features(d)
    d = _apply_promo_flags(d)
    d = _add_lag_features(d, target_col='transaction_count')

    # Drop rows with NaN lags (first ~28 days)
    d = d.dropna().reset_index(drop=True)

    # Log-transform target for variance stabilization
    d['transaction_count_log'] = np.log1p(d['transaction_count'])

    # ------------------------------------------------------------------
    # Train / Test Split (Year 1 vs Year 2, out-of-time)
    # ------------------------------------------------------------------
    unique_dates = d['t_dat'].sort_values().unique()
    split_idx = len(unique_dates) // 2
    train_dates = unique_dates[:split_idx]
    test_dates  = unique_dates[split_idx:]

    train_df = d[d['t_dat'].isin(train_dates)].copy()
    test_df  = d[d['t_dat'].isin(test_dates)].copy()

    feature_cols = sorted([
        c for c in d.columns if c not in NON_FEATURE_COLS
    ])
    print(f"[Features] {len(feature_cols)} features: {feature_cols[:10]}...")
    print(f"[Split] Train: {len(train_df)} days | Test: {len(test_df)} days")

    # Sample weights: upweight promo periods so the model doesn't ignore spikes
    train_weights = np.ones(len(train_df))
    promo_mask = (
        (train_df['is_black_friday_week'] == 1) |
        (train_df['is_june_summer_sale'] == 1) |
        (train_df['is_september_payday_sale'] == 1) |
        (train_df['is_spring_payday_sale'] == 1)
    ).values
    train_weights[promo_mask] = 5.0  # 5x weight on promotional spike days

    model = lgb.LGBMRegressor(
        objective='regression_l1',      # MAE loss -- robust to outliers
        n_estimators=5000,
        learning_rate=0.01,
        num_leaves=127,
        max_depth=12,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_samples=3,
        reg_alpha=0.05,
        reg_lambda=0.05,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(
        train_df[feature_cols], train_df['transaction_count_log'],
        sample_weight=train_weights,
        eval_set=[(test_df[feature_cols], test_df['transaction_count_log'])],
        callbacks=[
            lgb.log_evaluation(period=0),
            lgb.early_stopping(stopping_rounds=200, verbose=False),
        ],
    )

    # ------------------------------------------------------------------
    # Predict & invert log transform
    # ------------------------------------------------------------------
    train_df['pred'] = np.expm1(model.predict(train_df[feature_cols]))
    test_df['pred']  = np.expm1(model.predict(test_df[feature_cols]))

    # Clamp predictions to non-negative
    train_df['pred'] = train_df['pred'].clip(lower=0)
    test_df['pred']  = test_df['pred'].clip(lower=0)

    # Residual analysis
    test_df['error']    = test_df['pred'] - test_df['transaction_count']
    test_df['residual'] = test_df['error']
    residual_std = test_df['residual'].std()
    residual_mean = test_df['residual'].mean()
    test_df['z_score'] = (test_df['residual'] - residual_mean) / residual_std if residual_std > 0 else 0

    # ------------------------------------------------------------------
    # Compute Metrics
    # ------------------------------------------------------------------
    daily_wmape    = (np.abs(test_df['error']).sum() / test_df['transaction_count'].sum()) * 100.0
    daily_accuracy = 100.0 - daily_wmape

    train_mae  = round(float(mean_absolute_error(train_df["transaction_count"], train_df["pred"])), 2)
    train_rmse = round(float(np.sqrt(mean_squared_error(train_df["transaction_count"], train_df["pred"]))), 2)
    train_mape = round(float(np.mean(np.abs((train_df["transaction_count"] - train_df["pred"]) / train_df["transaction_count"].replace(0, 1))) * 100), 2)
    train_r2   = round(float(r2_score(train_df["transaction_count"], train_df["pred"])), 4)

    test_mae  = round(float(mean_absolute_error(test_df["transaction_count"], test_df["pred"])), 2)
    test_rmse = round(float(np.sqrt(mean_squared_error(test_df["transaction_count"], test_df["pred"]))), 2)
    test_wmape = round(float(daily_wmape), 2)
    test_accuracy = round(float(daily_accuracy), 2)
    test_r2   = round(float(r2_score(test_df["transaction_count"], test_df["pred"])), 4)

    print(f"\n{'='*60}")
    print(f"  TRAIN -- MAE: {train_mae}  RMSE: {train_rmse}  MAPE: {train_mape}%  R2: {train_r2}")
    print(f"  TEST  -- MAE: {test_mae}  RMSE: {test_rmse}  WMAPE: {test_wmape}%  R2: {test_r2}")
    print(f"  TEST  -- Accuracy: {test_accuracy}%")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Export Artifacts
    # ------------------------------------------------------------------
    print("[Export] Writing evaluation_metrics.json...")
    metrics = {
        "train": {"MAE": train_mae, "RMSE": train_rmse, "MAPE": train_mape, "R2": train_r2},
        "test":  {"MAE": test_mae, "RMSE": test_rmse, "WMAPE": test_wmape, "Accuracy": test_accuracy, "R2": test_r2}
    }
    with open("artifacts/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[Export] Writing test_predictions.parquet...")
    test_df.to_parquet("artifacts/test_predictions.parquet", index=False)

    # ------------------------------------------------------------------
    # Recursive Future Forecast (90 days)
    # ------------------------------------------------------------------
    print("[Forecast] Generating recursive 90-day future forecast...")

    # Seed the history buffer with the last 28 days of actual data for lag lookback
    history = d[['t_dat', 'transaction_count']].tail(28).copy()
    future_start = test_df['t_dat'].max() + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=future_start, periods=90, freq="D")

    future_records = []
    for fdate in future_dates:
        row = pd.DataFrame({"t_dat": [fdate]})
        row = _add_calendar_features(row)
        row = _apply_promo_flags(row)

        # Build lag features from history buffer
        hist_vals = history['transaction_count'].values
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            if len(hist_vals) >= lag:
                row[f'lag_{lag}'] = hist_vals[-lag]
            else:
                row[f'lag_{lag}'] = hist_vals.mean()

        # Rolling statistics from history
        recent_7  = hist_vals[-7:]  if len(hist_vals) >= 7  else hist_vals
        recent_14 = hist_vals[-14:] if len(hist_vals) >= 14 else hist_vals
        recent_28 = hist_vals[-28:] if len(hist_vals) >= 28 else hist_vals

        row['rolling_mean_7']  = np.mean(recent_7)
        row['rolling_std_7']   = np.std(recent_7) if len(recent_7) > 1 else 0.0
        row['rolling_mean_14'] = np.mean(recent_14)
        row['rolling_mean_28'] = np.mean(recent_28)
        row['rolling_max_7']   = np.max(recent_7)
        row['expanding_mean']  = np.mean(hist_vals)
        row['momentum_7_28']   = np.mean(recent_7) / np.mean(recent_28) if np.mean(recent_28) > 0 else 1.0
        row['lag_diff_1_7']    = float(row['lag_1'].iloc[0]) - float(row['lag_7'].iloc[0])

        # Predict (in log space, then invert)
        pred_log = model.predict(row[feature_cols])
        pred_val = float(np.expm1(pred_log[0]))
        pred_val = max(pred_val, 0)

        future_records.append({"t_dat": fdate, "yhat": pred_val})

        # Append prediction to history buffer for next iteration
        history = pd.concat([
            history,
            pd.DataFrame({"t_dat": [fdate], "transaction_count": [pred_val]})
        ], ignore_index=True)

    future_out = pd.DataFrame(future_records)

    # Confidence intervals based on test-set residual distribution
    test_residual_std = test_df['residual'].std()
    # 80% CI ~ +/-1.28*std, 95% CI ~ +/-1.96*std, 85% CI ~ +/-1.44*std, 90% CI ~ +/-1.64*std
    future_out['yhat_lower_80'] = (future_out['yhat'] - 1.28 * test_residual_std).clip(lower=0)
    future_out['yhat_upper_80'] = future_out['yhat'] + 1.28 * test_residual_std
    future_out['yhat_lower_85'] = (future_out['yhat'] - 1.44 * test_residual_std).clip(lower=0)
    future_out['yhat_upper_85'] = future_out['yhat'] + 1.44 * test_residual_std
    future_out['yhat_lower_90'] = (future_out['yhat'] - 1.64 * test_residual_std).clip(lower=0)
    future_out['yhat_upper_90'] = future_out['yhat'] + 1.64 * test_residual_std
    future_out['yhat_lower_95'] = (future_out['yhat'] - 1.96 * test_residual_std).clip(lower=0)
    future_out['yhat_upper_95'] = future_out['yhat'] + 1.96 * test_residual_std

    future_out.to_parquet("artifacts/future_forecast.parquet", index=False)
    print(f"[Export] future_forecast.parquet -- {len(future_out)} days")

    # ------------------------------------------------------------------
    # Quality Gate Assertion
    # ------------------------------------------------------------------
    print("\n[QualityGate] Checking production thresholds...")
    assert test_r2 >= 0.65, f"FAIL: Test R2 = {test_r2} < 0.65"
    assert test_wmape <= 15.0, f"FAIL: Test WMAPE = {test_wmape}% > 15%"
    print("[QualityGate] PASSED -- Model meets production quality targets.")
    print("\nPipeline complete. Artifacts saved for Streamlit UI.")


if __name__ == "__main__":
    run_pipeline()
