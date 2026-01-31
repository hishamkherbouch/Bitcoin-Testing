# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 17:42:05 2026

@author: hkher
"""

# backtest.py
import os
import json
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Params:
    k: float = 2.0
    vol_z_threshold: float = 2.0
    hold_bars: int = 30
    vol_window_1h: int = 60           # 60 bars for 1h if 1-min bars
    volsum_window_15m: int = 15       # 15 bars for 15m if 1-min bars
    zscore_window: int = 360          # your original 360 window


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Keep only the columns you used
    keep = ["Open", "High", "Low", "Close", "Volume BTC", "Volume USD"]
    df = df[keep].dropna()
    return df


def engineer_features(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = df.copy()

    out["r_1m"] = np.log(out["Close"]).diff()
    out["r_15m"] = np.log(out["Close"]).diff(p.volsum_window_15m)

    out["vol_1h"] = out["r_1m"].rolling(p.vol_window_1h).std()

    out["vol_15m"] = out["Volume BTC"].rolling(p.volsum_window_15m).sum()
    roll_mean = out["vol_15m"].rolling(p.zscore_window).mean()
    roll_std = out["vol_15m"].rolling(p.zscore_window).std()
    out["vol_15m_z"] = (out["vol_15m"] - roll_mean) / roll_std

    # Forward return used for PnL
    out["r_1m_fwd"] = out["r_1m"].shift(-1)
    return out


def build_signal(df: pd.DataFrame, p: Params) -> pd.Series:
    # contrarian: fade spikes with high volume z-score
    up_spike = (df["r_15m"] > p.k * df["vol_1h"]) & (df["vol_15m_z"] > p.vol_z_threshold)
    down_spike = (df["r_15m"] < -p.k * df["vol_1h"]) & (df["vol_15m_z"] > p.vol_z_threshold)

    signal_entry = pd.Series(0, index=df.index, dtype=float)
    signal_entry[up_spike] = -1.0
    signal_entry[down_spike] = 1.0

    # Enter next bar to avoid lookahead
    return signal_entry.shift(1)


def apply_hold(signal: pd.Series, hold_bars: int) -> pd.Series:
    held = signal.copy()
    for i in range(1, hold_bars):
        held = held.combine_first(signal.shift(i))
    return held.fillna(0.0)


def strategy_returns(df: pd.DataFrame, held_signal: pd.Series) -> pd.Series:
    strat_r = held_signal * df["r_1m_fwd"]
    strat_r = strat_r.dropna()
    return strat_r


def profit_factor(r: pd.Series) -> float:
    gains = r[r > 0].sum()
    losses = r[r < 0].abs().sum()
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    return float(gains / losses)


def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return float(dd.min())


def sharpe_ratio(r: pd.Series, bars_per_year: int) -> float:
    # log returns are additive; Sharpe uses mean/std of returns
    mu = r.mean()
    sig = r.std(ddof=0)
    if sig == 0 or np.isnan(sig):
        return 0.0
    return float((mu / sig) * np.sqrt(bars_per_year))


def sortino_ratio(r: pd.Series, bars_per_year: int) -> float:
    mu = r.mean()
    downside = r[r < 0]
    dd = downside.std(ddof=0)
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float((mu / dd) * np.sqrt(bars_per_year))


def cagr_from_log_returns(r: pd.Series, bars_per_year: int) -> float:
    # Convert log return series to total return, then annualize
    total_log = r.sum()
    total_return = float(np.exp(total_log) - 1.0)
    years = len(r) / bars_per_year
    if years <= 0:
        return 0.0
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def compute_metrics(strat_r: pd.Series, held_signal: pd.Series, bars_per_year: int) -> dict:
    equity = np.exp(strat_r.cumsum())
    pf = profit_factor(strat_r)

    win_rate = float((strat_r > 0).mean()) if len(strat_r) else 0.0
    exposure = float((held_signal.loc[strat_r.index] != 0).mean()) if len(strat_r) else 0.0

    metrics = {
        "count_bars": int(len(strat_r)),
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "cagr": cagr_from_log_returns(strat_r, bars_per_year) if len(strat_r) else 0.0,
        "sharpe": sharpe_ratio(strat_r, bars_per_year) if len(strat_r) else 0.0,
        "sortino": sortino_ratio(strat_r, bars_per_year) if len(strat_r) else 0.0,
        "max_drawdown": max_drawdown(equity) if len(equity) else 0.0,
        "profit_factor": float(pf),
        "win_rate": win_rate,
        "avg_log_return": float(strat_r.mean()) if len(strat_r) else 0.0,
        "exposure": exposure,
    }
    return metrics


def permute_returns(df: pd.DataFrame) -> pd.Series:
    # Permute forward returns while keeping signal fixed
    r = df["r_1m_fwd"].values.copy()
    valid = ~np.isnan(r)
    shuffled = r[valid].copy()
    np.random.shuffle(shuffled)
    r_perm = r.copy()
    r_perm[valid] = shuffled
    return pd.Series(r_perm, index=df.index)


def monte_carlo_permutation_test(df: pd.DataFrame, held_signal: pd.Series, n_perm: int = 1000) -> dict:
    real_r = strategy_returns(df, held_signal)
    real_pf = profit_factor(real_r)

    perm_pfs = []
    better = 1  # add-one smoothing
    for _ in range(n_perm):
        r_perm = permute_returns(df)
        perm_r = held_signal * r_perm
        perm_r = perm_r.dropna()
        pf = profit_factor(perm_r)
        perm_pfs.append(pf)
        if pf >= real_pf:
            better += 1

    p_value = better / n_perm
    return {"real_pf": float(real_pf), "p_value": float(p_value), "perm_pfs": perm_pfs}


def split_df(df: pd.DataFrame, train_frac=0.6, val_frac=0.2):
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    train = df.iloc[:i1].copy()
    val = df.iloc[i1:i2].copy()
    test = df.iloc[i2:].copy()
    return train, val, test


def ensure_outputs_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path


def plot_equity(strat_r: pd.Series, out_path: str, title: str):
    equity = np.exp(strat_r.cumsum())
    plt.figure()
    equity.plot()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (start = 1.0)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_permutation_hist(perm_pfs, real_pf: float, out_path: str, title: str):
    plt.figure()
    pd.Series(perm_pfs).hist(bins=30)
    plt.axvline(real_pf, linewidth=2, label="Real PF")
    plt.title(title)
    plt.xlabel("Profit Factor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_one(df_raw: pd.DataFrame, params: Params, bars_per_year: int) -> dict:
    df = engineer_features(df_raw, params)
    sig = build_signal(df, params)
    held = apply_hold(sig, params.hold_bars)
    r = strategy_returns(df, held)
    metrics = compute_metrics(r, held, bars_per_year)
    return {"metrics": metrics, "strat_r": r, "held": held, "df_feat": df}


def grid_search(train_df, val_df, bars_per_year: int):
    # Small but meaningful grid
    k_values = [1.5, 2.0, 2.5, 3.0]
    z_values = [1.5, 2.0, 2.5, 3.0]
    hold_values = [10, 20, 30, 45, 60]

    rows = []
    best = None
    best_score = -1e9

    for k in k_values:
        for z in z_values:
            for hold in hold_values:
                p = Params(k=k, vol_z_threshold=z, hold_bars=hold)
                # Use train+val feature engineering separately to avoid accidental leakage
                train_run = run_one(train_df, p, bars_per_year)
                val_run = run_one(val_df, p, bars_per_year)

                score = val_run["metrics"]["sharpe"]  # optimize on validation Sharpe
                row = {
                    "k": k,
                    "vol_z_threshold": z,
                    "hold_bars": hold,
                    "val_sharpe": val_run["metrics"]["sharpe"],
                    "val_profit_factor": val_run["metrics"]["profit_factor"],
                    "val_max_drawdown": val_run["metrics"]["max_drawdown"],
                    "val_total_return": val_run["metrics"]["total_return"],
                }
                rows.append(row)

                if score > best_score:
                    best_score = score
                    best = p

    results = pd.DataFrame(rows).sort_values("val_sharpe", ascending=False)
    return best, results


def main():
    csv_path = "data.csv"
    outputs = ensure_outputs_dir("outputs")

    # If your bars are 1-minute, approximate 365*24*60
    bars_per_year = 365 * 24 * 60

    df_raw = load_data(csv_path)
    train, val, test = split_df(df_raw, train_frac=0.6, val_frac=0.2)

    best_params, grid = grid_search(train, val, bars_per_year)
    grid_path = os.path.join(outputs, "grid_results.csv")
    grid.to_csv(grid_path, index=False)

    with open(os.path.join(outputs, "best_params.json"), "w") as f:
        json.dump(best_params.__dict__, f, indent=2)

    # Evaluate final on test
    test_run = run_one(test, best_params, bars_per_year)
    test_metrics = test_run["metrics"]

    # Save metrics
    metrics_path = os.path.join(outputs, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save equity curve
    plot_equity(
        test_run["strat_r"],
        os.path.join(outputs, "equity_curve_test.png"),
        title="Test Equity Curve"
    )

    # Permutation test on test
    mc = monte_carlo_permutation_test(test_run["df_feat"], test_run["held"], n_perm=1000)
    with open(os.path.join(outputs, "mcpt_test.json"), "w") as f:
        json.dump({"real_pf": mc["real_pf"], "p_value": mc["p_value"]}, f, indent=2)

    plot_permutation_hist(
        mc["perm_pfs"],
        mc["real_pf"],
        os.path.join(outputs, "mcpt_hist_test.png"),
        title=f"MC Permutation Test (test) p={mc['p_value']:.4f}"
    )

    print("Best params:", best_params)
    print("Test metrics:", test_metrics)
    print("Saved outputs to:", outputs)


if __name__ == "__main__":
    main()
