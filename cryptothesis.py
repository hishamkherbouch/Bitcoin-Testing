# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



df = pd.read_csv("data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

df = df[["Open","High","Low","Close","Volume BTC","Volume USD"]].dropna()

df["r_1m"] = np.log(df["Close"]).diff()
df["r_15m"] = np.log(df["Close"]).diff(15)
df["vol_1h"] = df["r_1m"].rolling(60).std()

# 15m volume + Z-score
df["vol_15m"] = df["Volume BTC"].rolling(15).sum()
roll_mean = df["vol_15m"].rolling(360).mean()
roll_std = df["vol_15m"].rolling(360).std()
df["vol_15m_z"] = (df["vol_15m"] - roll_mean) / roll_std



signal = pd.Series(0, index=df.index)
k = 2.0
vol_z_threshold=2.0

up_spike = (df["r_15m"] > k*df["vol_1h"]) & (df["vol_15m_z"] > vol_z_threshold)
down_spike = (df["r_15m"] < -k*df["vol_1h"]) & (df["vol_15m_z"] > vol_z_threshold)

signal[up_spike] = -1
signal[down_spike] = 1

df["signal_entry"] = signal
df["signal"] = df["signal_entry"].shift(1)

hold_bars = 30
held = df["signal"].copy()
for i in range(1, hold_bars):
    held = held.combine_first(df["signal"].shift(i))

df["held_signal"] = held

df["r_1m_fwd"] = df["r_1m"].shift(-1)
df["strat_r"] = df["held_signal"] * df["r_1m_fwd"]
strat_r = df["strat_r"].dropna()


real_pf = strat_r[strat_r>0].sum() / strat_r[strat_r<0].abs().sum()




def permute_returns(df):
    r = df["r_1m_fwd"].values.copy()
    valid_idx = ~np.isnan(r)

    shuffled = r[valid_idx].copy()
    np.random.shuffle(shuffled)

    r[valid_idx] = shuffled

    perm_df = df.copy()
    perm_df["r_1m_perm"] = r
    return perm_df


n_permutations = 1000
perm_better_count = 1
permuted_pfs = []

print("Running In-Sample MCPT...")

for _ in tqdm(range(n_permutations)):
    perm_df = permute_returns(df)
    perm_strat_r = df["held_signal"] * perm_df["r_1m_perm"]
    perm_pf = (
        perm_strat_r[perm_strat_r > 0].sum() /
        perm_strat_r[perm_strat_r < 0].abs().sum()
        if (perm_strat_r < 0).any() else 0
    )
    permuted_pfs.append(perm_pf)

    if perm_pf >= real_pf:
        perm_better_count += 1

p_value = perm_better_count / n_permutations
print("Real PF:", real_pf)
print("MCPT p-value:", p_value)



plt.style.use("dark_background")
pd.Series(permuted_pfs).hist(color="skyblue", bins=30)
plt.axvline(real_pf, color="red", linewidth=2, label="Real PF")
plt.title(f"Monte Carlo Permutation Test (p-value = {p_value:.4f})")
plt.xlabel("Profit Factor")
plt.legend()
plt.grid(False)
plt.show()

