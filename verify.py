# -*- coding: utf-8 -*-
"""data/ の CSV だけから、レポート（index.html）の表の数字を再現する。
使い方: python verify.py   （必要なもの: Python 3.10 以上、pandas、numpy）
外部への通信はしない。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent / "data"
sys.stdout.reconfigure(encoding="utf-8")


def load():
    g = pd.read_csv(D / "uk_gilt_nominal_zero_coupon_daily.csv", index_col="date", parse_dates=True)
    f = pd.read_csv(D / "gbpusd_daily_frb_h10.csv", index_col="date", parse_dates=True)["usd_per_gbp"]
    uk = pd.concat([g.rename(columns={"gilt_10y": "y10", "gilt_30y": "y30"}), (1 / f).rename("fx")], axis=1, sort=True).dropna()
    j = pd.read_csv(D / "jgb_yield_daily_mof.csv", index_col="date", parse_dates=True)
    x = pd.read_csv(D / "usdjpy_daily_boj.csv", index_col="date", parse_dates=True)["jpy_per_usd_tokyo_17h"]
    jp_all = pd.concat([j.rename(columns={"jgb_10y": "y10", "jgb_30y": "y30"})[["y10", "y30"]], x.rename("fx")], axis=1, sort=True)
    jp = jp_all[jp_all.index >= "2018-01-01"].dropna()
    return uk, jp, jp_all


def kday(y, fx, k, start, end, skip=None):
    """k 営業日の30年金利の最大上昇（bp）、同じ窓の通貨の変化（%、+は自国通貨安）、出来事の前2年の日次の標準偏差（bp）"""
    ch = (y - y.shift(k)) * 100
    fch = (np.log(fx) - np.log(fx.shift(k))) * 100
    w = ch[start:end].dropna()
    if skip:
        w = w.drop(w[skip[0]:skip[1]].index)
    dt = w.idxmax()
    t0 = y.index[y.index.get_loc(dt) - k]
    dy = (y.diff() * 100)[:t0].iloc[1:]
    sd = dy[dy.index > t0 - pd.DateOffset(years=2)].std()
    return dt, w.max(), fch.get(dt, np.nan), sd


def main():
    uk, jp, jp_all = load()
    print("== 表1 30年金利の k 営業日の最大上昇（bp）、通貨（%、+は自国通貨安）、倍率 = 上昇 ÷ (前2年の日次の標準偏差 × √k)")
    for name, d, s, e, skip in [("英国 トラス期", uk, "2022-09-01", "2022-10-31", None),
                                ("英国 それ以外", uk, "2019-01-01", "2026-12-31", ("2022-09-01", "2022-11-30")),
                                ("日本 2026年1月", jp, "2026-01-05", "2026-02-27", None),
                                ("日本 2024年以降", jp, "2024-01-01", "2026-12-31", None),
                                ("日本 2019〜2023年", jp, "2019-01-01", "2023-12-31", None)]:
        for k in (1, 3, 5, 20):
            dt, v, fc, sd = kday(d["y30"], d["fx"], k, s, e, skip)
            print(f"  {name:12s} {k:2d}日 {dt.date()} 上昇 {v:+7.1f}bp 通貨 {fc:+6.2f}% 前2年の日次 {sd:5.2f}bp 倍率 {v / (sd * np.sqrt(k)):4.1f}")

    print("\n== 表2 30年金利の1日の上昇と自国通貨の1日の下落が、どちらも閾値以上だった日の数")
    for name, d in [("英国", uk), ("日本", jp)]:
        dd = pd.DataFrame({"d30": d["y30"].diff() * 100, "dfx": np.log(d["fx"]).diff() * 100}).dropna()
        for a, b in [(8, 0.5), (6, 0.3), (5, 0.3)]:
            hit = (dd["d30"] >= a) & (dd["dfx"] >= b)
            yrs = " ".join(f"{y}:{int(hit[hit.index.year == y].sum())}" for y in range(2019, 2027))
            extra = f"  2022年1〜8月:{int(hit['2022-01-01':'2022-08-31'].sum())} 9〜10月:{int(hit['2022-09-01':'2022-10-31'].sum())}" if name == "英国" else ""
            print(f"  {name} +{a}bp・{b}%  {yrs}{extra}")

    print("\n== 図 出来事の前日からの累積（30年 bp、通貨 %）")
    for name, d, base in [("英国 ミニ予算", uk, "2022-09-22"), ("日本 2026年1月", jp, "2026-01-16"),
                          ("日本 2025年5月", jp, "2025-05-19"), ("日本 2025年4月", jp, "2025-04-02")]:
        i0 = d.index.get_loc(pd.Timestamp(base))
        r0 = d.iloc[i0]
        pts = []
        for k in (3, 5, 10, 15):
            rk = d.iloc[i0 + k]
            pts.append(f"{k}日 {(rk['y30'] - r0['y30']) * 100:+.1f}bp/{(np.log(rk['fx']) - np.log(r0['fx'])) * 100:+.2f}%")
        print(f"  {name:12s} " + "  ".join(pts))

    print("\n== 日本30年の日次変化の標準偏差（bp、年ごと。2026年は途中まで）")
    a = jp_all.dropna(subset=["y10", "fx"])  # ドル円のある営業日で、前の営業日からの変化
    d30 = (a["y30"].diff() * 100).dropna()
    sd = d30.groupby(d30.index.year).std()
    print("  " + " ".join(f"{y}:{v:.2f}" for y, v in sd.items() if y in (2003, 2022, 2023, 2024, 2025, 2026)))

    print("\n== 20営業日で30年 +20bp 以上かつ円安 2% 以上（窓の終わりの月、2024年6月以降）")
    q = jp_all[jp_all.index >= "2024-06-01"].dropna(subset=["y30", "fx"])
    r = pd.DataFrame({"dy": (q["y30"] - q["y30"].shift(20)) * 100, "df": (np.log(q["fx"]) - np.log(q["fx"].shift(20))) * 100}).dropna()
    print("  " + ", ".join(sorted({d.strftime("%Y-%m") for d in r[(r["dy"] >= 20) & (r["df"] >= 2)].index})))

    print("\n== 日本国債の保有比率（%、時価の末残）")
    h = pd.read_csv(D / "jgb_holders_quarterly_boj_flow_of_funds.csv")
    for inst in ["国債・財投債（国庫短期証券を除く）", "国債・財投債＋国庫短期証券"]:
        p = h[h["instrument"] == inst].pivot_table(index="date", columns="sector", values="value_oku_yen", aggfunc="sum")
        sh = p.drop(columns="合計").div(p["合計"], axis=0) * 100
        print(f"  [{inst}]")
        print(sh.loc[[d for d in ["2019-12-31", "2024-03-31", sh.index[-1]]], ["中央銀行", "海外", "生命保険"]].round(1).to_string())

    ca = pd.read_csv(D / "japan_current_account_monthly_boj.csv", index_col="date")["current_account_100m_yen"]
    print(f"\n== 経常収支の直近12か月合計: {ca.rolling(12).sum().iloc[-1] / 10000:.1f}兆円（{ca.index[-1]} まで）")


if __name__ == "__main__":
    main()
