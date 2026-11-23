"""
İlk (sərt) versiya ilə CARİ botu eyni məlumatda müqayisə edir.
V1: giriş yalnız qiymət EMA20-nin düzgün tərəfindədirsə, fitil >= gövdə, zonaya yaxınlıq 2$, 2 setup, 30 dəq gözləmə.
CARİ: trend yalnız EMA20/EMA50, fitil >= 0.5 gövdə, yaxınlıq 3$, 4 setup, 60 dəq gözləmə.
"""
import os

import numpy as np
import pandas as pd

import bot
import bt


def bias_v1(df):
    m15 = df["close"].resample("15min").last().dropna()
    e20 = m15.ewm(span=20).mean().iloc[-1]
    e50 = m15.ewm(span=50).mean().iloc[-1]
    p = m15.iloc[-1]
    if e20 < e50 and p < e20:
        return "SELL"
    if e20 > e50 and p > e20:
        return "BUY"
    return "NONE"


def check_v1(df, touch=2.0):
    bias = bias_v1(df)
    c = df.iloc[-1]
    body = max(abs(c.close - c.open), 0.01)
    up = c.high - max(c.open, c.close)
    dn = min(c.open, c.close) - c.low
    if bias == "SELL" and c.close < c.open and up >= body:
        for z in bot.cluster(bot.pivots(df, "high")):
            if c.high >= z - touch and c.high <= z + 6 and c.close < z:
                return "SELL", c.close, z, "V1 Müqavimətdən rədd olunma"
    if bias == "BUY" and c.close > c.open and dn >= body:
        for z in bot.cluster(bot.pivots(df, "low")):
            if c.low <= z + touch and c.low >= z - 6 and c.close > z:
                return "BUY", c.close, z, "V1 Dəstəkdən sıçrayış"
    return None


def scan(df, check_fn, cooldown, log=print):
    idx = df.index
    o, h, l, c = (df[x].values for x in ["open", "high", "low", "close"])
    N = len(df)
    raw = [None] * N
    for i in range(bt.WINDOW - 1, N - 1):
        if bot.weekend(idx[i]):
            continue
        raw[i] = check_fn(df.iloc[i - bt.WINDOW + 1:i + 1])
        if i % 20000 == 0:
            log(f"Analiz: {i}/{N}")
    rows = []
    for i in range(N - 1):
        if raw[i] is None:
            continue
        if any(raw[j] is not None for j in range(max(0, i - cooldown), i)):
            continue
        side, _, zone, label = raw[i]
        gross, out, tp1 = bt.simulate(o, h, l, c, i, side, bot.SL, bot.TP1, bot.TP2)
        rows.append({"ts": idx[i], "side": side, "label": label, "gross": gross})
    return rows


def compare(df, log=print):
    v1 = scan(df, check_v1, 6, log)
    cur = scan(df, bot.check, bot.COOLDOWN_CANDLES, log)
    weeks = max((df.index[-1] - df.index[0]).days / 7, 1)
    head = (f"MÜQAYİSƏ: V1 (sərt) və CARİ bot\nMəlumat: {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}\n"
            f"Giriş: siqnaldan sonrakı şamın açılışı. Eyni şamda SL+TP: SL birinci.\n"
            f"Net/DD: 0.01 lot-da təxmini $.\n"
            f"Həftəlik siqnal: V1 {len(v1) / weeks:.1f}, CARİ {len(cur) / weeks:.1f}")
    blocks = [head]
    for name, rows, fn in [("V1 (hamısı)", v1, lambda r: True), ("CARİ (hamısı)", cur, lambda r: True)]:
        blocks.append(bt.evaluate(df, rows, name, fn)[0])
    for lb in sorted(set(r["label"] for r in cur)):
        blocks.append(bt.evaluate(df, cur, "CARİ setup: " + lb, lambda r, lb=lb: r["label"] == lb)[0])
    for side in ("SELL", "BUY"):
        blocks.append(bt.evaluate(df, cur, "CARİ tərəf: " + side, lambda r, s=side: r["side"] == s)[0])
    msgs, cur_msg = [], ""
    for b in blocks:
        if len(cur_msg) + len(b) > 3300:
            msgs.append(cur_msg)
            cur_msg = ""
        cur_msg += b + "\n\n"
    msgs.append(cur_msg)
    return msgs


def main():
    df = bt.fetch_history(bt.MONTHS)
    print("Cəmi:", len(df), "şam,", df.index[0], "→", df.index[-1])
    for m in compare(df):
        print(m)
        print("-" * 20)
        bot.send(m[:4000])


if __name__ == "__main__":
    main()
