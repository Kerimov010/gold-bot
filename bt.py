"""
Backtest: canlı botun (bot.py) strategiyasını tarixi XAU/USD M5 məlumatında yoxlayır.
- Yalnız bağlanmış şamlar, gələcək şamdan istifadə yoxdur.
- Giriş siqnal şamından SONRAKI şamın açılışındadır.
- Spread/komissiya hər əməliyyatdan çıxılır.
- Eyni şamda həm SL, həm TP olarsa, SL birinci sayılır (mühafizəkar).
Nəticələr: dollar = 0.01 lot-da təxmini ABŞ dolları (qiymət dollarına bərabər).
"""
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import requests

import bot

MONTHS = int(os.environ.get("MONTHS", "6"))
SPREAD = float(os.environ.get("SPREAD", "0.30"))
COMM = float(os.environ.get("COMM", "0.0"))
WINDOW = 1500
MAX_HOLD = 72


def fetch_history(months):
    key = os.environ["TD_API_KEY"]
    target = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=30 * months)
    frames, end, tries = [], None, 0
    while True:
        p = {"symbol": "XAU/USD", "interval": "5min", "outputsize": 5000,
             "timezone": "UTC", "apikey": key}
        if end is not None:
            p["end_date"] = end.strftime("%Y-%m-%d %H:%M:%S")
        j = requests.get("https://api.twelvedata.com/time_series", params=p, timeout=60).json()
        if "values" not in j:
            msg = str(j.get("message", j))
            if j.get("code") == 429 and tries < 5:
                tries += 1
                print("Limit, gozleyirem...")
                time.sleep(65)
                continue
            print("Data dayandi:", msg[:200])
            break
        tries = 0
        df = pd.DataFrame(j["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype(float)
        frames.append(df[["datetime", "open", "high", "low", "close"]])
        earliest = df["datetime"].min()
        print("Alindi:", len(df), "sam, en erkən:", earliest)
        if earliest <= target or len(df) < 5000:
            break
        end = earliest - pd.Timedelta(minutes=5)
        time.sleep(8)
    if not frames:
        raise RuntimeError("Məlumat alınmadı")
    out = pd.concat(frames).drop_duplicates("datetime").set_index("datetime").sort_index()
    return out[out.index >= target]


def closed_resample(w, minutes):
    rule = f"{minutes}min"
    bars = pd.DataFrame({
        "open": w["open"].resample(rule).first(),
        "high": w["high"].resample(rule).max(),
        "low": w["low"].resample(rule).min(),
        "close": w["close"].resample(rule).last(),
    }).dropna()
    last_start = w.index[-1]
    bucket_start = last_start.floor(rule)
    bucket_end = bucket_start + pd.Timedelta(minutes=minutes)
    if len(bars) and bars.index[-1] == bucket_start and last_start + pd.Timedelta(minutes=5) < bucket_end:
        bars = bars.iloc[:-1]
    return bars


def h1_bias(w):
    b = closed_resample(w, 60)["close"]
    if len(b) < 55:
        return "NONE"
    e20 = b.ewm(span=20).mean().iloc[-1]
    e50 = b.ewm(span=50).mean().iloc[-1]
    return "SELL" if e20 < e50 else "BUY" if e20 > e50 else "NONE"


def structure(w, n=2):
    b = closed_resample(w, 15)
    hi, lo, cl = b["high"].values, b["low"].values, b["close"].values
    last_sh = last_sl = None
    direction, kind = 0, None
    for t in range(len(b)):
        p = t - n
        if p >= n:
            if hi[p] == hi[p - n:p + n + 1].max():
                last_sh = hi[p]
            if lo[p] == lo[p - n:p + n + 1].min():
                last_sl = lo[p]
        if last_sh is not None and cl[t] > last_sh:
            kind = "BOS" if direction >= 0 else "CHOCH"
            direction, last_sh = 1, None
        elif last_sl is not None and cl[t] < last_sl:
            kind = "BOS" if direction <= 0 else "CHOCH"
            direction, last_sl = -1, None
    return direction, kind


def sweep(w, side, look=24, recent=3):
    h, l, c = w["high"].values, w["low"].values, w["close"].values
    n = len(w)
    for k in range(n - recent, n):
        if k - look < 0:
            continue
        if side == "SELL":
            ref = h[k - look:k].max()
            if h[k] > ref and c[k] < ref:
                return True
        else:
            ref = l[k - look:k].min()
            if l[k] < ref and c[k] > ref:
                return True
    return False


def atr15(w, period=14):
    b = closed_resample(w, 15)
    pc = b["close"].shift(1)
    tr = pd.concat([b["high"] - b["low"], (b["high"] - pc).abs(), (b["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def adx15(w, period=14):
    b = closed_resample(w, 15)
    if len(b) < period * 3:
        return -1.0
    h, l, c = b["high"].values, b["low"].values, b["close"].values
    up = h[1:] - h[:-1]
    dn = l[:-1] - l[1:]
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])

    def wilder(x, p):
        out = np.zeros(len(x))
        out[p - 1] = x[:p].sum()
        for i in range(p, len(x)):
            out[i] = out[i - 1] - out[i - 1] / p + x[i]
        return out[p - 1:]

    a, pp, mm = wilder(tr, period), wilder(plus, period), wilder(minus, period)
    pdi, mdi = 100 * pp / a, 100 * mm / a
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi + 1e-12)
    if len(dx) < period:
        return -1.0
    adx = dx[:period].mean()
    for v in dx[period:]:
        adx = (adx * (period - 1) + v) / period
    return float(adx)


def atr_ratio(w, period=14):
    b = closed_resample(w, 15)
    if len(b) < period * 3:
        return -1.0
    pc = b["close"].shift(1)
    tr = pd.concat([b["high"] - b["low"], (b["high"] - pc).abs(), (b["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().dropna()
    return float(atr.iloc[-1] / atr.median())


def features(w, side):
    want = 1 if side == "BUY" else -1
    sd, kind = structure(w)
    return {
        "h1_agree": h1_bias(w) == side,
        "struct_agree": sd == want,
        "choch_against": kind == "CHOCH" and sd == -want,
        "sweep": sweep(w, side),
        "adx": adx15(w),
        "atr_ratio": atr_ratio(w),
    }


def simulate(o, h, l, c, i, side, sl_d, tp1_d, tp2_d):
    n = len(o)
    d = -1 if side == "SELL" else 1
    entry = o[i + 1]
    sl = entry - d * sl_d
    tp1 = entry + d * tp1_d
    tp2 = entry + d * tp2_d
    tp1_hit = False
    last = min(i + MAX_HOLD, n - 1)
    for j in range(i + 1, last + 1):
        if d == -1:
            adverse = h[j] >= (entry if tp1_hit else sl)
            f1, f2 = l[j] <= tp1, l[j] <= tp2
        else:
            adverse = l[j] <= (entry if tp1_hit else sl)
            f1, f2 = h[j] >= tp1, h[j] >= tp2
        if not tp1_hit:
            if adverse:
                return -sl_d, "SL", False
            if f1:
                tp1_hit = True
                if f2:
                    return 0.5 * tp1_d + 0.5 * tp2_d, "TP2", True
        else:
            if adverse:
                return 0.5 * tp1_d, "BE", True
            if f2:
                return 0.5 * tp1_d + 0.5 * tp2_d, "TP2", True
    open_pnl = d * (c[last] - entry)
    if tp1_hit:
        return 0.5 * tp1_d + 0.5 * open_pnl, "TIME", True
    return open_pnl, "TIME", False


def stats(rows, pk="pnl", ok="out", tk="tp1"):
    n = len(rows)
    if n == 0:
        return None
    pnl = np.array([r[pk] for r in rows])
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    cum = np.cumsum(pnl)
    outs = Counter(r[ok] for r in rows)
    return {
        "n": n, "wr": (pnl > 0).mean() * 100,
        "pf": wins / losses if losses > 0 else float("inf"),
        "net": pnl.sum(), "avg": pnl.mean(),
        "dd": (np.maximum.accumulate(cum) - cum).max(),
        "sl": outs["SL"] / n * 100, "be": outs["BE"] / n * 100, "tp2": outs["TP2"] / n * 100,
        "tp1": sum(1 for r in rows if r[tk]) / n * 100,
    }


def line(name, s):
    if s is None:
        return f"{name}: əməliyyat yoxdur"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    small = " (kiçik nümunə!)" if s["n"] < 30 else ""
    return (f"{name}: n={s['n']} WR={s['wr']:.0f}% PF={pf} Net={s['net']:+.0f}$ "
            f"DD={s['dd']:.0f}$ | SL {s['sl']:.0f}% TP1 {s['tp1']:.0f}% "
            f"TP2 {s['tp2']:.0f}% BE {s['be']:.0f}%{small}")


def run(df, log=print):
    idx = df.index
    o, h, l, c = (df[x].values for x in ["open", "high", "low", "close"])
    N = len(df)
    raw = [None] * N
    t0 = time.time()
    for i in range(WINDOW - 1, N - 1):
        if bot.weekend(idx[i]):
            continue
        raw[i] = bot.check(df.iloc[i - WINDOW + 1:i + 1])
        if i % 10000 == 0:
            log(f"Analiz: {i}/{N}, {time.time() - t0:.0f}s")
    rows = []
    for i in range(N - 1):
        if raw[i] is None:
            continue
        if any(raw[j] is not None for j in range(max(0, i - bot.COOLDOWN_CANDLES), i)):
            continue
        side, _, zone, label = raw[i]
        w = df.iloc[max(0, i - WINDOW + 1):i + 1]
        f = features(w, side)
        gross, out, tp1 = simulate(o, h, l, c, i, side, bot.SL, bot.TP1, bot.TP2)
        rows.append({"ts": idx[i], "side": side, "label": label, "gross": gross,
                     "out": out, "tp1": tp1, **f})
    return rows


COSTS = (0.3, 0.5, 0.7)

VARIANTS = [
    ("BAZA (indiki bot)", lambda r: True),
    ("A: yalnız trend bazarı (ADX≥20)", lambda r: r["adx"] >= 20),
    ("B: M15 struktur uyğun", lambda r: r["struct_agree"]),
    ("C: ADX≥20 + struktur uyğun", lambda r: r["adx"] >= 20 and r["struct_agree"]),
    ("D: normal volatillik (0.6-1.8×)", lambda r: 0.6 <= r["atr_ratio"] <= 1.8),
    ("E: ADX≥20 + struktur + normal vol.", lambda r: r["adx"] >= 20 and r["struct_agree"] and 0.6 <= r["atr_ratio"] <= 1.8),
    ("F: müqavimət-sınması BUY çıxarılıb (!)", lambda r: r["label"] != "Müqavimət sınması + retest"),
]


def pf_of(p):
    wins, losses = p[p > 0].sum(), -p[p < 0].sum()
    return wins / losses if losses > 0 else float("inf")


def fmt_pf(x):
    return "∞" if x == float("inf") else f"{x:.2f}"


def longest_losing(p):
    best = cur = 0
    for v in p:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


def evaluate(df, rows, name, fn):
    sel = sorted([r for r in rows if fn(r)], key=lambda r: r["ts"])
    if len(sel) < 5:
        return f"{name}\n  əməliyyat çox azdır (n={len(sel)})", False
    g = np.array([r["gross"] for r in sel])
    p3, p5, p7 = (g - c for c in COSTS)
    cum = np.cumsum(p3)
    dd = (np.maximum.accumulate(cum) - cum).max()
    cut = df.index[0] + (df.index[-1] - df.index[0]) * 2 / 3
    first = np.array([r["gross"] - COSTS[0] for r in sel if r["ts"] < cut])
    last = np.array([r["gross"] - COSTS[0] for r in sel if r["ts"] >= cut])
    pf_first = pf_of(first) if len(first) else 0.0
    pf_last = pf_of(last) if len(last) else 0.0
    months = {}
    for r in sel:
        k = r["ts"].strftime("%Y-%m")
        months[k] = months.get(k, 0.0) + r["gross"] - COSTS[0]
    pos = sum(1 for v in months.values() if v > 0)
    ok1 = len(sel) >= 100
    ok2 = pf_first >= 1.05 and pf_last >= 1.05
    ok3 = pf_of(p5) >= 1.10
    ok4 = pos / len(months) >= 0.6
    passed = ok1 and ok2 and ok3 and ok4
    mstr = " ".join(f"{v:+.0f}" for _, v in sorted(months.items()))
    txt = (f"{name}\n"
           f"  n={len(sel)} | PF {fmt_pf(pf_of(p3))} (spread 0.5: {fmt_pf(pf_of(p5))}, 0.7: {fmt_pf(pf_of(p7))})"
           f" | Net {p3.sum():+.0f}$ | DD {dd:.0f}$\n"
           f"  İlk 2/3 PF {fmt_pf(pf_first)}, son 1/3 PF {fmt_pf(pf_last)} | Müsbət aylar {pos}/{len(months)}"
           f" | ən uzun zərər seriyası {longest_losing(p3)}\n"
           f"  Aylıq $: {mstr}\n"
           f"  Meyar (n≥100, hər iki dövrdə PF≥1.05, spread 0.5-də PF≥1.10, aylar≥60% müsbət): "
           f"{'KEÇDİ ✅' if passed else 'KEÇMƏDİ ❌'} "
           f"[{'✓' if ok1 else '✗'}{'✓' if ok2 else '✗'}{'✓' if ok3 else '✗'}{'✓' if ok4 else '✗'}]")
    return txt, passed


def report(df, rows):
    span_days = (df.index[-1] - df.index[0]).days or 1
    head = (f"BACKTEST v2 (rejim təhlili)\nMəlumat: {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d} "
            f"({span_days} gün)\nGiriş: siqnaldan sonrakı şamın açılışı. Eyni şamda SL+TP: SL birinci.\n"
            f"Net/DD: 0.01 lot-da təxmini $. Əməliyyatlar: {len(rows)}\n"
            f"Meyarlar nəticələrə baxmadan əvvəl təyin olunub. (!) = nəticəyə baxandan sonra seçilib.\n")
    if not rows:
        return [head + "Heç bir siqnal tapılmadı."]
    blocks, passed_names = [], []
    for name, fn in VARIANTS:
        txt, ok = evaluate(df, rows, name, fn)
        blocks.append(txt)
        if ok:
            passed_names.append(name)
    msgs, cur = [head], ""
    for b in blocks:
        if len(cur) + len(b) > 3300:
            msgs.append(cur)
            cur = ""
        cur += b + "\n\n"
    msgs.append(cur)
    final = ("YEKUN: " + ("meyarı keçən variant(lar): " + "; ".join(passed_names) if passed_names
             else "heç bir variant bütün meyarları keçmədi.") +
             "\n7 variant sınandığı üçün təsadüfən keçmə ehtimalı var, buna görə keçən variantı canlı bota qoymazdan "
             "əvvəl yeni məlumatda (növbəti aylarda) təkrar yoxlamaq lazımdır.")
    msgs.append(final)
    return msgs


def main():
    df = fetch_history(MONTHS)
    print("Cəmi:", len(df), "şam,", df.index[0], "→", df.index[-1])
    rows = run(df)
    msgs = report(df, rows)
    for m in msgs:
        print(m)
        print("-" * 20)
        bot.send(m[:4000])


if __name__ == "__main__":
    main()
