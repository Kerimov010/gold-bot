import io
import json
import os
import time
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT = os.environ["TG_CHAT_ID"]
TD_KEY = os.environ.get("TD_API_KEY", "")

SL, TP1, TP2 = 9.0, 7.0, 15.0
TOUCH = 3.0
CLUSTER = 4.0
COOLDOWN_CANDLES = 12
FRESH_MINUTES = 4.5
BREAK_BUFFER = 1.0
BREAK_LOOKBACK = 36
NO_TRADE_UTC = []

# Yüksək təsirli xəbərlər (UTC). Mənbə: BLS cədvəli və FOMC təqvimi. Cədvəl bitəndə bot xəbərdarlıq edir.
# ABŞ qış vaxtı (EST): 08:30 ET = 13:30 UTC, FOMC 14:00 ET = 19:00 UTC. Yay vaxtı (mart 2027-dən): 12:30 / 18:00 UTC.
NEWS_EVENTS = [
    ("2026-12-04 13:30", "NFP (ABŞ məşğulluq)"),
    ("2026-12-09 19:00", "FOMC faiz qərarı"),
    ("2026-12-10 13:30", "CPI (ABŞ inflyasiya)"),
    ("2026-12-15 13:30", "PPI (ABŞ istehsalçı qiymətləri)"),
    ("2027-01-08 13:30", "NFP (ABŞ məşğulluq)"),
    ("2027-01-13 13:30", "CPI (ABŞ inflyasiya)"),
    ("2027-01-14 13:30", "PPI (ABŞ istehsalçı qiymətləri)"),
    ("2027-01-27 19:00", "FOMC faiz qərarı"),
    ("2027-02-05 13:30", "NFP (ABŞ məşğulluq)"),
    ("2027-02-11 13:30", "CPI (ABŞ inflyasiya)"),
    ("2027-03-10 13:30", "CPI (ABŞ inflyasiya)"),
    ("2027-03-17 18:00", "FOMC faiz qərarı"),
]
NEWS_BEFORE, NEWS_AFTER = 30, 60   # dəqiqə
BIAS_CONFIRM = {"M15": 6, "H1": 12}  # bias dəyişməsi neçə ardıcıl M5 şamla təsdiqlənməlidir


def from_twelve():
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={"symbol": "XAU/USD", "interval": "5min", "outputsize": 1500,
                "timezone": "UTC", "apikey": TD_KEY},
        timeout=20,
    )
    j = r.json()
    if "values" not in j:
        raise RuntimeError(str(j.get("message", j))[:200])
    df = pd.DataFrame(j["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df[["datetime", "open", "high", "low", "close"]].set_index("datetime").sort_index()


def from_kraken():
    r = requests.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": "PAXGUSD", "interval": 5}, timeout=20,
    )
    r.raise_for_status()
    j = r.json()
    if j.get("error"):
        raise RuntimeError(str(j["error"]))
    key = [k for k in j["result"] if k != "last"][0]
    df = pd.DataFrame(
        j["result"][key],
        columns=["t", "open", "high", "low", "close", "vwap", "vol", "n"],
    )
    df["datetime"] = pd.to_datetime(df["t"], unit="s")
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df[["datetime", "open", "high", "low", "close"]].set_index("datetime").sort_index()


def get_m5():
    sources = []
    if TD_KEY:
        sources.append(("Twelve XAU/USD", from_twelve))
    sources.append(("Kraken PAXG", from_kraken))
    errs = []
    for name, fn in sources:
        try:
            df = fn()
            break
        except Exception as e:
            errs.append(f"{name}: {e}")
    else:
        raise RuntimeError(" | ".join(errs))
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    if df.index[-1] + pd.Timedelta(minutes=5) > now:
        df = df.iloc[:-1]
    return df, now, name, errs


def get_bias(df):
    m15 = df["close"].resample("15min").last().dropna()
    e20 = m15.ewm(span=20).mean().iloc[-1]
    e50 = m15.ewm(span=50).mean().iloc[-1]
    if e20 < e50:
        return "SELL"
    if e20 > e50:
        return "BUY"
    return "NONE"


def pivots(df, kind, n=5):
    m15 = df.resample("15min").agg({"high": "max", "low": "min"}).dropna()
    s = m15["high"] if kind == "high" else m15["low"]
    roll = s.rolling(2 * n + 1, center=True)
    ref = roll.max() if kind == "high" else roll.min()
    return list(s[s == ref])


def cluster(levels):
    out, grp = [], []
    for l in sorted(levels):
        if grp and l - grp[0] > CLUSTER:
            out.append(sum(grp) / len(grp))
            grp = []
        grp.append(l)
    if grp:
        out.append(sum(grp) / len(grp))
    return out


def in_blackout(ts):
    for rng in NO_TRADE_UTC:
        a, b = rng.split("-")
        if a <= ts.strftime("%H:%M") <= b:
            return True
    return False


def weekend(ts):
    wd = ts.weekday()
    return wd == 5 or (wd == 4 and ts.hour >= 21) or (wd == 6 and ts.hour < 22)


def check(df):
    bias = get_bias(df)
    c = df.iloc[-1]
    body = max(abs(c.close - c.open), 0.01)
    up = c.high - max(c.open, c.close)
    dn = min(c.open, c.close) - c.low
    sup = cluster(pivots(df, "low"))
    res = cluster(pivots(df, "high"))
    recent = df.iloc[-BREAK_LOOKBACK - 1:-1]

    if bias == "SELL" and c.close < c.open:
        for s in sup:
            broke = ((recent["close"].shift(1) >= s) & (recent["close"] < s - BREAK_BUFFER)).any()
            if broke and c.high >= s - TOUCH and c.high <= s + 4 and c.close < s:
                return "SELL", c.close, s, "Dəstək sınması + retest"
        if up >= 0.5 * body:
            for z in res:
                if c.high >= z - TOUCH and c.high <= z + 6 and c.close < z:
                    return "SELL", c.close, z, "Müqavimətdən rədd olunma"

    if bias == "BUY" and c.close > c.open:
        for r in res:
            broke = ((recent["close"].shift(1) <= r) & (recent["close"] > r + BREAK_BUFFER)).any()
            if broke and c.low <= r + TOUCH and c.low >= r - 4 and c.close > r:
                return "BUY", c.close, r, "Müqavimət sınması + retest"
        if dn >= 0.5 * body:
            for z in sup:
                if c.low <= z + TOUCH and c.low >= z - 6 and c.close > z:
                    return "BUY", c.close, z, "Dəstəkdən sıçrayış"
    return None


def levels_for(side, entry):
    d = -1 if side == "SELL" else 1
    return entry - d * SL, entry + d * TP1, entry + d * TP2


def nearby(df):
    p = df.iloc[-1].close
    lv = cluster(pivots(df, "high") + pivots(df, "low"))
    above = sorted([z for z in lv if z > p + 0.5])[:3]
    below = sorted([z for z in lv if z < p - 0.5], reverse=True)[:3]
    return p, above, below


def status_text(df, src):
    bias = get_bias(df)
    p, above, below = nearby(df)
    icon = {"SELL": "📉", "BUY": "📈"}.get(bias, "➖")
    m15c = df["close"].resample("15min").last().dropna()
    s = f"{icon} Bias (M15): {bias} | H1: {bias_closed(df, 60)}\nQiymət: {p:.2f} (şam {df.index[-1]:%d.%m %H:%M} UTC)\nM15 EMA20: {m15c.ewm(span=20).mean().iloc[-1]:.2f} | EMA50: {m15c.ewm(span=50).mean().iloc[-1]:.2f}\n\n"

    s = f"{icon} Bias (M15): {bias} | H1: {bias_closed(df, 60)}\nQiymət: {p:.2f} (şam {df.index[-1]:%d.%m %H:%M} UTC)\nM15 EMA20: {m15c.ewm(span=20).mean().iloc[-1]:.2f} | EMA50: {m15c.ewm(span=50).mean().iloc[-1]:.2f}\n\n"


    s += "🔴 Müqavimət zonaları:\n"
    s += "\n".join(f"• {z:.2f} (+{z - p:.1f})" for z in above) if above else "• yoxdur"
    s += "\n\n🟢 Dəstək zonaları:\n"
    s += "\n".join(f"• {z:.2f} (-{p - z:.1f})" for z in below) if below else "• yoxdur (qiymət yeni dibdədir)"
    s += "\n\n"

    def sell_plan(z):
        e = z - 1.0
        return (f"SELL: {z:.2f} zonasında M5 rədd şamı bağlananda\n"
                f"  Giriş ≈ {e:.2f} | SL {e + SL:.2f} | TP1 {e - TP1:.2f} | TP2 {e - TP2:.2f}")

    def buy_plan(z):
        e = z + 1.0
        return (f"BUY: {z:.2f} zonasında M5 dönmə şamı bağlananda\n"
                f"  Giriş ≈ {e:.2f} | SL {e - SL:.2f} | TP1 {e + TP1:.2f} | TP2 {e + TP2:.2f}")

    sp = sell_plan(above[0]) if above else "SELL: yaxın zona yoxdur"
    bp = buy_plan(below[0]) if below else "BUY: yaxın zona yoxdur"
    if bias == "SELL":
        s += f"🎯 Əsas plan:\n{sp}\n\nƏks ssenari (riskli):\n{bp}"
    elif bias == "BUY":
        s += f"🎯 Əsas plan:\n{bp}\n\nƏks ssenari (riskli):\n{sp}"
    else:
        s += f"{sp}\n\n{bp}"
    cands = [(z - p, z) for z in above] + [(p - z, z) for z in below]
    if cands:
        s += f"\n\nƏn yaxın zona: {min(cands)[1]:.2f}"
    s += "\nRiskinizi düzgün idarə edin. Maliyyə məsləhəti deyil."
    return s


def _base_chart(df, count):
    d = df.iloc[-count:]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    w = 0.6
    for i, (_, r) in enumerate(d.iterrows()):
        col = "#26a69a" if r.close >= r.open else "#ef5350"
        ax.plot([i, i], [r.low, r.high], color=col, lw=1)
        ax.add_patch(Rectangle((i - w / 2, min(r.open, r.close)), w,
                               max(abs(r.close - r.open), 0.05), color=col))
    return d, fig, ax


def _finish(d, fig, ax, n, title, ys):
    ax.set_xlim(-1, n + 16)
    ax.set_ylim(min(list(ys) + [d.low.min()]) - 3, max(list(ys) + [d.high.max()]) + 3)
    ax.set_title(title, color="white", fontsize=12)
    ax.tick_params(colors="#8b949e")
    xt = list(range(0, n, 15))
    ax.set_xticks(xt)
    ax.set_xticklabels([d.index[i].strftime("%H:%M") for i in xt], color="#8b949e")
    ax.set_xlabel("UTC", color="#8b949e")
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def make_chart(df, side, entry, zone, label):
    d, fig, ax = _base_chart(df, 90)
    sl, tp1, tp2 = levels_for(side, entry)
    n = len(d)
    lines = [
        (zone, "#f0b90b", "--", f"Zona {zone:.2f}"),
        (entry, "#58a6ff", "-", f"Giris {entry:.2f}"),
        (sl, "#ff5252", "-", f"SL {sl:.2f}"),
        (tp1, "#00e676", "-", f"TP1 {tp1:.2f}"),
        (tp2, "#00e676", "--", f"TP2 {tp2:.2f}"),
    ]
    for y, col, ls, txt in lines:
        ax.hlines(y, -1, n, colors=col, linestyles=ls, linewidth=1.2)
        ax.text(n + 1, y, txt, color=col, va="center", fontsize=9)
    return _finish(d, fig, ax, n, f"XAUUSD M5 | {side} | {label}", [sl, tp1, tp2])


def make_status_chart(df):
    bias = get_bias(df)
    p, above, below = nearby(df)
    d, fig, ax = _base_chart(df, 120)
    n = len(d)
    ys = []
    for z in above:
        ax.hlines(z, -1, n, colors="#ef5350", linestyles="-", linewidth=1.4)
        ax.text(n + 1, z, f"Müqavimət {z:.2f}", color="#ef5350", va="center", fontsize=8)
        ys.append(z)
    for z in below:
        ax.hlines(z, -1, n, colors="#26a69a", linestyles="-", linewidth=1.4)
        ax.text(n + 1, z, f"Dəstək {z:.2f}", color="#26a69a", va="center", fontsize=8)
        ys.append(z)
    ax.hlines(p, -1, n, colors="#58a6ff", linestyles=":", linewidth=1)
    return _finish(d, fig, ax, n, f"XAUUSD M5 | Bias: {bias} | Qiymət: {p:.2f}", ys + [p])


STATE_FILE = os.environ.get("STATE_FILE", "state.json")
SPREAD_EST = 0.30
MAX_HOLD = 72
BACKTEST_REF = "6 aylıq backtest: SL ~41%, TP2 ~30%, BE ~27%, PF ~1.08"


def _retry(fn, tries=3):
    last = None
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            print(f"Telegram xetasi ({k + 1}/{tries}):", e)
            time.sleep(2 * (k + 1))
    raise last


def send(text):
    def go():
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text}, timeout=20,
        )
        if not r.ok:
            raise RuntimeError("sendMessage: " + r.text[:200])
    _retry(go)


def send_photo(buf, caption):
    def go():
        buf.seek(0)
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
            data={"chat_id": TG_CHAT, "caption": caption[:1000]},
            files={"photo": ("signal.png", buf, "image/png")}, timeout=30,
        )
        if not r.ok:
            raise RuntimeError("sendPhoto: " + r.text[:200])
    _retry(go)


def send_status(df, src):
    text = status_text(df, src) + f"\n{news_line(df.index[-1])}\n(Mənbə: {src})"
    try:
        send_photo(make_status_chart(df), text)
    except Exception as e:
        print("Status sekli gonderilmedi:", e)
        send(text)


def closed_bars(df, minutes):
    rule = f"{minutes}min"
    bars = pd.DataFrame({
        "open": df["open"].resample(rule).first(),
        "high": df["high"].resample(rule).max(),
        "low": df["low"].resample(rule).min(),
        "close": df["close"].resample(rule).last(),
    }).dropna()
    last_start = df.index[-1]
    bucket_start = last_start.floor(rule)
    bucket_end = bucket_start + pd.Timedelta(minutes=minutes)
    if len(bars) and bars.index[-1] == bucket_start and last_start + pd.Timedelta(minutes=5) < bucket_end:
        bars = bars.iloc[:-1]
    return bars


def bias_closed(df, minutes):
    b = closed_bars(df, minutes)["close"]
    if len(b) < 55:
        return "NEUTRAL"
    gap = b.ewm(span=20).mean().iloc[-1] - b.ewm(span=50).mean().iloc[-1]
    thr = 1.0 if minutes <= 15 else 2.0
    if gap < -thr:
        return "SELL"
    if gap > thr:
        return "BUY"
    return "NEUTRAL"


def atr_ratio(df, period=14):
    b = closed_bars(df, 15)
    if len(b) < period * 3:
        return -1.0
    pc = b["close"].shift(1)
    tr = pd.concat([b["high"] - b["low"], (b["high"] - pc).abs(), (b["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().dropna()
    return float(atr.iloc[-1] / atr.median())


def baku(ts):
    return (ts + pd.Timedelta(hours=4)).strftime("%d.%m %H:%M")


def _news_list():
    return [(pd.Timestamp(ts), name) for ts, name in NEWS_EVENTS]


def news_block(ts):
    for et, name in _news_list():
        if et - pd.Timedelta(minutes=NEWS_BEFORE) <= ts <= et + pd.Timedelta(minutes=NEWS_AFTER):
            return et, name
    return None


def next_news(ts):
    fut = [(et, n) for et, n in _news_list() if et > ts]
    return min(fut) if fut else None


def news_line(ts):
    nx = next_news(ts)
    if not nx:
        return "Xəbər cədvəli bitib, yenilənməlidir."
    return f"Növbəti xəbər: {nx[1]}, {baku(nx[0])} (Bakı)"


def bias_alerts(df, state, t):
    price = df.iloc[-1].close
    cur = {"M15": bias_closed(df, 15), "H1": bias_closed(df, 60)}
    store = state.setdefault("bias", {})
    pend = state.setdefault("bias_pend", {})
    for tf, new in cur.items():
        old = store.get(tf)
        if old is None:
            store[tf] = new
            continue
        if new == old:
            pend.pop(tf, None)
            continue
        p = pend.get(tf)
        if p and p[0] == new:
            if p[2] != str(t):
                p[1] += 1
                p[2] = str(t)
        else:
            p = [new, 1, str(t)]
            pend[tf] = p
        if p[1] >= BIAS_CONFIRM[tf]:
            send(f"🔄 {tf} Bias dəyişdi: {old} → {new}\n"
                 f"Qiymət: {price:.2f}\nVaxt: {baku(t + pd.Timedelta(minutes=5))} (Bakı)\n"
                 f"({p[1]} ardıcıl bağlanmış M5 şamla təsdiqləndi)")
            store[tf] = new
            pend.pop(tf, None)


def news_notice(state, t):
    nb = news_block(t + pd.Timedelta(minutes=5))
    if nb:
        et, name = nb
        if str(et) not in state["news_notified"]:
            send(f"⏸ Xəbər: {name}\nVaxt: {baku(et)} (Bakı)\n"
                 f"Xəbərdən {NEWS_BEFORE} dəq əvvəl və {NEWS_AFTER} dəq sonra yeni siqnal verilmir.")
            state["news_notified"] = (state["news_notified"] + [str(et)])[-30:]
    return nb


def news_stale_warning(state, t):
    iso = t.isocalendar()
    wk = f"{iso[0]}-W{iso[1]:02d}"
    if next_news(t) is None and state["last_news_warn"] != wk:
        send("⚠️ Xəbər filtri: cədvəldəki bütün xəbərlərin vaxtı keçib, filtr işləmir. "
             "bot.py-dakı NEWS_EVENTS siyahısını yeniləmək lazımdır.")
        state["last_news_warn"] = wk


def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            s = json.load(f)
    except FileNotFoundError:
        s = {}
    s.setdefault("next_id", 1)
    s.setdefault("open", [])
    s.setdefault("closed", [])
    s.setdefault("last_daily", "")
    s.setdefault("last_weekly", "")
    s.setdefault("last_status", "")
    s.setdefault("bias", {})
    s.setdefault("bias_pend", {})
    s.setdefault("news_notified", [])
    s.setdefault("last_news_warn", "")
    return s


def save_state(s):
    s["closed"] = s["closed"][-500:]
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE_FILE)


def resolve(df, sig):
    """Siqnalın nəticəsi. Qaytarır: (bitdi, tp1_vurulub, pnl, nəticə). Giriş = siqnaldan sonrakı şamın açılışı."""
    ts = pd.Timestamp(sig["ts"])
    if ts not in df.index:
        if df.index[0] > ts:
            return True, False, 0.0, "NAMƏLUM"
        return False, False, 0.0, None
    i = df.index.get_loc(ts)
    n = len(df)
    if i + 1 >= n:
        return False, False, 0.0, None
    o, h, l, c = (df[x].values for x in ["open", "high", "low", "close"])
    d = -1 if sig["side"] == "SELL" else 1
    entry = o[i + 1]
    sl_d, tp1_d, tp2_d = SL, TP1, TP2
    sl, tp1, tp2 = entry - d * sl_d, entry + d * tp1_d, entry + d * tp2_d
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
                return True, False, -sl_d, "SL"
            if f1:
                tp1_hit = True
                if f2:
                    return True, True, 0.5 * tp1_d + 0.5 * tp2_d, "TP2"
        else:
            if adverse:
                return True, True, 0.5 * tp1_d, "BE"
            if f2:
                return True, True, 0.5 * tp1_d + 0.5 * tp2_d, "TP2"
    if i + MAX_HOLD <= n - 1:
        open_pnl = d * (c[i + MAX_HOLD] - entry)
        pnl = 0.5 * tp1_d + 0.5 * open_pnl if tp1_hit else open_pnl
        return True, tp1_hit, pnl, "TIME"
    return False, tp1_hit, 0.0, None


def summarize(trades):
    tr = [x for x in trades if x["outcome"] != "NAMƏLUM"]
    if not tr:
        return None
    pnl = [x["gross"] - SPREAD_EST for x in tr]
    wins = sum(p for p in pnl if p > 0)
    losses = -sum(p for p in pnl if p < 0)
    cnt = {k: sum(1 for x in tr if x["outcome"] == k) for k in ("TP2", "BE", "SL", "TIME")}
    return {"n": len(tr), "net": sum(pnl), "pf": (wins / losses) if losses > 0 else None, "cnt": cnt}


def fmt_sum(s):
    if not s:
        return "hələ bağlanmış əməliyyat yoxdur"
    pf = "∞" if s["pf"] is None else f"{s['pf']:.2f}"
    c, n = s["cnt"], s["n"]
    return (f"{n} əməliyyat | TP2 {c['TP2']} ({c['TP2'] * 100 // n}%), BE {c['BE']} ({c['BE'] * 100 // n}%), "
            f"SL {c['SL']} ({c['SL'] * 100 // n}%), vaxt {c['TIME']} | PF {pf} | Net {s['net']:+.1f}$")


def daily_text(state, d, title="Gündəlik hesabat"):
    issued = [x for x in state["open"] + state["closed"] if x["ts"][:10] == d]
    done = [x for x in state["closed"] if x.get("closed_at") == d]
    t = f"📊 {title} ({d})\n"
    t += f"Bu gün siqnal: {len(issued)} | hələ açıq: {len(state['open'])}\n\n"
    t += "Bu gün bağlananlar:\n" + fmt_sum(summarize(done)) + "\n\n"
    t += "Ümumi:\n" + fmt_sum(summarize(state["closed"])) + "\n\n"
    t += f"Net: 0.01 lot-da təxmini $, spread {SPREAD_EST} çıxılıb.\nMaliyyə məsləhəti deyil."
    return t


def weekly_text(state, wk):
    last_day = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    week = [x for x in state["closed"]
            if x.get("closed_at") and pd.Timestamp(x["closed_at"]) >= last_day - pd.Timedelta(days=6)]
    t = f"📈 Həftəlik hesabat ({wk})\n\nBu həftə:\n" + fmt_sum(summarize(week)) + "\n\n"
    t += "Ümumi:\n" + fmt_sum(summarize(state["closed"])) + "\n\n"
    labels = sorted(set(x["label"] for x in state["closed"]))
    if labels:
        t += "Setup üzrə (ümumi):\n"
        for lb in labels:
            t += f"• {lb}: " + fmt_sum(summarize([x for x in state["closed"] if x["label"] == lb])) + "\n"
        t += "\n"
    t += "Müqayisə üçün " + BACKTEST_REF + ".\n"
    total = summarize(state["closed"])
    if total and total["n"] < 30:
        t += "Diqqət: 30-dan az əməliyyat var, nəticəyə güvənmək tezdir.\n"
    t += "Maliyyə məsləhəti deyil."
    return t[:4000]


def process_open(df, state):
    still = []
    today = df.index[-1].strftime("%Y-%m-%d")
    for s in state["open"]:
        done, tp1, pnl, out = resolve(df, s)
        head = f"Siqnal #{s['id']} ({s['side']}, {s['label']})"
        if not done:
            if tp1 and not s.get("tp1_notified"):
                send(f"🎯 {head}: TP1 vuruldu. Yarısını bağla, SL-i girişə çək.")
                s["tp1_notified"] = True
            still.append(s)
            continue
        if out == "TP2":
            msg = f"✅ {head}: TP2 vuruldu. Nəticə ≈ {pnl:+.1f}$"
        elif out == "BE":
            msg = f"🟡 {head}: TP1 vuruldu, sonra girişə qayıdıb (breakeven). Nəticə ≈ {pnl:+.1f}$"
        elif out == "SL":
            msg = f"❌ {head}: SL vuruldu. Nəticə ≈ {pnl:+.1f}$"
        elif out == "TIME":
            msg = f"⏱ {head}: 6 saat keçdi, nəticə ≈ {pnl:+.1f}$ (qapanış qiymətinə görə)"
        else:
            msg = f"⚠️ {head}: məlumat pəncərəsindən çıxdı, nəticə hesablanmadı."
        send(msg + "\n(0.01 lotda, spreadsiz)")
        state["closed"].append({"id": s["id"], "ts": s["ts"], "side": s["side"], "label": s["label"],
                                "outcome": out, "tp1": bool(tp1), "gross": float(pnl), "closed_at": today})
    state["open"] = still


def maybe_reports(state, t):
    d = t.strftime("%Y-%m-%d")
    if t.weekday() < 5 and t.hour == 17 and state["last_daily"] != d:
        send(daily_text(state, d))
        state["last_daily"] = d
    iso = t.isocalendar()
    wk = f"{iso[0]}-W{iso[1]:02d}"
    if t.weekday() == 4 and t.hour == 19 and state["last_weekly"] != wk:
        send(weekly_text(state, wk))
        state["last_weekly"] = wk


def _run(state):
    manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    if manual:
        send("Bot işləyir. Aşağıda nümunə analiz və hesabat gəlir.")

    df, now, src, errs = get_m5()
    t = df.index[-1]
    closed_ago = (now - (t + pd.Timedelta(minutes=5))).total_seconds() / 60
    print(f"Son sam: {t}, baglanib: {closed_ago:.1f} deq evvel, menbe: {src}")

    if manual:
        if errs:
            send("Xəbərdarlıq: " + " | ".join(errs)[:300])
        send_status(df, src)
        send(daily_text(state, t.strftime("%Y-%m-%d"), "TEST hesabat"))

    process_open(df, state)

    if weekend(t):
        print("Bazar bagli, kecildi.")
        return
    if closed_ago > FRESH_MINUTES:
        print("Sam kohnedir, kecildi.")
        return

    if not manual:
        maybe_reports(state, t)
        if t.minute in (25, 55) and state["last_status"] != str(t):
            send_status(df, src)
            state["last_status"] = str(t)
        bias_alerts(df, state, t)
        news_stale_warning(state, t)
    nb = news_notice(state, t) if not manual else news_block(t + pd.Timedelta(minutes=5))
    if nb:
        print("Xeber filtri:", nb[1])
        return

    if in_blackout(t):
        print("Xeber saati, kecildi.")
        return
    for k in range(1, COOLDOWN_CANDLES + 1):
        if check(df.iloc[:-k]):
            print("Cooldown, kecildi.")
            return

    sig = check(df)
    if not sig:
        print("Siqnal yoxdur.")
        return
    if any(x["ts"] == str(t) for x in state["open"] + state["closed"]):
        print("Bu siqnal artiq gonderilib.")
        return

    side, entry, zone, label = sig
    sl, tp1, tp2 = levels_for(side, entry)
    sid = state["next_id"]
    caption = (
        f"🚨 GOLD M5 | {side} | #{sid}\n"
        f"Setup: {label}\n\n"
        f"Giriş: {entry:.2f} (indi, market)\n"
        f"SL: {sl:.2f}\n"
        f"TP1: {tp1:.2f} (yarısını bağla, SL-i girişə çək)\n"
        f"TP2: {tp2:.2f}\n"
        f"Zona: {zone:.2f}\n"
        f"Risk/mükafat: 1:{TP1 / SL:.1f} (TP1), 1:{TP2 / SL:.1f} (TP2)\n"
    )
    if not src.startswith("Twelve"):
        caption += "DİQQƏT: mənbə spot qızıl deyil, ~17$ fərq var.\n"
    if atr_ratio(df) > 2.0:
        caption += "⚠️ Volatillik normaldan 2 dəfədən çox yüksəkdir (xəbər və ya geniş spread ola bilər).\n"
    caption += "Məsafələri brokerin qiymətinə tətbiq et.\nMaliyyə məsləhəti deyil."
    try:
        send_photo(make_chart(df, side, entry, zone, label), caption)
    except Exception as e:
        print("Sekil gonderilmedi:", e)
        send(caption)
    state["open"].append({"id": sid, "ts": str(t), "side": side, "label": label,
                          "zone": float(zone), "tp1_notified": False})
    state["next_id"] = sid + 1
    print(caption)


def main():
    state = load_state()
    try:
        _run(state)
    finally:
        save_state(state)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
        if manual or pd.Timestamp.now(tz="UTC").minute < 5:
            try:
                send("XETA: " + repr(e)[:400])
            except Exception:
                pass
        raise
