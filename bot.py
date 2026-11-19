import io
import os
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


def explain(df):
    bias = get_bias(df)
    p = df.iloc[-1].close
    hi = [z for z in cluster(pivots(df, "high")) if z > p]
    lo = [z for z in cluster(pivots(df, "low")) if z < p]
    s = f"Trend: {bias}\nQiymət: {p:.2f}"
    if hi:
        z = min(hi)
        s += f"\nYaxın müqavimət: {z:.2f} (+{z - p:.1f})"
    else:
        s += "\nYaxın müqavimət: yoxdur"
    if lo:
        z = max(lo)
        s += f"\nYaxın dəstək: {z:.2f} (-{p - z:.1f})"
    else:
        s += "\nYaxın dəstək: yoxdur"
    return s


def levels_for(side, entry):
    d = -1 if side == "SELL" else 1
    return entry - d * SL, entry + d * TP1, entry + d * TP2


def make_chart(df, side, entry, zone, label):
    d = df.iloc[-90:]
    sl, tp1, tp2 = levels_for(side, entry)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=110)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    w = 0.6
    for i, (_, r) in enumerate(d.iterrows()):
        col = "#26a69a" if r.close >= r.open else "#ef5350"
        ax.plot([i, i], [r.low, r.high], color=col, lw=1)
        ax.add_patch(Rectangle((i - w / 2, min(r.open, r.close)), w,
                               max(abs(r.close - r.open), 0.05), color=col))
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
    ax.set_xlim(-1, n + 16)
    ax.set_ylim(min(d.low.min(), sl, tp1, tp2) - 3, max(d.high.max(), sl, tp1, tp2) + 3)
    ax.set_title(f"XAUUSD M5 | {side} | {label}", color="white", fontsize=12)
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


def send(text):
    requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        json={"chat_id": TG_CHAT, "text": text}, timeout=20,
    )


def send_photo(buf, caption):
    r = requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
        data={"chat_id": TG_CHAT, "caption": caption[:1000]},
        files={"photo": ("signal.png", buf, "image/png")}, timeout=30,
    )
    if not r.ok:
        raise RuntimeError("sendPhoto: " + r.text[:200])


def main():
    manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    if manual:
        send("Bot işləyir. Siqnal yalnız şərt uyğun gələndə gələcək.")

    df, now, src, errs = get_m5()
    t = df.index[-1]
    closed_ago = (now - (t + pd.Timedelta(minutes=5))).total_seconds() / 60
    print(f"Son sam: {t}, baglanib: {closed_ago:.1f} deq evvel, menbe: {src}")

    if manual:
        txt = f"Mənbə: {src}\nSon şam: {t} UTC\n" + explain(df)
        if errs:
            txt += "\nXəbərdarlıq: " + " | ".join(errs)[:300]
        send(txt)
        try:
            p = df.iloc[-1].close
            send_photo(make_chart(df, "SELL", p, p + 3, "TEST (siqnal deyil)"),
                       "Test şəkil. Bu siqnal deyil, yalnız şəklin işlədiyini yoxlayır.")
        except Exception as e:
            send("Şəkil xətası: " + repr(e)[:300])

    if weekend(t):
        print("Bazar bagli, kecildi.")
        return
    if closed_ago > FRESH_MINUTES:
        print("Sam kohnedir, kecildi.")
        return

    if t.minute == 55:
        send("Saatlıq status (bot işləyir)\n" + explain(df) + f"\nMənbə: {src}")

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

    side, entry, zone, label = sig
    sl, tp1, tp2 = levels_for(side, entry)
    caption = (
        f"GOLD M5 | {side}\n"
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
    caption += "Qiymət brokerdən fərqli ola bilər, məsafələri brokerin qiymətinə tətbiq et.\nMaliyyə məsləhəti deyil."
    try:
        send_photo(make_chart(df, side, entry, zone, label), caption)
    except Exception as e:
        print("Sekil gonderilmedi:", e)
        send(caption)
    print(caption)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
        if manual or pd.Timestamp.now(tz="UTC").minute < 5:
            send("XETA: " + repr(e)[:400])
        raise
