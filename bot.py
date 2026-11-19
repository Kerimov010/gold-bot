import os
import requests
import pandas as pd

TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT = os.environ["TG_CHAT_ID"]
TD_KEY = os.environ.get("TD_API_KEY", "")

SL, TP1, TP2 = 9.0, 7.0, 15.0
TOUCH = 3.0
CLUSTER = 4.0
COOLDOWN_CANDLES = 6
FRESH_MINUTES = 4.5
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

    if bias == "SELL" and c.close < c.open and up >= 0.5 * body:
        for z in cluster(pivots(df, "high")):
            if c.high >= z - TOUCH and c.high <= z + 6 and c.close < z:
                return "SELL", c.close, z
    if bias == "BUY" and c.close > c.open and dn >= 0.5 * body:
        for z in cluster(pivots(df, "low")):
            if c.low <= z + TOUCH and c.low >= z - 6 and c.close > z:
                return "BUY", c.close, z
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


def send(text):
    requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        json={"chat_id": TG_CHAT, "text": text}, timeout=20,
    )


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

    side, entry, zone = sig
    d = -1 if side == "SELL" else 1
    msg = (
        f"GOLD M5 | {side} ({src})\n"
        f"Zona: {zone:.2f}\n"
        f"Giriş: {entry:.2f}\n"
        f"SL: {entry - d * SL:.2f}\n"
        f"TP1: {entry + d * TP1:.2f} (yarısını bağla, SL-i girişə çək)\n"
        f"TP2: {entry + d * TP2:.2f}\n"
    )
    if not src.startswith("Twelve"):
        msg += "DİQQƏT: qiymət mənbəyi spot qızıl deyil, ~17$ fərq var. Yalnız məsafələri işlət.\n"
    msg += "Qiymət brokerdən fərqli ola bilər, SL/TP-ni brokerin qiymətinə görə hesabla.\nMaliyyə məsləhəti deyil."
    send(msg)
    print(msg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
        if manual or pd.Timestamp.now(tz="UTC").minute < 5:
            send("XETA: " + repr(e)[:400])
        raise
