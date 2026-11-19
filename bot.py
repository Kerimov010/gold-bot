import os
import requests
import pandas as pd

TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT = os.environ["TG_CHAT_ID"]

SL, TP1, TP2 = 9.0, 7.0, 15.0
TOUCH = 2.0
CLUSTER = 4.0
COOLDOWN_CANDLES = 6
FRESH_MINUTES = 4.5
NO_TRADE_UTC = []


def get_m5():
    last_err = None
    for host in ("query1", "query2"):
        try:
            r = requests.get(
                f"https://{host}.finance.yahoo.com/v8/finance/chart/XAUUSD=X",
                params={"interval": "5m", "range": "5d"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=20,
            )
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"Yahoo olmadi: {last_err}")
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(res["timestamp"], unit="s"),
        "open": q["open"], "high": q["high"],
        "low": q["low"], "close": q["close"],
    })
    df = df.dropna().set_index("datetime").sort_index()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    if df.index[-1] + pd.Timedelta(minutes=5) > now:
        df = df.iloc[:-1]
    return df, now


def get_bias(df):
    m15 = df["close"].resample("15min").last().dropna()
    e20 = m15.ewm(span=20).mean().iloc[-1]
    e50 = m15.ewm(span=50).mean().iloc[-1]
    p = m15.iloc[-1]
    if e20 < e50 and p < e20:
        return "SELL"
    if e20 > e50 and p > e20:
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


def check(df):
    bias = get_bias(df)
    c = df.iloc[-1]
    body = max(abs(c.close - c.open), 0.01)
    up = c.high - max(c.open, c.close)
    dn = min(c.open, c.close) - c.low

    if bias == "SELL" and c.close < c.open and up >= body:
        for z in cluster(pivots(df, "high")):
            if c.high >= z - TOUCH and c.high <= z + 6 and c.close < z:
                return "SELL", c.close, z
    if bias == "BUY" and c.close > c.open and dn >= body:
        for z in cluster(pivots(df, "low")):
            if c.low <= z + TOUCH and c.low >= z - 6 and c.close > z:
                return "BUY", c.close, z
    return None


def send(text):
    requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        json={"chat_id": TG_CHAT, "text": text}, timeout=20,
    )


def main():
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        send("Bot işləyir. Siqnal yalnız şərt uyğun gələndə gələcək.")

    df, now = get_m5()
    t = df.index[-1]
    closed_ago = (now - (t + pd.Timedelta(minutes=5))).total_seconds() / 60
    print(f"Son sam: {t}, baglanib: {closed_ago:.1f} deq evvel")

    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        send(f"Qiymət məlumatı gəlir. Son şam: {t} UTC, son qiymət: {df.iloc[-1].close:.2f}")

    if closed_ago > FRESH_MINUTES:
        print("Sam kohnedir, kecildi.")
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

    side, entry, zone = sig
    d = -1 if side == "SELL" else 1
    msg = (
        f"GOLD M5 | {side}\n"
        f"Zona: {zone:.2f}\n"
        f"Giriş: {entry:.2f}\n"
        f"SL: {entry - d * SL:.2f}\n"
        f"TP1: {entry + d * TP1:.2f} (yarısını bağla, SL-i girişə çək)\n"
        f"TP2: {entry + d * TP2:.2f}\n"
        f"Riskinizi düzgün idarə edin. Maliyyə məsləhəti deyil."
    )
    send(msg)
    print(msg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
            send("XETA: " + repr(e)[:400])
        raise
