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
OFFSET = 0.0   # broker qiyməti minus mənbə qiyməti (sonra təyin edəcəyik)


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


def from_yahoo():
    last = None
    for host in ("query1", "query2"):
        try:
            r = requests.get(
                f"https://{host}.finance.yahoo.com/v8/finance/chart/GC=F",
                params={"interval": "5m", "range": "5d"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=20,
            )
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame({
                "datetime": pd.to_datetime(res["timestamp"], unit="s"),
                "open": q["open"], "high": q["high"],
                "low": q["low"], "close": q["close"],
            })
            return df.dropna().set_index("datetime").sort_index()
        except Exception as e:
            last = e
    raise RuntimeError(str(last))


def get_m5():
    errs = []
    for name, fn in (("Kraken PAXG", from_kraken), ("Yahoo GC=F", from_yahoo)):
        try:
            df = fn()
            break
        except Exception as e:
            errs.append(f"{name}: {e}")
    else:
        raise RuntimeError(" | ".join(errs))
    df[["open", "high", "low", "close"]] += OFFSET
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    if df.index[-1] + pd.Timedelta(minutes=5) > now:
        df = df.iloc[:-1]
    return df, now, name


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


def weekend(ts):
    wd = ts.weekday()
    return wd == 5 or (wd == 4 and ts.hour >= 21) or (wd == 6 and ts.hour < 22)


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
    manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    if manual:
        send("Bot işləyir. Siqnal yalnız şərt uyğun gələndə gələcək.")

    df, now, src = get_m5()
    t = df.index[-1]
    closed_ago = (now - (t + pd.Timedelta(minutes=5))).total_seconds() / 60
    print(f"Son sam: {t}, baglanib: {closed_ago:.1f} deq evvel, menbe: {src}")

    if manual:
        send(f"Mənbə: {src}\nSon şam: {t} UTC\nSon qiymət: {df.iloc[-1].close:.2f}\n"
             f"Bunu brokerindəki qiymətlə müqayisə et və mənə yaz.")

    if weekend(t):
        print("Bazar bagli, kecildi.")
        return
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
        f"GOLD M5 | {side} ({src})\n"
        f"Zona: {zone:.2f}\n"
        f"Giriş: {entry:.2f}\n"
        f"SL: {entry - d * SL:.2f}\n"
        f"TP1: {entry + d * TP1:.2f} (yarısını bağla, SL-i girişə çək)\n"
        f"TP2: {entry + d * TP2:.2f}\n"
        f"Qiymət brokerdən fərqli ola bilər, SL/TP-ni brokerin qiymətinə görə hesabla.\n"
        f"Maliyyə məsləhəti deyil."
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
