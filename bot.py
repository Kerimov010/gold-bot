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
BUY_TREND_M5 = True   # YENİ: BUY üçün M5 trendi də yuxarı olmalıdır
BUY_MAX_DROP = 20.0   # YENİ: son 3 saatın zirvəsindən bu qədərdən çox düşübsə BUY yoxdur
DROP_LOOKBACK = 36
PLAN_EXPIRE_CANDLES = 24   # YENİ: plan 24 şam (2 saat) qüvvədədir
PLAN_MIN_GAP = 2.0         # qiymət EMA20-dən ən azı bu qədər uzaq olmalıdır
PLAN_MAX_DIST = 12.0       # ...amma bundan çox uzaq olmamalıdır
PLAN_ALERT_WINDOW = 3      # giriş toxunuşu bu qədər şamdan köhnədirsə bildiriş verilmir
PLAN_COOLDOWN_MIN = 60
PLAN_MIN_SCORE = 2         # YENİ: bütün timeframe-lərin çəkili səsi (BUY ≥ +2, SELL ≤ -2)
REPORT_EVERY_MIN = 240     # YENİ: bazar analizi hesabatı 4 saatdan bir
CONFIRM_BUY = True  # YENİ: BUY siqnalı bir şam gec, təsdiqdən sonra göndərilir
MAX_PRICE_DRIFT = 1.5  # YENİ: canlı qiymət girişdən bu qədər uzaqlaşıbsa siqnal göndərilmir

# YENİ: Asiya seansı bağlıdır (gecə yarısını keçən aralıq iki hissəyə bölünüb)
NO_TRADE_UTC = ["22:00-23:59", "00:00-06:59"]

NEWS_EVENTS = [
    ("2026-12-04 13:30", "NFP"),
    ("2026-12-09 19:00", "FOMC"),
    ("2026-12-10 13:30", "CPI"),
    ("2026-12-15 13:30", "PPI"),
    ("2027-01-08 13:30", "NFP"),
    ("2027-01-13 13:30", "CPI"),
    ("2027-01-14 13:30", "PPI"),
    ("2027-01-27 19:00", "FOMC"),
    ("2027-02-05 13:30", "NFP"),
    ("2027-02-11 13:30", "CPI"),
    ("2027-03-10 13:30", "CPI"),
    ("2027-03-17 18:00", "FOMC"),
]

NEWS_BEFORE, NEWS_AFTER = 30, 60
STATE_FILE = os.environ.get("STATE_FILE", "state.json")
MAX_HOLD = 72


def from_twelve():
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol": "XAU/USD",
            "interval": "5min",
            "outputsize": 5000,
            "timezone": "UTC",
            "apikey": TD_KEY,
        },
        timeout=20,
    )
    j = r.json()
    if "values" not in j:
        raise RuntimeError(str(j.get("message", j))[:200])

    df = pd.DataFrame(j["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])

    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)

    return (
        df[["datetime", "open", "high", "low", "close"]]
        .set_index("datetime")
        .sort_index()
    )


def get_m5():
    if not TD_KEY:
        raise RuntimeError("TD_API_KEY yoxdur")

    df = from_twelve()

    if df.empty:
        raise RuntimeError("XAU/USD məlumatı boşdur")

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)

    if df.index[-1] + pd.Timedelta(minutes=5) > now:
        df = df.iloc[:-1]

    if df.empty:
        raise RuntimeError("Bağlanmış M5 şamı yoxdur")

    return df, now, "Twelve Data XAU/USD"


# YENİ: canlı qiymət (https://twelvedata.com/docs/api, /price endpoint)
def live_price():
    r = requests.get(
        "https://api.twelvedata.com/price",
        params={"symbol": "XAU/USD", "apikey": TD_KEY},
        timeout=10,
    )
    j = r.json()
    if "price" not in j:
        raise RuntimeError(str(j.get("message", j))[:200])
    return float(j["price"])


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

    if (
        len(bars)
        and bars.index[-1] == bucket_start
        and last_start + pd.Timedelta(minutes=5) < bucket_end
    ):
        bars = bars.iloc[:-1]

    return bars


def bias_closed(df, minutes):
    b = closed_bars(df, minutes)["close"]

    if len(b) < 55:
        return "NEUTRAL"

    gap = (
        b.ewm(span=20).mean().iloc[-1]
        - b.ewm(span=50).mean().iloc[-1]
    )

    threshold = 1.0 if minutes <= 15 else 2.0

    if gap < -threshold:
        return "SELL"
    if gap > threshold:
        return "BUY"

    return "NEUTRAL"


def pivots(df, kind, n=5):
    m15 = df.resample("15min").agg({
        "high": "max",
        "low": "min",
    }).dropna()

    s = m15["high"] if kind == "high" else m15["low"]
    roll = s.rolling(2 * n + 1, center=True)
    ref = roll.max() if kind == "high" else roll.min()

    return list(s[s == ref])


def cluster(levels):
    out, grp = [], []

    for level in sorted(levels):
        if grp and level - grp[0] > CLUSTER:
            out.append(sum(grp) / len(grp))
            grp = []

        grp.append(level)

    if grp:
        out.append(sum(grp) / len(grp))

    return out


# YENİ: BUY üçün əlavə güvənlik filtri
def buy_ok(df):
    if not BUY_TREND_M5:
        return True

    close = df["close"]
    e20 = close.ewm(span=20).mean()
    e50 = close.ewm(span=50).mean()

    # M5 EMA20 EMA50-dən yuxarı və yüksələn olmalıdır
    if not (e20.iloc[-1] > e50.iloc[-1] and e20.iloc[-1] > e20.iloc[-4]):
        return False

    # qiymət EMA50-nin altında olmamalıdır
    if close.iloc[-1] < e50.iloc[-1]:
        return False

    # sərt düşüşdən sonrakı sıçrayışa girmə
    recent_high = df["high"].iloc[-DROP_LOOKBACK:].max()
    if recent_high - close.iloc[-1] > BUY_MAX_DROP:
        return False

    return True


# YENİ: trend davamı (EMA20-yə geri çəkilmə) - yalnız BUY, ayrı etiketlə
def pullback_buy(df):
    if bias_closed(df, 60) != "BUY":
        return None

    ema = df["close"].ewm(span=20).mean()
    e = float(ema.iloc[-1])
    c = df.iloc[-1]
    prev = df.iloc[-2]

    if (
        prev.close < prev.open
        and c.low <= e + 0.5
        and c.close > e
        and c.close - e < 3.0
        and c.close > c.open
    ):
        return ("BUY", c.close, e, "EMA pullback BUY")

    return None


def check_raw(df):
    bias = bias_closed(df, 15)

    if bias not in ("BUY", "SELL") or len(df) < 5:
        return None

    if bias == "BUY" and not buy_ok(df):
        return None

    if bias == "BUY":
        pb = pullback_buy(df)
        if pb:
            return pb

    last3 = df.iloc[-3:]
    c = df.iloc[-1]
    prev = df.iloc[-2]

    if bias == "BUY" and not (
        last3["close"] > last3["open"]
    ).all():
        return None

    if bias == "SELL" and not (
        last3["close"] < last3["open"]
    ).all():
        return None

    sup = cluster(pivots(df, "low"))
    res = cluster(pivots(df, "high"))

    recent = df.iloc[-BREAK_LOOKBACK - 2:-2]

    if bias == "SELL":
        for s in sup:
            broke = (
                (recent["close"].shift(1) >= s)
                & (recent["close"] < s - BREAK_BUFFER)
            ).any()

            if (
                broke
                and prev.close < s - BREAK_BUFFER
                and c.high >= s - TOUCH
                and c.high <= s + TOUCH
                and c.close < s
                and c.close < c.open
            ):
                return (
                    "SELL",
                    c.close,
                    s,
                    "Qırılma + retest + M5 təsdiqi",
                )

    if bias == "BUY":
        for r in res:
            broke = (
                (recent["close"].shift(1) <= r)
                & (recent["close"] > r + BREAK_BUFFER)
            ).any()

            if (
                broke
                and prev.close > r + BREAK_BUFFER
                and c.low <= r + TOUCH
                and c.low >= r - TOUCH
                and c.close > r
                and c.close > c.open
            ):
                return (
                    "BUY",
                    c.close,
                    r,
                    "Qırılma + retest + M5 təsdiqi",
                )

    return None


def check(df):
    # SELL: dəyişməyib. BUY: CONFIRM_BUY açıqdırsa, siqnal şamından sonrakı
    # şam onun high-ını yuxarı bağlayıb yaşıl olarsa göndərilir.
    if CONFIRM_BUY and len(df) > 6:
        prev_sig = check_raw(df.iloc[:-1])

        if prev_sig and prev_sig[0] == "BUY":
            p = df.iloc[-2]
            c = df.iloc[-1]

            if c.close > p.high and c.close > c.open:
                return (
                    "BUY",
                    c.close,
                    prev_sig[2],
                    prev_sig[3] + " (təsdiqli)",
                )

    sig = check_raw(df)

    if sig and sig[0] == "BUY" and CONFIRM_BUY:
        return None

    return sig


def levels_for(side, entry):
    d = -1 if side == "SELL" else 1
    return (
        entry - d * SL,
        entry + d * TP1,
        entry + d * TP2,
    )


def atr_ratio(df, period=14):
    b = closed_bars(df, 15)

    if len(b) < period * 3:
        return -1.0

    pc = b["close"].shift(1)

    tr = pd.concat(
        [
            b["high"] - b["low"],
            (b["high"] - pc).abs(),
            (b["low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period).mean().dropna()

    return float(atr.iloc[-1] / atr.median())


def weekend(ts):
    wd = ts.weekday()
    return (
        wd == 5
        or (wd == 4 and ts.hour >= 21)
        or (wd == 6 and ts.hour < 22)
    )


def in_blackout(ts):
    for rng in NO_TRADE_UTC:
        a, b = rng.split("-")
        if a <= ts.strftime("%H:%M") <= b:
            return True

    return False


def news_block(ts):
    for event_time, name in NEWS_EVENTS:
        et = pd.Timestamp(event_time)

        if (
            et - pd.Timedelta(minutes=NEWS_BEFORE)
            <= ts
            <= et + pd.Timedelta(minutes=NEWS_AFTER)
        ):
            return et, name

    return None


def _retry(fn, tries=3):
    last = None

    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            print(f"Telegram xətası ({k + 1}/{tries}):", e)
            time.sleep(2 * (k + 1))

    raise last


def send(text):
    def go():
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text},
            timeout=20,
        )

        if not r.ok:
            raise RuntimeError("sendMessage: " + r.text[:200])

    _retry(go)


def send_photo(buf, caption):
    def go():
        buf.seek(0)

        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
            data={
                "chat_id": TG_CHAT,
                "caption": caption[:1000],
            },
            files={
                "photo": ("signal.png", buf, "image/png"),
            },
            timeout=30,
        )

        if not r.ok:
            raise RuntimeError("sendPhoto: " + r.text[:200])

    _retry(go)


def _base_chart(df, count):
    d = df.iloc[-count:]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    w = 0.6

    for i, (_, r) in enumerate(d.iterrows()):
        col = "#26a69a" if r.close >= r.open else "#ef5350"

        ax.plot(
            [i, i],
            [r.low, r.high],
            color=col,
            lw=1,
        )

        ax.add_patch(
            Rectangle(
                (i - w / 2, min(r.open, r.close)),
                w,
                max(abs(r.close - r.open), 0.05),
                color=col,
            )
        )

    return d, fig, ax


def _finish(d, fig, ax, n, title, ys):
    ax.set_xlim(-1, n + 16)

    ax.set_ylim(
        min(list(ys) + [d.low.min()]) - 3,
        max(list(ys) + [d.high.max()]) + 3,
    )

    ax.set_title(title, color="white", fontsize=12)
    ax.tick_params(colors="#8b949e")

    xt = list(range(0, n, 15))
    ax.set_xticks(xt)
    ax.set_xticklabels(
        [d.index[i].strftime("%H:%M") for i in xt],
        color="#8b949e",
    )

    ax.set_xlabel("UTC", color="#8b949e")

    for sp in ax.spines.values():
        sp.set_color("#30363d")

    buf = io.BytesIO()

    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)
    buf.seek(0)

    return buf


def make_chart(df, side, entry, zone, label):
    d, fig, ax = _base_chart(df, 90)

    sl, tp1, tp2 = levels_for(side, entry)
    n = len(d)

    lines = [
        (zone, "#f0b90b", "--", f"Zona {zone:.2f}"),
        (entry, "#58a6ff", "-", f"Giriş {entry:.2f}"),
        (sl, "#ff5252", "-", f"SL {sl:.2f}"),
        (tp1, "#00e676", "-", f"TP1 {tp1:.2f}"),
        (tp2, "#00e676", "--", f"TP2 {tp2:.2f}"),
    ]

    for y, col, ls, txt in lines:
        ax.hlines(
            y,
            -1,
            n,
            colors=col,
            linestyles=ls,
            linewidth=1.2,
        )

        ax.text(
            n + 1,
            y,
            txt,
            color=col,
            va="center",
            fontsize=9,
        )

    return _finish(
        d,
        fig,
        ax,
        n,
        f"XAUUSD M5 | {side} | {label}",
        [sl, tp1, tp2],
    )


def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            s = json.load(f)
    except FileNotFoundError:
        s = {}

    s.setdefault("next_id", 1)
    s.setdefault("open", [])
    s.setdefault("closed", [])
    s.setdefault("pending", [])
    s.setdefault("next_plan", 1)

    return s


def save_state(s):
    s["closed"] = s["closed"][-500:]

    tmp = STATE_FILE + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)

    os.replace(tmp, STATE_FILE)


def resolve(df, sig):
    ts = pd.Timestamp(sig["ts"])

    if ts not in df.index:
        if df.index[0] > ts:
            return True, False, 0.0, "NAMƏLUM"

        return False, False, 0.0, None

    i = df.index.get_loc(ts)
    n = len(df)

    if i + 1 >= n:
        return False, False, 0.0, None

    o, h, l, c = (
        df[x].values
        for x in ["open", "high", "low", "close"]
    )

    d = -1 if sig["side"] == "SELL" else 1

    entry = float(sig.get("entry", o[i + 1]))

    sl = entry - d * SL
    tp1 = entry + d * TP1
    tp2 = entry + d * TP2

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
                return True, False, -SL, "SL"

            if f1:
                tp1_hit = True

                if f2:
                    return (
                        True,
                        True,
                        0.5 * TP1 + 0.5 * TP2,
                        "TP2",
                    )
        else:
            if adverse:
                return True, True, 0.5 * TP1, "BE"

            if f2:
                return (
                    True,
                    True,
                    0.5 * TP1 + 0.5 * TP2,
                    "TP2",
                )

    if i + MAX_HOLD <= n - 1:
        open_pnl = d * (c[i + MAX_HOLD] - entry)
        pnl = (
            0.5 * TP1 + 0.5 * open_pnl
            if tp1_hit
            else open_pnl
        )

        return True, tp1_hit, pnl, "TIME"

    return False, tp1_hit, 0.0, None


def process_open(df, state):
    still = []
    today = df.index[-1].strftime("%Y-%m-%d")

    for s in state["open"]:
        done, tp1, pnl, out = resolve(df, s)

        head = f"Siqnal #{s['id']} ({s['side']}, {s['label']})"

        if not done:
            if tp1 and not s.get("tp1_notified"):
                send(
                    f"🎯 {head}: TP1 vuruldu. "
                    "Yarısını bağla, SL-i girişə çək."
                )

                s["tp1_notified"] = True

            still.append(s)
            continue

        if out == "TP2":
            msg = f"✅ {head}: TP2 vuruldu. Nəticə ≈ {pnl:+.1f}$"

        elif out == "BE":
            msg = (
                f"🟡 {head}: TP1 vuruldu, sonra girişə qayıtdı "
                f"(breakeven). Nəticə ≈ {pnl:+.1f}$"
            )

        elif out == "SL":
            msg = f"❌ {head}: SL vuruldu. Nəticə ≈ {pnl:+.1f}$"

        else:
            # TIME və NAMƏLUM Telegram-a göndərilmir.
            msg = None

        if msg:
            send(msg + "\n(0.01 lotda, spreadsiz)")

        state["closed"].append({
            "id": s["id"],
            "ts": s["ts"],
            "side": s["side"],
            "label": s["label"],
            "outcome": out,
            "tp1": bool(tp1),
            "vol": bool(s.get("vol", False)),
            "gross": float(pnl),
            "closed_at": today,
        })

    state["open"] = still


# ---------------- YENİ: TAM BAZAR ANALİZİ ----------------
TF_NAMES = [(5, "M5", 1), (15, "M15", 2), (60, "H1", 3), (240, "H4", 4), (1440, "D1", 3)]
TF_SPANS = {5: (20, 50), 15: (20, 50), 60: (20, 50), 240: (20, 50), 1440: (5, 10)}
TF_THR = {5: 0.5, 15: 1.0, 60: 2.0, 240: 4.0, 1440: 8.0}
LEVEL_TFS = [(15, 5, 1), (60, 4, 2), (240, 3, 3), (1440, 2, 4)]
ARROW = {"BUY": "↑", "SELL": "↓", "NEUTRAL": "↔"}


def tf_trend(df, minutes):
    b = closed_bars(df, minutes)
    fast, slow = TF_SPANS[minutes]

    if len(b) < slow + 5:
        return "NEUTRAL"

    close = b["close"]
    gap = (
        close.ewm(span=fast).mean().iloc[-1]
        - close.ewm(span=slow).mean().iloc[-1]
    )

    if gap > TF_THR[minutes]:
        return "BUY"
    if gap < -TF_THR[minutes]:
        return "SELL"
    return "NEUTRAL"


def collect_points(df):
    pts = []

    for minutes, n, w in LEVEL_TFS:
        b = closed_bars(df, minutes)

        if len(b) < 2 * n + 3:
            continue

        for kind in ("high", "low"):
            ser = b[kind]
            roll = ser.rolling(2 * n + 1, center=True)
            ref = roll.max() if kind == "high" else roll.min()

            for v in ser[ser == ref].dropna().values:
                pts.append((float(v), w, kind, minutes))

    return pts


def day_levels(df):
    out = []
    today = df.index[-1].floor("1D")
    cnt = df.groupby(df.index.floor("1D")).size()
    prev_days = [d for d in cnt.index if d < today and cnt[d] >= 100]

    if prev_days:
        d = df[df.index.floor("1D") == prev_days[-1]]
        out.append((float(d["high"].max()), 3, "high", 1440))
        out.append((float(d["low"].min()), 3, "low", 1440))

    asia = df[(df.index >= today) & (df.index < today + pd.Timedelta(hours=7))]

    if len(asia) >= 12:
        out.append((float(asia["high"].max()), 2, "high", 60))
        out.append((float(asia["low"].min()), 2, "low", 60))

    return out


def cluster_levels(pts, gap=CLUSTER):
    pts = sorted(pts, key=lambda x: x[0])
    out, grp = [], []

    def flush(g):
        if g:
            out.append({
                "price": sum(x[0] for x in g) / len(g),
                "weight": sum(x[1] for x in g),
                "n": len(g),
            })

    for p in pts:
        if grp and p[0] - grp[0][0] > gap:
            flush(grp)
            grp = []
        grp.append(p)

    flush(grp)
    return out


def liquidity_pools(pts, gap=2.5):
    # bərabər zirvələr / bərabər dibləri: stop-lossların yığıldığı yerlər
    res = {"high": [], "low": []}

    for kind in ("high", "low"):
        ps = sorted(
            (p for p in pts if p[2] == kind and p[3] in (15, 60)),
            key=lambda x: x[0],
        )
        grp = []

        for p in ps + [None]:
            if p is not None and (not grp or p[0] - grp[0][0] <= gap):
                grp.append(p)
                continue

            if len(grp) >= 2:
                res[kind].append(sum(x[0] for x in grp) / len(grp))

            grp = [p] if p is not None else []

    return res


def detect_sweep(df, levels, pools):
    cands = [l["price"] for l in levels if l["weight"] >= 3]
    cands += pools["high"] + pools["low"]

    for k in range(1, 7):
        c = df.iloc[-k]
        for L in cands:
            if c.high > L + 0.5 and c.close < L:
                return f"{df.index[-k].strftime('%H:%M')} UTC: {L:.1f} zirvə likvidliyi süpürüldü (aşağı işarə)"
            if c.low < L - 0.5 and c.close > L:
                return f"{df.index[-k].strftime('%H:%M')} UTC: {L:.1f} dib likvidliyi süpürüldü (yuxarı işarə)"

    return None


def analyze(full):
    price = float(full["close"].iloc[-1])
    trends, score, total = {}, 0, 0

    for minutes, name, w in TF_NAMES:
        tr = tf_trend(full, minutes)
        trends[name] = tr
        total += w
        score += w if tr == "BUY" else (-w if tr == "SELL" else 0)

    pts = collect_points(full) + day_levels(full)
    levels = cluster_levels(pts)
    pools = liquidity_pools(pts)

    sup = sorted(
        [l for l in levels if l["price"] < price - 0.5],
        key=lambda l: -l["price"],
    )
    res = sorted(
        [l for l in levels if l["price"] > price + 0.5],
        key=lambda l: l["price"],
    )

    direction = "BUY" if score >= 4 else ("SELL" if score <= -4 else "NEUTRAL")

    return {
        "price": price,
        "trends": trends,
        "score": score,
        "total": total,
        "direction": direction,
        "sup": sup,
        "res": res,
        "liq_above": min((p for p in pools["high"] if p > price), default=None),
        "liq_below": max((p for p in pools["low"] if p < price), default=None),
        "sweep": detect_sweep(full, levels, pools),
    }


def trend_line(ana):
    return " | ".join(f"{k} {ARROW[v]}" for k, v in ana["trends"].items())


def fmt_levels(lst, n=3):
    if not lst:
        return "yoxdur"
    return ", ".join(f"{l['price']:.1f}(×{l['weight']})" for l in lst[:n])


def ana_summary(ana):
    return (
        f"Trend: {trend_line(ana)}\n"
        f"Müqavimət: {fmt_levels(ana['res'], 2)}\n"
        f"Dəstək: {fmt_levels(ana['sup'], 2)}"
    )


def make_report(ana, t):
    la = f"{ana['liq_above']:.1f}" if ana["liq_above"] else "yoxdur"
    lb = f"{ana['liq_below']:.1f}" if ana["liq_below"] else "yoxdur"

    txt = (
        f"📊 BAZAR ANALİZİ | GOLD | {t.strftime('%H:%M')} UTC\n"
        f"Qiymət: {ana['price']:.2f}\n"
        f"Trend: {trend_line(ana)}\n"
        f"Ümumi istiqamət: {ana['direction']} "
        f"({ana['score']:+d}/{ana['total']})\n\n"
        f"Müqavimət: {fmt_levels(ana['res'])}\n"
        f"Dəstək: {fmt_levels(ana['sup'])}\n"
        f"Likvidlik: yuxarıda {la}, aşağıda {lb}\n"
    )

    if ana["sweep"]:
        txt += f"Son süpürmə: {ana['sweep']}\n"

    txt += "(×N: səviyyənin çəkisi, böyük timeframe-lər daha ağırdır)\n"
    txt += "Maliyyə məsləhəti deyil."
    return txt


def maybe_report(state, t, ana):
    lr = state.get("last_report")

    if lr and (t - pd.Timestamp(lr)) < pd.Timedelta(minutes=REPORT_EVERY_MIN):
        return

    send(make_report(ana, t))
    state["last_report"] = str(t)


# ---------------- YENİ: İKİ MƏRHƏLƏLİ PLAN ----------------
def sell_ok(df):
    if not BUY_TREND_M5:
        return True

    close = df["close"]
    e20 = close.ewm(span=20).mean()
    e50 = close.ewm(span=50).mean()

    if not (e20.iloc[-1] < e50.iloc[-1] and e20.iloc[-1] < e20.iloc[-4]):
        return False

    if close.iloc[-1] > e50.iloc[-1]:
        return False

    recent_low = df["low"].iloc[-DROP_LOOKBACK:].min()
    if close.iloc[-1] - recent_low > BUY_MAX_DROP:
        return False

    return True


def make_plan(df, ana):
    close = df["close"]
    e20 = float(close.ewm(span=20).mean().iloc[-1])
    e50 = float(close.ewm(span=50).mean().iloc[-1])
    c = float(close.iloc[-1])

    b15 = bias_closed(df, 15)
    b60 = bias_closed(df, 60)

    if b15 == "BUY" and b60 == "BUY" and buy_ok(df):
        if PLAN_MIN_GAP <= c - e20 <= PLAN_MAX_DIST and ana["score"] >= PLAN_MIN_SCORE:
            r = next((l for l in ana["res"] if l["weight"] >= 3 and l["price"] > e20), None)
            if r is None or r["price"] - e20 >= TP1 + 1.0:
                return {
                    "side": "BUY",
                    "entry": round(e20, 2),
                    "inval": round(e50 - 1.0, 2),
                }

    if b15 == "SELL" and b60 == "SELL" and sell_ok(df):
        if PLAN_MIN_GAP <= e20 - c <= PLAN_MAX_DIST and ana["score"] <= -PLAN_MIN_SCORE:
            sp = next((l for l in ana["sup"] if l["weight"] >= 3 and l["price"] < e20), None)
            if sp is None or e20 - sp["price"] >= TP1 + 1.0:
                return {
                    "side": "SELL",
                    "entry": round(e20, 2),
                    "inval": round(e50 + 1.0, 2),
                }

    return None


def maybe_create_plan(df, state, t, ana):
    if state["pending"] or state["open"]:
        return

    lp = state.get("last_plan_ts")
    if lp and (t - pd.Timestamp(lp)) < pd.Timedelta(minutes=PLAN_COOLDOWN_MIN):
        return

    plan = make_plan(df, ana)
    if not plan:
        return

    side, entry, inval = plan["side"], plan["entry"], plan["inval"]
    sl, tp1, tp2 = levels_for(side, entry)
    pid = state["next_plan"]
    c = float(df["close"].iloc[-1])

    text = (
        f"📋 HAZIRLIQ #P{pid} | GOLD M5 | {side}\n"
        f"Plan: qiymət {entry:.2f} səviyyəsinə (EMA20) gələndə giriş.\n"
        f"Hələ girmə, gözlə.\n\n"
        f"Hazırkı qiymət: {c:.2f}\n"
        f"Giriş: {entry:.2f}\n"
        f"SL: {sl:.2f}\n"
        f"TP1: {tp1:.2f} (yarısını bağla, SL-i girişə çək)\n"
        f"TP2: {tp2:.2f}\n"
        f"Ləğv səviyyəsi: {inval:.2f} "
        f"(M5 şam bunu keçib bağlanarsa plan ləğv olur)\n"
        f"{ana_summary(ana)}\n"
        f"Plan {PLAN_EXPIRE_CANDLES * 5 // 60} saat qüvvədədir.\n"
        f"Qiymət girişə çatanda ayrıca bildiriş göndərəcəm.\n"
        f"Maliyyə məsləhəti deyil."
    )

    try:
        send_photo(make_chart(df, side, entry, entry, "HAZIRLIQ"), text)
    except Exception as e:
        print("Şəkil göndərilmədi:", e)
        send(text)

    state["pending"].append({
        "pid": pid,
        "ts": str(t),
        "side": side,
        "entry": float(entry),
        "inval": float(inval),
    })
    state["next_plan"] = pid + 1
    state["last_plan_ts"] = str(t)
    print(f"Plan yaradıldı: {side} @ {entry:.2f}")


def process_pending(df, state):
    still = []

    for p in state["pending"]:
        pts = pd.Timestamp(p["ts"])
        side = p["side"]
        entry = p["entry"]
        inval = p["inval"]
        tag = f"#P{p['pid']} ({side})"
        after = df[df.index > pts]

        # 1) ləğv: M5 şam ləğv səviyyəsini keçib bağlanıb
        if len(after):
            if side == "BUY":
                broken = (after["close"] < inval).any()
            else:
                broken = (after["close"] > inval).any()

            if broken:
                send(f"❌ Plan {tag} ləğv oldu: trend pozuldu. Girmə.")
                continue

        # 2) toxunuş
        if len(after):
            if side == "BUY":
                touched = after[after["low"] <= entry]
            else:
                touched = after[after["high"] >= entry]
        else:
            touched = after

        if len(touched):
            tt = touched.index[0]
            age = int((df.index > tt).sum())

            if age > PLAN_ALERT_WINDOW:
                send(
                    f"⌛ Plan {tag}: qiymət girişə çatmışdı, amma bildiriş "
                    f"gecikdi. Girmə."
                )
                continue

            b15 = bias_closed(df, 15)
            ok = (
                (side == "BUY" and b15 == "BUY" and buy_ok(df))
                or (side == "SELL" and b15 == "SELL" and sell_ok(df))
            )

            if not ok:
                send(
                    f"⚠️ Plan {tag}: qiymət girişə çatdı, amma şərtlər "
                    f"artıq uyğun deyil. Girmə."
                )
                continue

            try:
                live = live_price()
            except Exception as e:
                print("Canlı qiymət alınmadı:", e)
                live = float(df["close"].iloc[-1])

            sl, tp1, tp2 = levels_for(side, entry)
            sid = state["next_id"]

            msg = (
                f"📍 QİYMƏT GİRİŞƏ ÇATDI | GOLD M5 | {side} | #{sid}\n"
                f"(plan #P{p['pid']})\n\n"
                f"Giriş: {entry:.2f}\n"
                f"SL: {sl:.2f}\n"
                f"TP1: {tp1:.2f} (yarısını bağla, SL-i girişə çək)\n"
                f"TP2: {tp2:.2f}\n"
                f"Hazırkı qiymət: {live:.2f}\n\n"
                f"Trend şərtləri hələ də uyğundur. "
                f"Uyğun görürsənsə gir, yoxsa keç.\n"
                f"Maliyyə məsləhəti deyil."
            )

            if abs(live - entry) > 3.0:
                msg += "\n⚠️ Qiymət giriş səviyyəsindən uzaqlaşıb."

            vol = atr_ratio(df) > 2.0
            if vol:
                msg += "\n⚠️ Volatillik yüksəkdir."

            send(msg)

            state["open"].append({
                "id": sid,
                "ts": str(tt),
                "side": side,
                "label": f"PLAN {side}",
                "zone": float(entry),
                "entry": float(entry),
                "tp1_notified": False,
                "vol": bool(vol),
            })
            state["next_id"] = sid + 1
            print(f"Plan işə düşdü: {tag}")
            continue

        # 3) vaxt bitib
        if len(after) > PLAN_EXPIRE_CANDLES:
            send(f"⌛ Plan {tag} vaxtı bitdi: qiymət girişə gəlmədi.")
            continue

        still.append(p)

    state["pending"] = still


def _run(state):
    full, now, src = get_m5()
    df = full.iloc[-1500:]
    t = df.index[-1]

    closed_ago = (
        now - (t + pd.Timedelta(minutes=5))
    ).total_seconds() / 60

    print(f"Son şam: {t}, mənbə: {src}")

    process_open(df, state)

    if weekend(t):
        print("Bazar bağlıdır.")
        return

    if closed_ago > FRESH_MINUTES:
        print("Şam köhnədir.")
        return

    if news_block(t + pd.Timedelta(minutes=5)):
        print("Xəbər vaxtıdır.")
        return

    if in_blackout(t):
        print("Qeyri-işlək seans (Asiya).")
        return

    try:
        ana = analyze(full)
    except Exception as e:
        print("Analiz xətası:", e)
        ana = None

    process_pending(df, state)

    if ana:
        maybe_create_plan(df, state, t, ana)
        maybe_report(state, t, ana)

    for k in range(1, COOLDOWN_CANDLES + 1):
        if check(df.iloc[:-k]):
            print("Cooldown, siqnal ötürüldü.")
            return

    sig = check(df)

    if not sig:
        print("Siqnal yoxdur.")
        return

    if any(
        x["ts"] == str(t)
        for x in state["open"] + state["closed"]
    ):
        print("Bu siqnal artıq göndərilib.")
        return

    side, entry, zone, label = sig

    # YENİ: canlı qiymət yoxlaması
    try:
        live = live_price()
    except Exception as e:
        print("Canlı qiymət alınmadı:", e)
        return

    if abs(live - entry) > MAX_PRICE_DRIFT:
        print(f"Qiymət qaçıb: {live:.2f} vs {entry:.2f}")
        return

    entry = live

    sl, tp1, tp2 = levels_for(side, entry)
    sid = state["next_id"]

    caption = (
        f"🚨 GOLD M5 | {side} | #{sid}\n"
        f"Setup: {label}\n\n"
        f"Giriş: {entry:.2f}\n"
        f"SL: {sl:.2f}\n"
        f"TP1: {tp1:.2f} (yarısını bağla, SL-i girişə çək)\n"
        f"TP2: {tp2:.2f}\n"
        f"Zona: {zone:.2f}\n"
        f"Risk/mükafat: 1:{TP1 / SL:.1f} (TP1), "
        f"1:{TP2 / SL:.1f} (TP2)\n"
        f"Qiymət mənbəyi: {src}\n"
        f"Broker qiyməti fərqlənə bilər.\n"
        f"Yalnız qiymət {entry - MAX_PRICE_DRIFT:.2f}-"
        f"{entry + MAX_PRICE_DRIFT:.2f} aralığındadırsa gir.\n"
        f"Maliyyə məsləhəti deyil."
    )

    if atr_ratio(df) > 2.0:
        caption += "\n⚠️ Volatillik yüksəkdir."

    try:
        send_photo(
            make_chart(df, side, entry, zone, label),
            caption,
        )
    except Exception as e:
        print("Şəkil göndərilmədi:", e)
        send(caption)

    state["open"].append({
        "id": sid,
        "ts": str(t),
        "side": side,
        "label": label,
        "zone": float(zone),
        "entry": float(entry),
        "tp1_notified": False,
        "vol": atr_ratio(df) > 2.0,
    })

    state["next_id"] = sid + 1

    print(caption)


def loop_forever():
    # fasiləsiz rejim: server/kompüterdə işləyəndə hər M5 şamından sonra bir dəfə analiz
    while True:
        try:
            main()
        except Exception as e:
            print("Dövrə xətası:", repr(e)[:300])

        now = pd.Timestamp.now(tz="UTC")
        nxt = now.floor("5min") + pd.Timedelta(minutes=5, seconds=20)
        time.sleep(max(15, (nxt - now).total_seconds()))


def main():
    state = load_state()

    try:
        _run(state)
    finally:
        save_state(state)


if __name__ == "__main__":
    if os.environ.get("LOOP") == "1":
        loop_forever()
        raise SystemExit

    try:
        main()
        if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
            send("TEST — BOT ISLEYIR — REAL SIQNAL DEYIL")
    except Exception as e:
        manual = (
            os.environ.get("GITHUB_EVENT_NAME")
            == "workflow_dispatch"
        )

        if manual or pd.Timestamp.now(tz="UTC").minute < 5:
            try:
                send("XETA: " + repr(e)[:400])
            except Exception:
                pass

        raise
