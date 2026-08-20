#!/usr/bin/env python3
"""
MIMO SWEEP — Broad symbol x strategy edge hunt, self-contained.
Runs in MiMo Claw's 4-hour window. numpy + requests ONLY (no pandas dependency).

METHOD (honest, anti-overfit):
  For each (symbol, strategy-family): grid-search params on the FIRST 70% of history
  (train), then evaluate the chosen params ONLY on the LAST 30% (test / out-of-sample).
  Report only strategies with OOS Sharpe>0.8, OOS trades>=20, OOS PF>1.3.
  Train/test split is the single most defensible check; multiple-testing caveat noted.
  Costs: 0.1% round trip applied to every trade.

DATA: Yahoo Finance direct chart API (free, no key, 20+ yr daily history per symbol).
  Fallback to yfinance if importable. Cache every symbol to disk (cache dir) so a
  restart resumes without re-fetching.

USAGE:  python mimo_sweep.py                 # full run
        python mimo_sweep.py --symbols AAPL MSFT   # subset (short debugging run)
        python mimo_sweep.py --budget 200          # stop after ~200 mins, prune symbol list
"""

import os
import sys
import json
import time
import argparse
import requests
import numpy as np
from datetime import datetime, timedelta

CACHE_DIR = "mimo_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

COST = 0.001  # 0.1% round-trip per trade
START = "2000-01-01"
END   = datetime.utcnow().strftime("%Y-%m-%d")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# ---------------------------------------------------------------------------
# SYMBOL UNIVERSE (Yahoo tickers) — expanded: ~260 symbols
# ---------------------------------------------------------------------------
SYMBOLS = [
    # --- S&P 500 large caps (sector-diverse) ---
    "AAPL","MSFT","GOOGL","GOOG","AMZN","NVDA","AMD","META","TSLA","NFLX",
    "AVGO","ORCL","CRM","ADBE","CSCO","ACN","INTU","QCOM","TXN","IBM",
    "JPM","BAC","WFC","C","GS","MS","USB","PNC","COF","AXP",
    "XOM","CVX","COP","OXY","SLB","EOG","MPC","VLO","PSX","KMI",
    "JNJ","PFE","MRK","ABBV","UNH","LLY","TMO","ABT","DHR","BMY",
    "WMT","COST","HD","LOW","TGT","KR","DG","DLTR","MCD","SBUX",
    "V","MA","PYPL","SQ","AXP","FISV","GPN","PAYX","INTU","NOW",
    "PG","KO","PEP","PM","MO","CL","KMB","GIS","KHC","HSY",
    "DIS","CMCSA","T","VZ","TMUS","NKE","MCD","SBUX","ABNB","UBER",
    "INTC","MU","AMAT","LRCX","KLAC","NVDA","AMD","TXN","ADI","SWKS",
    # --- Mid/small-cap & growth ---
    "PLTR","SNOW","SHOP","SQ","RBLX","COIN","HOOD","DASH","AFRM","SOFI",
    "GME","AMC","BYND","PLUG","FCEL","RIVN","LCID","MARA","RIOT","SNAP",
    "NET","HUBS","ZM","DOCU","CRWD","PANW","FTNT","ZS","OKTA","MDB",
    # --- ETFs: broad/index ---
    "SPY","QQQ","IWM","DIA","VO","VB","VOO","VTI","VXUS","BND",
    # --- ETFs: sector ---
    "XLK","XLF","XLE","XLU","XLV","XLY","XLP","XLI","XLB","XLRE",
    "XLC","XBI","XLG","XLI","XLV","XLY","XLP","XLE","XLU","XLK",
    # --- ETFs: thematic/leveraged/inverse ---
    "TQQQ","SQQQ","SOXL","SOXX","SPXL","SPXS","UDOW","SDOW","TNA","TZA",
    "ARKK","ARKW","ARKG","ARKF","ARKX","BOTZ","IBB","XBI","XBI","IBB",
    # --- ETFs: fixed income ---
    "TLT","IEF","SHY","LQD","HYG","JNK","AGG","BND","MBB","TIP",
    # --- ETFs: international/EM ---
    "EEM","EFA","VWO","FXI","EWJ","EWG","EWU","ACWI","VEA","GXC",
    # --- ETFs: commodities ---
    "GLD","SLV","GDX","GDXJ","USO","UNG","UGA","DBA","DBC","CC",
    "COPX","URA","LIT","TAN","ICLN","PBW","PICK","REMX","XME","XLB",
    # --- Forex (Yahoo format) ---
    "EURUSD=X","GBPUSD=X","AUDUSD=X","NZDUSD=X","USDJPY=X","USDCAD=X","USDCHF=X",
    "USDSGD=X","USDHKD=X","USDSEK=X","USDNOK=X","USDDKK=X","EURJPY=X","EURGBP=X",
    "GBPJPY=X","AUDJPY=X","CHFJPY=X","CADJPY=X","NZDJPY=X","EURNZD=X","GBPAUD=X",
    "AUDNZD=X","NZDCAD=X","EURCHF=X","GBPCHF=X","AUDCHF=X","CADCHF=X","USDMXN=X",
    "USDZAR=X","USDTRY=X","USDBRL=X","USDRUB=X","USDKRW=X","USDTWD=X","USDINR=X",
    # --- Commodities/energy/metals ---
    "GC=F","SI=F","HG=F","PL=F","PA=F","CL=F","BZ=F","NG=F","HO=F","RB=F",
    "ZC=F","ZW=F","ZS=F","ZM=F","ZL=F","ZO=F","ZR=F","KC=F","SB=F","CC=F",
    "CT=F","OJ=F","LE=F","GF=F","HE=F","LH=F","W=F","RR=F","YI=F","MGC=F",
    # --- Crypto ---
    "BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD","DOGE-USD",
    "DOT-USD","LTC-USD","LINK-USD","MATIC-USD","AVAX-USD","UNI-USD","ATOM-USD",
    "XLM-USD","TRX-USD","ETC-USD","FIL-USD","NEAR-USD","SHIB-USD",
    # --- BRK special ---
    "BRK-B",
]

# ---------------------------------------------------------------------------
# DATA FETCH
# ---------------------------------------------------------------------------
def _cache_path(sym):
    safe = sym.replace("/", "_").replace("=", "_")
    return os.path.join(CACHE_DIR, f"{safe}.json")

def fetch_yahoo(sym, retries=3):
    """Get daily OHLC from Yahoo chart API. Returns list of dicts or []."""
    cp = _cache_path(sym)
    if os.path.exists(cp):
        try:
            with open(cp) as f:
                return json.load(f)
        except Exception:
            pass

    # Try yfinance first if available (robust)
    try:
        import yfinance as yf
        df = yf.download(sym, start=START, end=END, progress=False, auto_adjust=False)
        if df is not None and not df.empty:
            bars = []
            for idx, row in df.iterrows():
                dt = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
                try:
                    bars.append({"date": dt,
                                 "open": float(row["Open"]), "high": float(row["High"]),
                                 "low": float(row["Low"]), "close": float(row["Close"]),
                                 "volume": float(row.get("Volume") or 0)})
                except (TypeError, ValueError):
                    continue
            if len(bars) > 100:
                _write_cache(sym, bars)
                return bars
    except Exception:
        pass

    # Direct Yahoo chart API
    p1 = int(datetime.strptime(START, "%Y-%m-%d").timestamp())
    p2 = int(datetime.strptime(END, "%Y-%m-%d").timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    for attempt in range(retries):
        try:
            r = requests.get(url, params={"period1": p1, "period2": p2,
                                          "interval": "1d", "events": "history"},
                             headers=UA, timeout=30)
            if r.status_code != 200:
                time.sleep(2 + attempt * 2)
                continue
            data = r.json()
            res = data.get("chart", {}).get("result")
            if not res:
                return []
            ts  = res[0].get("timestamp", [])
            q   = res[0].get("indicators", {}).get("quote", [{}])[0]
            bars = []
            for i, t in enumerate(ts):
                try:
                    o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
                    if None in (o, h, l, c):
                        continue
                    bars.append({"date": datetime.utcfromtimestamp(t).strftime("%Y-%m-%d"),
                                 "open": float(o), "high": float(h), "low": float(l),
                                 "close": float(c), "volume": float(q.get("volume",[0]*len(ts))[i] or 0)})
                except (TypeError, IndexError, ValueError):
                    continue
            if len(bars) > 100:
                _write_cache(sym, bars)
            return bars
        except requests.exceptions.RequestException:
            time.sleep(2 + attempt * 2)
    return []

def _write_cache(sym, bars):
    try:
        with open(_cache_path(sym), "w") as f:
            json.dump(bars, f)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# INDICATORS (pure numpy)
# ---------------------------------------------------------------------------
def ema(a, p):
    out = np.empty(len(a)); out[0] = a[0]
    k = 2.0 / (p + 1)
    for i in range(1, len(a)):
        out[i] = a[i] * k + out[i-1] * (1 - k)
    return out

def atr(h, l, c, p=14):
    n = len(c); out = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    for i in range(p, n):
        out[i] = np.mean(tr[i-p+1:i+1])
    return out

def rsi(c, p=14):
    n = len(c); out = np.full(n, 50.0)
    if n < p+1: return out
    d = np.diff(c)
    for i in range(p, n):
        seg = d[i-p:i]
        up = seg[seg > 0].sum(); dn = -seg[seg < 0].sum()
        if dn > 0:
            out[i] = 100 - 100/(1 + up/dn)
        else:
            out[i] = 100
    return out

# ---------------------------------------------------------------------------
# STRATEGY FAMILIES  (return list of per-trade returns)
# ---------------------------------------------------------------------------
def strat_donchian(bars, lb, rr, use_short):
    c = np.array([b["close"] for b in bars], dtype=float)
    h = np.array([b["high"] for b in bars], dtype=float)
    l = np.array([b["low"] for b in bars], dtype=float)
    n = len(c)
    if n < lb + 5: return []
    a = atr(h, l, c)
    trades, pos, entry, stop = [], 0, 0.0, 0.0
    for i in range(lb+2, n):
        if pos == 0:
            up = h[i-lb:i].max(); dn = l[i-lb:i].min()
            if c[i] > up:
                pos, entry, stop = 1, c[i], c[i] - 2.0*a[i]
            elif c[i] < dn:
                pos, entry, stop = -1, c[i], c[i] + 2.0*a[i]
        elif pos == 1:
            if use_short and c[i] > entry + rr*abs(entry-stop):
                trades.append((c[i]/entry - 1)*1 - COST); pos = 0
            elif l[i] <= stop:
                trades.append((stop/entry - 1)*1 - COST); pos = 0
            else:
                stop = max(stop, c[i] - 2.0*a[i])
        else:
            if use_short and c[i] < entry - rr*abs(entry-stop):
                trades.append((c[i]/entry - 1)*-1 - COST); pos = 0
            elif h[i] >= stop:
                trades.append((stop/entry - 1)*-1 - COST); pos = 0
            else:
                stop = min(stop, c[i] + 2.0*a[i])
    if pos == 1: trades.append((c[n-1]/entry - 1)*1 - COST)
    if pos == -1: trades.append((c[n-1]/entry - 1)*-1 - COST)
    return trades

def strat_ema(bars, fast, slow):
    c = np.array([b["close"] for b in bars], dtype=float)
    n = len(c)
    if n < slow + 5: return []
    ef, es = ema(c, fast), ema(c, slow)
    h = np.array([b["high"] for b in bars], dtype=float)
    l = np.array([b["low"] for b in bars], dtype=float)
    a = atr(h, l, c)
    trades, pos, entry, stop = [], 0, 0.0, 0.0
    for i in range(slow+1, n):
        if pos == 0:
            if ef[i] > es[i] and ef[i-1] <= es[i-1]:
                pos, entry, stop = 1, c[i], c[i] - 2.0*a[i]
            elif ef[i] < es[i] and ef[i-1] >= es[i-1]:
                pos, entry, stop = -1, c[i], c[i] + 2.0*a[i]
        elif pos == 1:
            if l[i] <= stop:
                trades.append((stop/entry - 1) - COST); pos = 0
            else: stop = max(stop, c[i] - 2.0*a[i])
        else:
            if h[i] >= stop:
                trades.append((stop/entry - 1)*-1 - COST); pos = 0
            else: stop = min(stop, c[i] + 2.0*a[i])
    if pos == 1: trades.append((c[n-1]/entry - 1) - COST)
    if pos == -1: trades.append((c[n-1]/entry - 1)*-1 - COST)
    return trades

def strat_momentum(bars, lb, thr):
    c = np.array([b["close"] for b in bars], dtype=float)
    n = len(c)
    if n < lb + 5: return []
    h = np.array([b["high"] for b in bars], dtype=float)
    l = np.array([b["low"] for b in bars], dtype=float)
    a = atr(h, l, c)
    trades, pos, entry, stop = [], 0, 0.0, 0.0
    for i in range(lb+1, n):
        ret = c[i]/c[i-lb] - 1
        if pos == 0:
            if ret > thr:
                pos, entry, stop = 1, c[i], c[i] - 2.0*a[i]
            elif ret < -thr:
                pos, entry, stop = -1, c[i], c[i] + 2.0*a[i]
        elif pos == 1:
            if l[i] <= stop:
                trades.append((stop/entry - 1) - COST); pos = 0
            else: stop = max(stop, c[i] - 2.0*a[i])
        else:
            if h[i] >= stop:
                trades.append((stop/entry - 1)*-1 - COST); pos = 0
            else: stop = min(stop, c[i] + 2.0*a[i])
    if pos == 1: trades.append((c[n-1]/entry - 1) - COST)
    if pos == -1: trades.append((c[n-1]/entry - 1)*-1 - COST)
    return trades

# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def metrics(trades):
    if len(trades) < 5:
        return {"n": len(trades), "sharpe": 0.0, "pf": 0.0, "wr": 0.0,
                "maxdd": 0.0, "ret": 0.0}
    rets = np.array(trades)
    wins = rets[rets > 0]; loss = rets[rets <= 0]
    wr = len(wins)/len(rets)*100
    gp = wins.sum() if len(wins) else 0.0
    gl = abs(loss.sum()) if len(loss) else 1e-9
    pf = gp/gl if gl > 0 else 99.9
    ann = rets.mean()/rets.std()*np.sqrt(252) if rets.std() > 0 else 0.0
    cum = np.cumsum(rets); peak = np.maximum.accumulate(cum)
    dd = float(np.max(peak - cum))*100 if len(cum) else 0.0
    return {"n": int(len(rets)), "sharpe": float(ann), "pf": float(pf),
            "wr": float(wr), "maxdd": dd, "ret": float(rets.sum()*100)}

# ---------------------------------------------------------------------------
# PARAM GRIDS per family
# ---------------------------------------------------------------------------
GRIDS = {
    "donchian": {"lb": [10,20,30,50], "rr": [2.0,3.0], "use_short": [False]},
    "ema":      {"fast": [5,10,20], "slow": [20,50,100]},
    "momentum": {"lb": [10,20,50], "thr": [0.05,0.10]},
}
FAM = {
    "donchian": (strat_donchian, ["lb","rr","use_short"]),
    "ema":      (strat_ema, ["fast","slow"]),
    "momentum": (strat_momentum, ["lb","thr"]),
}

def gen_params(family):
    """Yield dicts of params for the family grid."""
    import itertools
    g = GRIDS[family]
    keys = list(g.keys()); vals = [g[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def run_family(bars, family):
    """Train/test split: optimize on first 70%, test on last 30%."""
    if len(bars) < 260:
        return None
    cut = int(len(bars) * 0.7)
    train, test = bars[:cut], bars[cut:]
    func = FAM[family][0]

    # Train: pick best params by Sharpe (min trades guard)
    best_p, best_sh = None, -1e9
    for p in gen_params(family):
        tr = func(train, **p)
        m = metrics(tr)
        if m["n"] >= 15 and m["sharpe"] > best_sh:
            best_sh = m["sharpe"]; best_p = p

    if best_p is None:
        return None

    # Test: run chosen params ONLY on unseen test slice
    tt = func(test, **best_p)
    tm = metrics(tt)
    fm = metrics(func(bars, **best_p))  # full-sample context
    return {"family": family, "params": best_p, "train_sharpe": best_sh,
            "test": tm, "full": fm}

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--budget", type=int, default=360, help="max minutes budget")
    args = ap.parse_args()

    symbols = list(dict.fromkeys(args.symbols if args.symbols else SYMBOLS))
    t0 = time.time()

    print(f"MIMO SWEEP — {len(symbols)} symbols x {len(GRIDS)} families")
    print(f"Budget: {args.budget} min | Costs: {COST*100:.1f}% RT | Train/Test: 70/30")
    print("=" * 90)

    results = []
    deadline = t0 + args.budget * 60

    for sym in symbols:
        if time.time() > deadline:
            print(f"\n[BUDGET REACHED] stopping at symbol {sym}")
            break
        print(f"\n--- {sym} ---")
        bars = fetch_yahoo(sym)
        if len(bars) < 260:
            print(f"  skipped: only {len(bars)} bars")
            continue
        print(f"  {len(bars)} bars {bars[0]['date']}..{bars[-1]['date']}")
        for fam in GRIDS:
            if time.time() > deadline:
                break
            r = run_family(bars, fam)
            if r and r["test"]["n"] >= 10:
                t = r["test"]; f = r["full"]
                star = "CANDIDATE" if (t["sharpe"] > 0.8 and t["n"] >= 20 and t["pf"] > 1.3) else ""
                print(f"  [{fam:9s}] {r['params']}  TRAIN Sh={r['train_sharpe']:.2f} | "
                      f"TEST n={t['n']} Sh={t['sharpe']:.2f} PF={t['pf']:.2f} WR={t['wr']:.0f}% "
                      f"DD={t['maxdd']:.1f}% Ret={t['ret']:.1f}%  {star}")
                results.append({"symbol": sym, **r})
        # brief pause to avoid hammering Yahoo (gentler pacing for ~260 symbols)
        time.sleep(1.0)

    # -------------------------------------------------------------------------
    # RANKING & FINAL REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("FINAL REPORT — Walk-forward candidates (OOS Sharpe>0.8, n>=20, PF>1.3)")
    print("=" * 90)

    cands = [r for r in results
             if r["test"]["sharpe"] > 0.8 and r["test"]["n"] >= 20 and r["test"]["pf"] > 1.3]
    cands.sort(key=lambda r: r["test"]["sharpe"], reverse=True)

    if not cands:
        print("\nNO candidates passed the out-of-sample bar.")
        print("Showing best test results for context:")
        best = sorted(results, key=lambda r: r["test"]["sharpe"], reverse=True)[:15]
        for r in best:
            print(f"  {r['symbol']:10s} {r['family']:9s} {r['params']}  "
                  f"OOS n={r['test']['n']} Sh={r['test']['sharpe']:.2f} "
                  f"PF={r['test']['pf']:.2f} (train Sh={r['train_sharpe']:.2f})")
    else:
        print(f"\n{len(cands)} candidate(s) survived train/test split.\n")
        print(f"{'Symbol':12s} {'Family':9s} {'Params':30s} {'OOS_n':>5s} {'OOS_Sh':>7s} "
              f"{'OOS_PF':>6s} {'OOS_WR':>6s} {'OOS_DD':>6s} {'OOS_Ret':>7s}")
        print("-" * 92)
        for r in cands:
            t = r["test"]
            print(f"{r['symbol']:12s} {r['family']:9s} {str(r['params']):30s} "
                  f"{t['n']:5d} {t['sharpe']:7.2f} {t['pf']:6.2f} {t['wr']:6.1f} "
                  f"{t['maxdd']:6.1f}% {t['ret']:7.1f}%")

    # Diversification note
    if cands:
        syms = sorted(set(r["symbol"] for r in cands))
        fams = sorted(set(r["family"] for r in cands))
        print(f"\nCovers {len(syms)} symbols: {', '.join(syms)}")
        print(f"Families: {', '.join(fams)}")
        if len(syms) < 3:
            print("WARNING: very few symbols — likely single-regime luck, not durable edge.")
        print("Next: deep-validate survivors on MiMo (longer history, MC, 2020+ not in train).")

    print(f"\nElapsed: {(time.time()-t0)/60:.1f} min. Cache in {CACHE_DIR}/")

if __name__ == "__main__":
    main()