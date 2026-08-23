import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib import request, error

import pandas as pd
import yfinance as yf

SCHEDULER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCHEDULER_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from strategy import SupertrendStrategy  # noqa: E402


CONFIG_PATH = Path(os.getenv("STRATEGY_ALERT_CONFIG", SCHEDULER_DIR / "strategy_alert_config.json"))
STATE_PATH = Path(os.getenv("ALERT_STATE_PATH", SCHEDULER_DIR / "alert_state.json"))
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
YFINANCE_CACHE_DIR = Path(os.getenv("YFINANCE_CACHE_DIR", SCHEDULER_DIR / ".yfinance_cache"))

MODE_CONFIG = {
    "tp": {
        "entry_mode": "flip",
        "exit_mode": "tp",
        "label": "Supertrend Exit with TP",
        "description": "Daily Supertrend flip strategy with take-profit exits. Longs require Supertrend above EMA200; shorts require it below EMA200.",
    },
    "trend": {
        "entry_mode": "flip",
        "exit_mode": "trend",
        "label": "Supertrend Exit on Trend Change",
        "description": "Daily Supertrend flip strategy with trend-change exits. Longs require Supertrend above EMA200; shorts require it below EMA200.",
    },
    "supertrend_no_ema_trend": {
        "entry_mode": "flip_no_ema",
        "exit_mode": "trend",
        "label": "Supertrend Exit on Trend Change",
        "description": "Daily Supertrend flip strategy with trend-change exits. Requires price to be on the correct side of EMA200, but does not require Supertrend to already be above or below EMA200.",
    },
    "supertrend_no_ema_tp": {
        "entry_mode": "flip_no_ema",
        "exit_mode": "tp",
        "label": "Supertrend Exit with TP",
        "description": "Daily Supertrend flip strategy with take-profit exits. Requires price to be on the correct side of EMA200, but does not require Supertrend to already be above or below EMA200.",
    },
    "cross_trend": {
        "entry_mode": "cross",
        "exit_mode": "trend",
        "label": "Supertrend/EMA Cross + Trend Change",
        "description": "Entries start on Supertrend/EMA200 cross signals, then continue with follow-on trend entries while the setup stays valid.",
    },
    "cross_tp": {
        "entry_mode": "cross",
        "exit_mode": "tp",
        "label": "Supertrend/EMA Cross + TP",
        "description": "Entries start on Supertrend/EMA200 cross signals, then continue with follow-on trend entries while the setup stays valid, using take-profit exits.",
    },
    "weekly_trend": {
        "entry_mode": "weekly_long",
        "exit_mode": "trend",
        "label": "Weekly Filter + Trend Change",
        "description": "Daily Supertrend flip entries that only fire when the weekly trend agrees and Supertrend is on the correct side of EMA200.",
    },
    "weekly_bull_ema": {
        "entry_mode": "weekly_bull_ema",
        "exit_mode": "trend",
        "label": "Daily + Weekly + EMA200",
        "description": "Daily Supertrend trend entries with weekly confirmation and an EMA200 Supertrend filter on both sides.",
    },
    "adx_trend": {
        "entry_mode": "adx_anytime",
        "exit_mode": "trend",
        "label": "Supertrend + EMA200 + ADX",
        "description": "Supertrend entries can occur anytime when daily direction, EMA200, Supertrend vs EMA200, and ADX conditions all align.",
    },
    "adx_tp": {
        "entry_mode": "adx_anytime",
        "exit_mode": "tp",
        "label": "Supertrend + EMA200 + ADX with TP",
        "description": "Same as the ADX anytime strategy, but exits use take-profit instead of trend-change exits.",
    },
    "adx_uptrend": {
        "entry_mode": "adx_uptrend",
        "exit_mode": "trend",
        "label": "Supertrend + EMA200 + ADX Rising",
        "description": "ADX anytime strategy with an extra rule that ADX must be rising over the configured lookback window.",
    },
    "adx_uptrend_tp": {
        "entry_mode": "adx_uptrend",
        "exit_mode": "tp",
        "label": "Supertrend + EMA200 + ADX Rising with TP",
        "description": "ADX rising strategy with take-profit exits instead of trend-change exits.",
    },
}

ADX_MODES = {"adx_trend", "adx_tp", "adx_uptrend", "adx_uptrend_tp"}
ADX_UPTREND_MODES = {"adx_uptrend", "adx_uptrend_tp"}
TP_MODES = {"tp", "adx_tp", "adx_uptrend_tp", "supertrend_no_ema_tp"}


def load_json(path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def normalize_yfinance_frame(data):
    if data is None or data.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    frame = data.copy()
    if frame.index.name is None:
        frame.index.name = "Date"
    frame.reset_index(inplace=True)

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [col[0] for col in frame.columns]

    if "Date" not in frame.columns:
        for candidate in ("Datetime", "index"):
            if candidate in frame.columns:
                frame = frame.rename(columns={candidate: "Date"})
                break

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Downloaded data missing columns: {', '.join(missing)}")

    out = frame[required].copy()
    dates = pd.to_datetime(out["Date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    out["Date"] = dates
    return out.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)


def download_history(symbol, interval, history_days):
    YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        yf.cache.set_cache_location(str(YFINANCE_CACHE_DIR))
    except Exception as exc:
        print(f"[WARN] Could not configure yfinance cache: {exc}")

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=int(history_days))
    data = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    return normalize_yfinance_frame(data)


def merged_params(defaults, symbol_params):
    merged = dict(defaults or {})
    merged.update(symbol_params or {})
    return merged


def build_strategy(params):
    mode = str(params.get("mode", "weekly_trend")).strip().lower()
    if mode not in MODE_CONFIG:
        raise ValueError(f"Unsupported mode {mode!r}. Use one of: {', '.join(MODE_CONFIG)}")

    mode_config = MODE_CONFIG[mode]
    return SupertrendStrategy(
        atr_length=int(params.get("atr_length", 14)),
        factor=float(params.get("factor", 3.0)),
        ema_length=int(params.get("ema_length", 200)),
        swing_lookback=int(params.get("swing_lookback", 12)),
        tp_multiplier=float(params.get("tp_multiplier", 1.0)),
        max_trades=int(params.get("max_trades", 1)),
        leverage=float(params.get("leverage", 1)),
        initial_balance=float(params.get("initial_balance", 10000)),
        long_only=bool(params.get("long_only", False)),
        exit_mode=mode_config["exit_mode"],
        entry_mode=mode_config["entry_mode"],
        adx_threshold=float(params.get("adx_threshold", 25)),
        adx_trend_lookback=int(params.get("adx_trend_lookback", 3)),
    )


def format_price(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def event_key(symbol, mode, event_type, event_date):
    return f"{symbol}:{mode}:{event_type}:{pd.to_datetime(event_date).strftime('%Y-%m-%d')}"


def format_strategy_params(params):
    mode = str(params.get("mode", "weekly_trend")).strip().lower()
    mode_label = MODE_CONFIG.get(mode, {}).get("label", mode)

    parts = [
        f"mode={mode_label}",
        f"atr_length={int(params.get('atr_length', 14))}",
        f"factor={float(params.get('factor', 3.0)):g}",
        f"ema_length={int(params.get('ema_length', 200))}",
        f"swing_lookback={int(params.get('swing_lookback', 12))}",
        f"max_trades={int(params.get('max_trades', 1))}",
    ]

    if "tp_multiplier" in params or mode in TP_MODES:
        parts.append(f"tp_multiplier={float(params.get('tp_multiplier', 1.0)):g}")

    if mode in ADX_MODES:
        parts.append(f"adx_threshold={float(params.get('adx_threshold', 25)):g}")

    if mode in ADX_UPTREND_MODES:
        parts.append(f"adx_trend_lookback={int(params.get('adx_trend_lookback', 3))}")

    if params.get("source"):
        parts.append(f"source={params['source']}")

    return ", ".join(parts)


def format_strategy_description(params):
    mode = str(params.get("mode", "weekly_trend")).strip().lower()
    return MODE_CONFIG.get(mode, {}).get("description", "Select a strategy to see its rules.")


def collect_latest_events(symbol, params, results, trades):
    if results.empty:
        return []

    mode = str(params.get("mode", "weekly_trend")).strip().lower()
    mode_label = MODE_CONFIG[mode]["label"]
    strategy_params = format_strategy_params(params)
    strategy_description = format_strategy_description(params)
    alert_on = set(params.get("alert_on") or ["entry_signal", "exit_signal"])
    latest = results.iloc[-1]
    latest_date = latest["Date"]
    adx_line = f"ADX: {format_price(latest.get('adx'))}\n" if mode in ADX_MODES else ""
    events = []

    if "entry_signal" in alert_on and int(latest.get("signal", 0) or 0) in {1, -1}:
        side = "LONG" if int(latest["signal"]) == 1 else "SHORT"
        events.append({
            "key": event_key(symbol, mode, f"{side.lower()}_entry_signal", latest_date),
            "title": f"{symbol} {side} entry signal",
            "body": (
                f"Signal bar: {pd.to_datetime(latest_date).strftime('%Y-%m-%d')}\n"
                f"Mode: {mode_label}\n"
                f"Strategy: {strategy_description}\n"
                f"Strategy params: {strategy_params}\n"
                f"Close: {format_price(latest.get('Close'))}\n"
                f"EMA {params.get('ema_length', 200)}: {format_price(latest.get('ema200'))}\n"
                f"Supertrend: {'Bullish' if int(latest.get('direction', 0)) == 1 else 'Bearish'}\n"
                f"{adx_line}"
                f"Weekly Supertrend: {format_weekly_direction(latest)}\n"
                "Action: review for next market open."
            ),
        })

    if "exit_signal" in alert_on:
        for trade in trades or []:
            exit_date = trade.get("exit_date")
            if exit_date is None:
                continue
            if str(trade.get("exit_reason") or "").strip().lower() == "end of data":
                continue
            if pd.to_datetime(exit_date).date() != pd.to_datetime(latest_date).date():
                continue
            events.append({
                "key": event_key(symbol, mode, f"{trade.get('type', 'position')}_exit", exit_date),
                "title": f"{symbol} exit signal",
                "body": (
                    f"Exit date: {pd.to_datetime(exit_date).strftime('%Y-%m-%d')}\n"
                    f"Mode: {mode_label}\n"
                    f"Strategy: {strategy_description}\n"
                    f"Strategy params: {strategy_params}\n"
                    f"Side: {str(trade.get('type') or '-').upper()}\n"
                    f"Exit reason: {trade.get('exit_reason') or '-'}\n"
                    f"Entry: {format_price(trade.get('entry_price'))}\n"
                    f"Exit: {format_price(trade.get('exit_price'))}\n"
                    f"PnL: {format_price(trade.get('pnl'))}\n"
                    f"Return: {format_price(trade.get('return'))}%"
                ),
            })

    return events


def format_weekly_direction(row):
    direction = row.get("weekly_direction")
    if direction is None or pd.isna(direction):
        return "-"
    direction = int(direction)
    if direction == 1:
        return "Bullish"
    if direction == -1:
        return "Bearish"
    return "-"


def send_slack_message(text):
    if not SLACK_WEBHOOK_URL:
        print("[WARN] SLACK_WEBHOOK_URL is not set. Message not sent:")
        print(text)
        return

    payload = json.dumps({"text": text}).encode("utf-8")
    req = request.Request(
        SLACK_WEBHOOK_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=20) as response:
            if response.status >= 400:
                raise RuntimeError(f"Slack returned HTTP {response.status}")
    except error.URLError as exc:
        raise RuntimeError(f"Failed to send Slack message: {exc}") from exc


def run_symbol(symbol, params):
    interval = str(params.get("interval", "1d"))
    history_days = int(params.get("history_days", 900))
    data = download_history(symbol, interval, history_days)
    if data.empty:
        raise ValueError(f"No data downloaded for {symbol}")

    strategy = build_strategy(params)
    results, trades = strategy.backtest(data.copy())
    return collect_latest_events(symbol, params, results, trades), results


def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found: {CONFIG_PATH}. Copy strategy_alert_config.example.json "
            "to strategy_alert_config.json and edit your symbols."
        )

    config = load_json(CONFIG_PATH, {})
    defaults = config.get("defaults", {})
    symbols = config.get("symbols", {})
    state = load_json(STATE_PATH, {"sent": {}})
    sent = state.setdefault("sent", {})

    pending_events = []
    for symbol, symbol_params in symbols.items():
        params = merged_params(defaults, symbol_params)
        if not params.get("enabled", True):
            print(f"[SKIP] {symbol} disabled")
            continue

        symbol = symbol.strip().upper()
        try:
            events, results = run_symbol(symbol, params)
            latest_date = results.iloc[-1]["Date"].strftime("%Y-%m-%d") if not results.empty else "-"
            print(f"[OK] {symbol}: checked through {latest_date}, events={len(events)}")
            for event in events:
                if event["key"] in sent:
                    print(f"[SKIP] Already alerted {event['key']}")
                    continue
                pending_events.append(event)
        except Exception as exc:
            pending_events.append({
                "key": event_key(symbol, str(params.get("mode", "unknown")), "error", datetime.utcnow()),
                "title": f"{symbol} alert check failed",
                "body": str(exc),
                "persist": False,
            })

    for event in pending_events:
        text = f"*{event['title']}*\n{event['body']}"
        send_slack_message(text)
        if event.get("persist", True):
            sent[event["key"]] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        print(f"[SENT] {event['key']}")

    state["last_run_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    save_json(STATE_PATH, state)
    print(f"[DONE] sent={len(pending_events)}")


if __name__ == "__main__":
    main()
