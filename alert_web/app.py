import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.getenv(
    "STRATEGY_ALERT_CONFIG",
    PROJECT_ROOT / "scheduler_alerts" / "strategy_alert_config.json",
))

DEFAULT_CONFIG = {
    "defaults": {
        "interval": "1d",
        "history_days": 900,
        "ema_length": 200,
        "initial_balance": 10000,
        "leverage": 1,
        "alert_on": ["entry_signal", "exit_signal"],
    },
    "symbols": {},
}

VALID_MODES = {
    "tp",
    "trend",
    "cross_trend",
    "weekly_trend",
    "weekly_bull_ema",
    "adx_trend",
    "adx_tp",
    "adx_uptrend",
    "adx_uptrend_tp",
}

app = Flask(__name__)


def load_config():
    if not CONFIG_PATH.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_config(config):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(CONFIG_PATH)


def positive_int(value, field_name, default=None):
    if value in {None, ""}:
        if default is not None:
            return default
        raise ValueError(f"{field_name} is required.")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return number


def positive_float(value, field_name, default=None):
    if value in {None, ""}:
        if default is not None:
            return default
        raise ValueError(f"{field_name} is required.")
    number = float(value)
    if number <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return number


def normalize_symbol_config(payload):
    symbol = str(payload.get("symbol", "")).strip().upper()
    if not symbol:
        raise ValueError("Symbol is required.")

    mode = str(payload.get("mode", "weekly_trend")).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Mode must be one of: {', '.join(sorted(VALID_MODES))}.")

    return symbol, {
        "enabled": bool(payload.get("enabled", True)),
        "source": str(payload.get("source", "phone_alert_web")).strip() or "phone_alert_web",
        "mode": mode,
        "interval": str(payload.get("interval", "1d")).strip().lower() or "1d",
        "history_days": positive_int(payload.get("history_days"), "history_days", 900),
        "atr_length": positive_int(payload.get("atr_length"), "atr_length"),
        "factor": positive_float(payload.get("factor"), "factor"),
        "swing_lookback": positive_int(payload.get("swing_lookback"), "swing_lookback"),
        "tp_multiplier": positive_float(payload.get("tp_multiplier"), "tp_multiplier", 1.0),
        "max_trades": positive_int(payload.get("max_trades"), "max_trades"),
        "adx_threshold": positive_float(payload.get("adx_threshold"), "adx_threshold", 25),
        "adx_trend_lookback": positive_int(payload.get("adx_trend_lookback"), "adx_trend_lookback", 3),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({
        "status": "success",
        "config_path": str(CONFIG_PATH),
        "config": load_config(),
    })


@app.route("/api/symbols", methods=["POST"])
def save_symbol():
    try:
        symbol, symbol_config = normalize_symbol_config(request.get_json(silent=True) or {})
        config = load_config()
        config.setdefault("defaults", DEFAULT_CONFIG["defaults"])
        symbols = config.setdefault("symbols", {})
        previous = symbols.get(symbol)
        symbols[symbol] = symbol_config
        save_config(config)
        return jsonify({
            "status": "success",
            "message": f"Saved alert config for {symbol}.",
            "symbol": symbol,
            "previous": previous,
            "current": symbol_config,
            "config_path": str(CONFIG_PATH),
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/api/symbols/<symbol>", methods=["DELETE"])
def delete_symbol(symbol):
    try:
        normalized_symbol = str(symbol or "").strip().upper()
        if not normalized_symbol:
            return jsonify({"status": "error", "message": "Symbol is required."}), 400

        config = load_config()
        symbols = config.setdefault("symbols", {})
        previous = symbols.pop(normalized_symbol, None)
        if previous is None:
            return jsonify({
                "status": "error",
                "message": f"{normalized_symbol} was not found in the alert config.",
            }), 404

        save_config(config)
        return jsonify({
            "status": "success",
            "message": f"Removed alert config for {normalized_symbol}.",
            "symbol": normalized_symbol,
            "removed": previous,
            "config_path": str(CONFIG_PATH),
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")), debug=False)
