import importlib
import io
import os
import sys
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_ohlcv_df(
    n=120,
    base_price=2650.0,
    step=0.25,
    start_dt=None,
    freq_minutes=5,
    wick=0.8,
    oscillate=False,
):
    if start_dt is None:
        start_dt = datetime(2026, 11, 17, 10, 0, 0, tzinfo=timezone.utc)
    times = [start_dt + timedelta(minutes=freq_minutes * i) for i in range(n)]
    if oscillate:
        closes = [base_price + step * i + (1.2 if i % 2 == 0 else -1.2) for i in range(n)]
    else:
        closes = [base_price + step * i for i in range(n)]
    opens = [closes[0] - 0.1] + closes[:-1]
    highs = [max(o, c) + wick for o, c in zip(opens, closes)]
    lows = [min(o, c) - wick for o, c in zip(opens, closes)]
    return pd.DataFrame(
        {
            "time": pd.to_datetime(times),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": [150 + (i % 20) for i in range(n)],
        }
    )


class _DummyStrategy:
    def __init__(self, name, payload=None, ui="[SCAN]"):
        self.name = name
        self._payload = payload
        self._ui = ui
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        p = self._payload(kwargs) if callable(self._payload) else self._payload
        return {"ui": self._ui, "payload": p}


def _setup_goldvx_harness(
    monkeypatch,
    tmp_path,
    *,
    check_dd_side_effect=None,
    check_dd_func=None,
    strategy_payload=None,
    time_step=2.0,
    fixed_utc_now=None,
    is_paused_val=False,
    deals_side_effect=None,
    positions_total_val=0,
):
    import src.goldvx as goldvx
    from core.state_manager import StateManager
    import core.license_manager as license_manager

    if fixed_utc_now is None:
        fixed_utc_now = datetime(2026, 11, 17, 12, 0, 0, tzinfo=timezone.utc)

    # Isolate persistent state so real data/nexus_state.json is never touched
    state_mgr = StateManager(
        state_file=str(tmp_path / "nexus_state.json"),
        lock_file=str(tmp_path / "nexus_state.lock"),
    )
    state_mgr.set_instance = MagicMock()
    monkeypatch.setattr(goldvx, "nexus_state", state_mgr)

    # Prevent CLI arg leakage from pytest
    monkeypatch.setattr(sys, "argv", ["goldvx.py", "--no-telegram", "--no-broadcast"])

    # Auth, MT5 connect, AI models
    monkeypatch.setattr(license_manager, "authenticate_nexus_session", MagicMock())
    monkeypatch.setattr(goldvx, "connect_mt5", MagicMock())
    monkeypatch.setattr(goldvx, "load_ai_models", MagicMock(return_value=(MagicMock(), MagicMock())))

    # Deterministic UTC clock without breaking datetime.now()
    class _FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return fixed_utc_now.astimezone(tz)
            return fixed_utc_now.replace(tzinfo=None)

    monkeypatch.setattr(goldvx, "datetime", _FakeDateTime)

    # Controllable monotonic wall clock + non-blocking sleep recorder
    clock = {"t": 1_700_000_000.0}
    sleep_calls = []

    def _fake_time():
        return clock["t"]

    def _fake_sleep(secs):
        sleep_calls.append(secs)
        clock["t"] += time_step

    monkeypatch.setattr(goldvx.time, "time", _fake_time)
    monkeypatch.setattr(goldvx.time, "sleep", _fake_sleep)

    # Mock MT5 API used in main()
    fake_tick = types.SimpleNamespace(bid=2680.50, ask=2680.70)
    fake_account = types.SimpleNamespace(equity=10000.0, balance=10000.0, login=9990001)
    fake_mt5 = MagicMock()
    fake_mt5.TIMEFRAME_H4 = 16388
    fake_mt5.ORDER_TYPE_BUY = 0
    fake_mt5.symbol_info_tick.return_value = fake_tick
    fake_mt5.account_info.return_value = fake_account
    if deals_side_effect is not None:
        fake_mt5.history_deals_total.side_effect = deals_side_effect
    else:
        fake_mt5.history_deals_total.return_value = 0
    fake_mt5.positions_total.return_value = positions_total_val
    fake_mt5.positions_get.return_value = []
    # Market data frames (oscillating so RSI stays ~50 and ADX stays moderate for realistic guard pass-through)
    df_m5 = _make_ohlcv_df(120, base_price=2650.0, step=0.05, freq_minutes=5, oscillate=True)
    df_h1 = _make_ohlcv_df(100, base_price=2640.0, step=0.10, freq_minutes=60, oscillate=True)
    df_h4 = _make_ohlcv_df(100, base_price=2620.0, step=0.15, freq_minutes=240, oscillate=True)
    df_adx = _make_ohlcv_df(100, base_price=2645.0, step=0.05, freq_minutes=15, oscillate=True)
    fake_mt5.copy_rates_from_pos.return_value = [{"time": df_m5.iloc[-1]["time"]}]

    monkeypatch.setattr(goldvx, "mt5", fake_mt5)

    def _fake_get_market_data(symbol, tf, n=100):
        if tf == goldvx.TIMEFRAME_ENTRY:
            return df_m5.copy()
        if tf == goldvx.TIMEFRAME_TREND:
            return df_h1.copy()
        if tf == fake_mt5.TIMEFRAME_H4:
            return df_h4.copy()
        if tf == goldvx.TIMEFRAME_ADX:
            return df_adx.copy()
        return df_m5.copy()

    get_market_data_mock = MagicMock(side_effect=_fake_get_market_data)
    monkeypatch.setattr(goldvx, "get_market_data", get_market_data_mock)

    # Spy on indicator functions while keeping real calculations
    atr_spy = MagicMock(wraps=goldvx.calculate_atr_simple)
    rsi_spy = MagicMock(wraps=goldvx.calculate_rsi_simple)
    adx_spy = MagicMock(wraps=goldvx.calculate_adx_simple)
    monkeypatch.setattr(goldvx, "calculate_atr_simple", atr_spy)
    monkeypatch.setattr(goldvx, "calculate_rsi_simple", rsi_spy)
    monkeypatch.setattr(goldvx, "calculate_adx_simple", adx_spy)
    monkeypatch.setattr(goldvx, "check_volatility_guard", MagicMock(return_value=(True, 1.0)))

    # News, ledger, supabase, telegram, position management, cooldowns, trade exec
    monkeypatch.setattr(goldvx, "fetch_economic_news", MagicMock())
    monkeypatch.setattr(goldvx, "is_news_blackout", MagicMock(return_value=False))
    monkeypatch.setattr(
        goldvx, "get_post_news_guard_status", MagicMock(return_value={"in_post_window": False})
    )
    export_ledger_mock = MagicMock()
    monkeypatch.setattr(goldvx, "export_trade_ledger", export_ledger_mock)
    sync_all_mock = MagicMock()
    monkeypatch.setattr(goldvx, "sync_all", sync_all_mock)

    # Run threading.Thread targets synchronously for deterministic assertions
    class _SyncThread:
        def __init__(self, target=None, daemon=None, args=(), kwargs=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            if self._target:
                self._target(*self._args, **self._kwargs)

    monkeypatch.setattr(goldvx.threading, "Thread", _SyncThread)

    manage_pos_mock = MagicMock()
    manage_ord_mock = MagicMock()
    monkeypatch.setattr(goldvx, "manage_open_positions", manage_pos_mock)
    monkeypatch.setattr(goldvx, "manage_pending_orders", manage_ord_mock)
    monkeypatch.setattr(goldvx, "close_all_positions", MagicMock())
    monkeypatch.setattr(goldvx, "get_dynamic_kelly_risk", MagicMock(side_effect=lambda r: r))
    monkeypatch.setattr(
        goldvx,
        "get_dynamic_strategy_ranking",
        MagicMock(
            side_effect=lambda all_league: (
                list(all_league),
                {s: {"win_rate": 60.0, "net_profit": 100.0, "total_trades": 3} for s in all_league},
            )
        ),
    )
    monkeypatch.setattr(goldvx, "check_global_directional_cooldown", MagicMock(return_value=(False, 0)))
    monkeypatch.setattr(goldvx, "check_strategy_cooldown", MagicMock(return_value=(False, 0)))
    monkeypatch.setattr(goldvx, "notify_regime_change", MagicMock())

    bot_state_updates = []

    def _capture_bot_state(**kwargs):
        bot_state_updates.append(dict(kwargs))

    update_bot_state_mock = MagicMock(side_effect=_capture_bot_state)
    monkeypatch.setattr(goldvx, "update_bot_state", update_bot_state_mock)
    monkeypatch.setattr(goldvx, "is_paused", MagicMock(return_value=is_paused_val))

    execute_trade_mock = MagicMock(return_value=True)
    monkeypatch.setattr(goldvx, "execute_trade", execute_trade_mock)

    # Drawdown mock
    if check_dd_func is not None:
        dd_mock = MagicMock(side_effect=check_dd_func)
    elif check_dd_side_effect is not None:
        dd_mock = MagicMock(side_effect=check_dd_side_effect)
    else:
        dd_mock = MagicMock(return_value=True)
    monkeypatch.setattr(goldvx, "check_daily_drawdown", dd_mock)

    # Replace strategy classes with controllable spies
    dummy_strategies = {
        "SMC_OB": _DummyStrategy("SMC_OB", payload=strategy_payload, ui="[OB:1]"),
        "LIQUIDITY_SWEEP": _DummyStrategy("LIQUIDITY_SWEEP", payload=None, ui="[LIQ B:2690.0 S:2670.0]"),
        "RSI_REVERSION": _DummyStrategy("RSI_REVERSION", payload=None, ui="[RSI-RNG:48]"),
        "BB_BREAKOUT": _DummyStrategy("BB_BREAKOUT", payload=None, ui="[BB:48]"),
    }
    monkeypatch.setattr(goldvx, "SMCOrderBlockStrategy", lambda *a, **kw: dummy_strategies["SMC_OB"])
    monkeypatch.setattr(goldvx, "LiquiditySweepStrategy", lambda *a, **kw: dummy_strategies["LIQUIDITY_SWEEP"])
    monkeypatch.setattr(goldvx, "RSIReversionStrategy", lambda *a, **kw: dummy_strategies["RSI_REVERSION"])
    monkeypatch.setattr(goldvx, "BBBreakoutStrategy", lambda *a, **kw: dummy_strategies["BB_BREAKOUT"])

    # Capture HUD writes safely
    hud_writes = []
    orig_stdout = sys.stdout

    class _CapturingStdout(io.StringIO):
        def write(self, s):
            hud_writes.append(s)
            return super().write(s)

    monkeypatch.setattr(sys, "stdout", _CapturingStdout())

    return {
        "goldvx": goldvx,
        "state_mgr": state_mgr,
        "mt5": fake_mt5,
        "dd_mock": dd_mock,
        "get_market_data": get_market_data_mock,
        "atr_spy": atr_spy,
        "rsi_spy": rsi_spy,
        "adx_spy": adx_spy,
        "export_ledger": export_ledger_mock,
        "sync_all": sync_all_mock,
        "manage_pos": manage_pos_mock,
        "manage_ord": manage_ord_mock,
        "update_bot_state": update_bot_state_mock,
        "bot_state_updates": bot_state_updates,
        "execute_trade": execute_trade_mock,
        "strategies": dummy_strategies,
        "hud_writes": hud_writes,
        "sleep_calls": sleep_calls,
        "clock": clock,
        "orig_stdout": orig_stdout,
    }


def test_import_goldvx_with_stringio_stdout(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    import src.goldvx as goldvx

    reloaded = importlib.reload(goldvx)
    assert hasattr(reloaded, "main")


def test_drawdown_halt_continues_market_viewing_without_60s_sleep(monkeypatch, tmp_path):
    payload = {
        "signal": "BUY",
        "sl": 2675.0,
        "tp_price": 2690.0,
        "confidence": 0.85,
        "comment": "OB TEST",
        "features": [0.1, 0.2],
    }
    h = _setup_goldvx_harness(
        monkeypatch,
        tmp_path,
        check_dd_side_effect=[False, False, False],
        strategy_payload=payload,
        time_step=2.0,
        deals_side_effect=[0, 1, 1, 1],
    )

    h["goldvx"].main(max_iterations=3)

    # Never short-circuits with time.sleep(60); runs 2.0s observation cadence
    assert 60 not in h["sleep_calls"]
    assert h["sleep_calls"] == [2.0, 2.0, 2.0]

    # check_daily_drawdown still runs every iteration
    assert h["dd_mock"].call_count == 3

    # Market viewing, indicators, position/order management, ledger, and sync keep running
    assert h["get_market_data"].call_count >= 3
    assert h["atr_spy"].call_count >= 3
    assert h["rsi_spy"].call_count >= 3
    assert h["adx_spy"].call_count >= 3
    assert h["manage_pos"].call_count == 3
    assert h["manage_ord"].call_count == 3
    assert h["export_ledger"].call_count >= 1
    assert h["sync_all"].call_count >= 2  # initial boot sync + deal-increase sync

    # Bot state & HUD expose [DD-HALT] and observe_only=True
    assert len(h["bot_state_updates"]) == 3
    for upd in h["bot_state_updates"]:
        assert "[DD-HALT]" in upd["regime"]
        assert upd["drawdown_halted"] is True
        assert upd["observe_only"] is True

    hud_line_writes = [w for w in h["hud_writes"] if "\033[K" in w]
    assert len(hud_line_writes) == 3
    assert all("[DD-HALT]" in w for w in hud_line_writes)

    # Strategies receive observe_only=True, and execute_trade is NEVER called
    for strat in h["strategies"].values():
        assert len(strat.calls) == 3
        assert all(c["observe_only"] is True for c in strat.calls)
    assert h["execute_trade"].call_count == 0


def test_drawdown_halt_blocks_execute_trade_even_after_telegram_resume(monkeypatch, tmp_path):
    payload = {
        "signal": "SELL",
        "sl": 2686.0,
        "tp_price": 2670.0,
        "confidence": 0.90,
        "comment": "TEST_RESUME_BLOCK",
        "features": None,
    }
    # Simulate /resume having set is_paused() -> False while drawdown halt is active
    h = _setup_goldvx_harness(
        monkeypatch,
        tmp_path,
        check_dd_side_effect=[False, False],
        strategy_payload=payload,
        is_paused_val=False,
    )

    h["goldvx"].main(max_iterations=2)

    assert h["execute_trade"].call_count == 0
    assert all(c["observe_only"] is True for c in h["strategies"]["SMC_OB"].calls)


def test_waking_from_observe_does_not_bypass_throttle_during_halt(monkeypatch, tmp_path):
    # Sub-1.5s loop steps (0.5s) with identical M5 bar timestamp across 4 halted ticks + 1 recovery tick
    h = _setup_goldvx_harness(
        monkeypatch,
        tmp_path,
        check_dd_side_effect=[False, False, False, False, True],
        time_step=0.5,
    )

    h["goldvx"].main(max_iterations=5)

    # Tick 1 (halted): initial bar boundary triggers fetch (1 M5 + 1 H1 + 1 H4 + 1 ADX = 4 calls).
    # Ticks 2, 3, 4 (still halted, 0.5s apart, same M5 bar): waking_from_observe MUST be False,
    # so get_market_data is NOT called on ticks 2, 3, 4.
    # Tick 5 (check_daily_drawdown -> True): waking_from_observe becomes True and immediately refreshes M5!
    # Total calls: 4 on tick 1 + 1 on tick 5 (since rates[0]['time'] == last_m5_time, light df_m5 refresh) = 5 calls.
    assert h["get_market_data"].call_count == 5


def test_drawdown_reset_resumes_trade_execution(monkeypatch, tmp_path):
    payload = {
        "signal": "BUY",
        "sl": 2674.0,
        "tp_price": 2692.0,
        "confidence": 0.88,
        "comment": "OB RECOVERY",
        "features": [0.5],
    }
    h = _setup_goldvx_harness(
        monkeypatch,
        tmp_path,
        check_dd_side_effect=[False, True],
        strategy_payload=payload,
        time_step=2.0,
    )

    h["goldvx"].main(max_iterations=2)

    assert len(h["bot_state_updates"]) == 2
    assert h["bot_state_updates"][0]["drawdown_halted"] is True
    assert h["bot_state_updates"][0]["observe_only"] is True
    assert "[DD-HALT]" in h["bot_state_updates"][0]["regime"]

    assert h["bot_state_updates"][1]["drawdown_halted"] is False
    assert h["bot_state_updates"][1]["observe_only"] is False
    assert "[DD-HALT]" not in h["bot_state_updates"][1]["regime"]

    # First iteration: observe_only=True, no trade; second iteration: observe_only=False, trade executes
    assert h["strategies"]["SMC_OB"].calls[0]["observe_only"] is True
    assert h["strategies"]["SMC_OB"].calls[1]["observe_only"] is False
    assert h["execute_trade"].call_count == 1
    assert h["execute_trade"].call_args.kwargs["signal"] == "BUY"


def test_real_check_daily_drawdown_liquidation_reconciliation_and_d1_reset(monkeypatch, tmp_path):
    import engine.mt5_interface as mt5_if
    from core.state_manager import StateManager

    state_mgr = StateManager(
        state_file=str(tmp_path / "nexus_state.json"),
        lock_file=str(tmp_path / "nexus_state.lock"),
    )
    monkeypatch.setattr(mt5_if, "nexus_state", state_mgr)
    if hasattr(mt5_if.check_daily_drawdown, "_startup_synced"):
        delattr(mt5_if.check_daily_drawdown, "_startup_synced")

    close_mock = MagicMock()
    monkeypatch.setattr(mt5_if, "close_all_positions", close_mock)

    fake_mt5 = MagicMock()
    fake_mt5.TIMEFRAME_D1 = 16408
    fake_mt5.DEAL_ENTRY_OUT = 1
    monkeypatch.setattr(mt5_if, "mt5", fake_mt5)

    # 1. Startup on Day 1: equity $10,000, 0 loss -> True
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=10000.0, login=111111)
    fake_mt5.copy_rates_from_pos.return_value = [{"time": 1700000000}]
    fake_mt5.history_deals_get.return_value = []
    fake_mt5.positions_get.return_value = []
    assert mt5_if.check_daily_drawdown() is True
    assert state_mgr.get("is_halted") is False

    # 2. Breach daily loss (-$1,500 on $10,000 = 15% >= 4.5% prop-firm limit) -> DAILY_LIMIT_REACHED, False
    fake_mt5.history_deals_get.return_value = [
        types.SimpleNamespace(
            magic=mt5_if.MAGIC_NUMBER, entry=1, profit=-1500.0, commission=0.0, swap=0.0, fee=0.0
        )
    ]
    assert mt5_if.check_daily_drawdown() is False
    assert state_mgr.get("is_halted") is True
    close_mock.assert_called_once_with(reason="DAILY_LIMIT_REACHED")

    # 3. Subsequent tick while halted with a surviving position -> HALT_RECONCILIATION, False
    close_mock.reset_mock()
    fake_mt5.positions_get.return_value = [
        types.SimpleNamespace(symbol=mt5_if.SYMBOL, magic=mt5_if.MAGIC_NUMBER, profit=0.0)
    ]
    assert mt5_if.check_daily_drawdown() is False
    close_mock.assert_called_once_with(reason="HALT_RECONCILIATION")

    # 4. Subsequent tick while halted with 0 surviving positions -> stays False, no extra close
    close_mock.reset_mock()
    fake_mt5.positions_get.return_value = []
    assert mt5_if.check_daily_drawdown() is False
    close_mock.assert_not_called()

    # 5. Next broker D1 candle -> resets is_halted and returns True
    fake_mt5.copy_rates_from_pos.return_value = [{"time": 1700086400}]
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=8500.0, login=111111)
    fake_mt5.history_deals_get.return_value = []
    assert mt5_if.check_daily_drawdown() is True
    assert state_mgr.get("is_halted") is False

    # 6. Intraday breach followed by recovery below active 4.5% limit -> unhalts
    fake_mt5.history_deals_get.return_value = [
        types.SimpleNamespace(
            magic=mt5_if.MAGIC_NUMBER, entry=1, profit=-1300.0, commission=0.0, swap=0.0, fee=0.0
        )
    ]
    assert mt5_if.check_daily_drawdown() is False
    assert state_mgr.get("is_halted") is True

    fake_mt5.history_deals_get.return_value = [
        types.SimpleNamespace(
            magic=mt5_if.MAGIC_NUMBER, entry=1, profit=-200.0, commission=0.0, swap=0.0, fee=0.0
        )
    ]
    assert mt5_if.check_daily_drawdown() is True
    assert state_mgr.get("is_halted") is False


def test_goldvx_with_real_check_daily_drawdown_enforces_r1_and_r2_end_to_end(monkeypatch, tmp_path):
    import engine.mt5_interface as mt5_if
    from core.state_manager import StateManager

    state_mgr = StateManager(
        state_file=str(tmp_path / "nexus_state_e2e.json"),
        lock_file=str(tmp_path / "nexus_state_e2e.lock"),
    )
    state_mgr.set("daily_start_equity", 10000.0)
    state_mgr.set("last_d1_time", 1700000000)
    state_mgr.set("last_account_login", 9990001)
    state_mgr.set("is_halted", False)
    monkeypatch.setattr(mt5_if, "nexus_state", state_mgr)
    if hasattr(mt5_if.check_daily_drawdown, "_startup_synced"):
        delattr(mt5_if.check_daily_drawdown, "_startup_synced")

    close_mock = MagicMock()
    monkeypatch.setattr(mt5_if, "close_all_positions", close_mock)

    payload = {
        "signal": "BUY",
        "sl": 2674.0,
        "tp_price": 2692.0,
        "confidence": 0.90,
        "comment": "E2E OB",
        "features": [0.2],
    }

    h = _setup_goldvx_harness(
        monkeypatch,
        tmp_path,
        check_dd_func=mt5_if.check_daily_drawdown,
        strategy_payload=payload,
        time_step=2.0,
    )
    # Use mt5_if.mt5 tied to the same controllable mock
    monkeypatch.setattr(mt5_if, "mt5", h["mt5"])
    h["mt5"].TIMEFRAME_D1 = 16408
    h["mt5"].DEAL_ENTRY_OUT = 1

    tick_idx = {"n": 0}

    def _d1_rates(symbol, tf, start, count):
        if tf == h["mt5"].TIMEFRAME_D1:
            # Tick 1 & 2: Day 1; Tick 3: Day 2 (reset)
            return [{"time": 1700086400 if tick_idx["n"] >= 3 else 1700000000}]
        return [{"time": 1700000100}]

    def _deals_get(*args, **kwargs):
        # Day 2 (from_ts >= 1700086400) has no closed deals yet
        if args and args[0] >= 1700086400:
            return []
        # Day 1 has -$1,500 closed loss (15% DD >= 4.5% prop-firm limit)
        return [
            types.SimpleNamespace(
                magic=mt5_if.MAGIC_NUMBER, entry=1, profit=-1500.0, commission=0.0, swap=0.0, fee=0.0
            )
        ]

    def _positions_get(*args, **kwargs):
        # On tick 2 reconciliation, simulate 1 surviving position
        if tick_idx["n"] == 2 and not kwargs:
            return [types.SimpleNamespace(symbol=mt5_if.SYMBOL, magic=mt5_if.MAGIC_NUMBER, profit=0.0)]
        return []

    orig_check = mt5_if.check_daily_drawdown

    def _stepped_check():
        tick_idx["n"] += 1
        return orig_check()

    h["dd_mock"].side_effect = _stepped_check
    h["mt5"].copy_rates_from_pos.side_effect = _d1_rates
    h["mt5"].history_deals_get.side_effect = _deals_get
    h["mt5"].positions_get.side_effect = _positions_get

    h["goldvx"].main(max_iterations=3)

    assert close_mock.call_args_list[0].kwargs["reason"] == "DAILY_LIMIT_REACHED"
    assert close_mock.call_args_list[1].kwargs["reason"] == "HALT_RECONCILIATION"
    assert h["get_market_data"].call_count >= 3
    assert h["bot_state_updates"][0]["drawdown_halted"] is True
    assert h["bot_state_updates"][1]["drawdown_halted"] is True
    assert h["bot_state_updates"][2]["drawdown_halted"] is False
    assert h["execute_trade"].call_count == 1


def test_strategies_observe_only_no_state_mutation_or_locks(monkeypatch, capsys):
    from strategies.smc_orderblock import SMCOrderBlockStrategy
    from strategies.rsi_reversion import RSIReversionStrategy
    from strategies.bb_breakout import BBBreakoutStrategy
    from strategies.smc_choch import SMCChochStrategy
    from strategies.vwap_reversion import VWAPReversionStrategy

    df_h1 = _make_ohlcv_df(80, base_price=2600.0, step=0.5, freq_minutes=60)
    df_h4 = _make_ohlcv_df(80, base_price=2580.0, step=0.6, freq_minutes=240)
    df_adx = _make_ohlcv_df(60, base_price=2620.0, step=0.3, freq_minutes=15)

    # 1. SMCOrderBlockStrategy: active BUY OB tapped with rejection candle
    ob_strat = SMCOrderBlockStrategy(ai_model=None)
    ob_strat._warmed_up = True
    ob_strat.ai_throttle_timer = 0
    df_ob = _make_ohlcv_df(60, base_price=2650.0, step=0.05, freq_minutes=5)
    # Force closed candle [-2] to tap OB [2648..2652] and close bullish
    ob_time = df_ob.iloc[-10]["time"]
    ob_strat.active_obs = [{"type": "BUY", "top": 2652.0, "bottom": 2648.0, "poc": 2650.0, "time": ob_time}]
    df_ob.loc[df_ob.index[-2], ["open", "high", "low", "close"]] = [2650.0, 2653.0, 2649.0, 2652.5]
    monkeypatch.setattr(
        "strategies.smc_orderblock.map_market_structure_two_tier",
        lambda *a, **kw: ([], [], {"major_high": None, "major_low": None, "idm_high": None, "idm_low": None}),
    )
    res_ob = ob_strat.evaluate(df_ob, df_h1, df_h4, df_adx, 2651.5, 2.0, 2.5, observe_only=True)
    assert res_ob["payload"] is None
    assert ob_strat.last_traded_ob_time is None
    assert len(ob_strat.active_obs) == 1
    assert "[OB:" in res_ob["ui"]

    # 2. RSIReversionStrategy (Range + Trend modes)
    rsi_strat = RSIReversionStrategy(reversal_model=None, trend_model=None)
    rsi_strat.ai_throttle_timer = 0
    df_rsi_rng = _make_ohlcv_df(60, base_price=2660.0, step=-0.4, freq_minutes=5)
    df_rsi_rng["time"] = pd.date_range("2026-11-17 10:00", periods=60, freq="5min")
    df_rsi_rng["rsi"] = 25.0
    df_rsi_rng["sma_14"] = 2665.0
    df_rsi_rng["ema_9"] = 2640.0
    df_rsi_rng["ema_21"] = 2645.0
    df_rsi_rng["ema_50"] = 2650.0
    df_rsi_rng.loc[df_rsi_rng.index[-2], ["open", "high", "low", "close"]] = [2636.0, 2639.0, 2632.0, 2638.5]
    df_adx_rng = df_adx.copy()
    df_adx_rng["adx"] = 15.0
    res_rsi_rng = rsi_strat.evaluate(df_rsi_rng, df_h1, df_h4, df_adx_rng, 2638.5, 2.0, 2.5, observe_only=True)
    assert res_rsi_rng["payload"] is None
    assert rsi_strat.last_rsi_signal_time is None
    assert rsi_strat.ai_throttle_timer == 0

    df_rsi_trd = _make_ohlcv_df(60, base_price=2640.0, step=0.4, freq_minutes=5)
    df_rsi_trd["time"] = pd.date_range("2026-11-17 10:00", periods=60, freq="5min")
    df_rsi_trd["ema_21"] = df_rsi_trd["close"] - 0.5
    df_rsi_trd["ema_50"] = df_rsi_trd["close"] - 5.0
    df_rsi_trd["ema_9"] = df_rsi_trd["close"] - 0.2
    df_rsi_trd["sma_14"] = df_rsi_trd["close"] - 0.3
    df_rsi_trd["rsi"] = 45.0
    df_rsi_trd.loc[df_rsi_trd.index[-1], "rsi"] = 48.0
    df_rsi_trd.loc[df_rsi_trd.index[-2], "low"] = float(df_rsi_trd["ema_21"].iloc[-2]) - 0.2
    df_adx_trd = df_adx.copy()
    df_adx_trd["adx"] = 28.0
    res_rsi_trd = rsi_strat.evaluate(df_rsi_trd, df_h1, df_h4, df_adx_trd, float(df_rsi_trd["close"].iloc[-1]), 2.0, 2.5, observe_only=True)
    assert res_rsi_trd["payload"] is None
    assert rsi_strat.last_rsi_signal_time is None

    # 3. BBBreakoutStrategy
    bb_strat = BBBreakoutStrategy(ai_model=None)
    df_bb = _make_ohlcv_df(40, base_price=2650.0, step=0.1, freq_minutes=5)
    df_bb["time"] = pd.date_range("2026-11-17 10:00", periods=40, freq="5min")
    df_bb["BBU"] = 2655.0
    df_bb["BBM"] = 2652.0
    df_bb["BBL"] = 2649.0
    df_bb["BandWidth"] = 0.002
    df_bb["BW_Percentile"] = 0.10
    df_bb["rsi"] = 60.0
    df_bb.loc[df_bb.index[-1], ["open", "high", "low", "close", "BandWidth", "BBU"]] = [2655.0, 2657.6, 2654.8, 2657.5, 0.005, 2656.0]
    df_adx_bb = df_adx.copy()
    df_adx_bb["adx"] = 26.0
    res_bb = bb_strat.evaluate(df_bb, df_h1, df_h4, df_adx_bb, 2657.5, 2.0, 2.0, observe_only=True)
    assert res_bb["payload"] is None
    assert bb_strat.last_signal_time is None
    assert bb_strat.throttle_timer == 0

    # 4. SMCChochStrategy: must not call nexus_state.set or mutate choch break state
    import core.state_manager as sm_mod
    fake_nexus = MagicMock()
    fake_nexus.get.return_value = {}
    monkeypatch.setattr(sm_mod, "nexus_state", fake_nexus)
    choch_strat = SMCChochStrategy(ai_model=None)
    df_choch = _make_ohlcv_df(60, base_price=2660.0, step=-0.5, freq_minutes=5)
    df_choch.loc[df_choch.index[-1], "close"] = 2610.0
    monkeypatch.setattr(
        "strategies.smc_choch.map_market_structure_two_tier",
        lambda *a, **kw: (
            [{"type": "HIGH", "price": 2665.0, "idx": 20}, {"type": "LOW", "price": 2630.0, "idx": 10}],
            [{"type": "HIGH", "price": 2660.0, "idx": 5}, {"type": "HIGH", "price": 2665.0, "idx": 20}, {"type": "LOW", "price": 2630.0, "idx": 10}],
            {
                "major_high": {"price": 2665.0, "idx": 20},
                "major_low": {"price": 2630.0, "idx": 10},
                "major_highs": [{"price": 2665.0, "idx": 20}],
                "major_lows": [{"price": 2630.0, "idx": 10}],
                "minor_highs": [{"price": 2660.0, "idx": 5}, {"price": 2665.0, "idx": 20}],
                "minor_lows": [{"price": 2630.0, "idx": 10}],
                "idm_high": None,
                "idm_low": None,
            },
        ),
    )
    res_choch = choch_strat.evaluate(df_choch, df_h1, df_h4, df_adx, 2610.0, 2.0, 2.5, observe_only=True)
    assert res_choch["payload"] is None
    assert choch_strat.last_traded_choch_time is None
    assert choch_strat.last_traded_break_price == {}
    fake_nexus.set.assert_not_called()

    # 5. VWAPReversionStrategy: must not write vwap_* locks or mutate last_signal_time
    import strategies.vwap_reversion as vwap_mod
    vwap_nexus = MagicMock()
    vwap_nexus.get.return_value = True  # Even with existing lock & price normalized, observe_only must not write
    monkeypatch.setattr(vwap_mod, "nexus_state", vwap_nexus)
    vwap_strat = VWAPReversionStrategy()
    df_vwap = _make_ohlcv_df(110, base_price=2650.0, step=0.05, freq_minutes=5)
    res_vwap = vwap_strat.evaluate(df_vwap, df_h1, df_h4, df_adx, 2680.0, 2.0, 2.5, observe_only=True)
    assert res_vwap["payload"] is None
    assert vwap_strat.last_signal_time is None
    assert vwap_strat.throttle_timer == 0
    vwap_nexus.set.assert_not_called()

    captured = capsys.readouterr().out
    assert "[RSI-RANGE" not in captured
    assert "[RSI-TREND" not in captured
    assert "[BB BUY" not in captured
    assert "[BB SELL" not in captured
    assert "CHOCH CONFIRMED" not in captured
    assert "EXTREME VOLUMETRIC" not in captured


# =====================================================================
# R1 / R2 / R3 Comprehensive Verification Tests
# =====================================================================


def test_dd_halt_cancels_pending_orders_and_blocks_execute_trade_directly(monkeypatch, tmp_path):
    import engine.mt5_interface as mt5_if
    from core.state_manager import StateManager

    state_mgr = StateManager(
        state_file=str(tmp_path / "nexus_state_orders.json"),
        lock_file=str(tmp_path / "nexus_state_orders.lock"),
    )
    state_mgr.set("daily_start_equity", 10000.0)
    state_mgr.set("last_d1_time", 1700000000)
    state_mgr.set("last_account_login", 111111)
    state_mgr.set("is_halted", False)
    monkeypatch.setattr(mt5_if, "nexus_state", state_mgr)
    monkeypatch.setattr(mt5_if, "DRY_RUN", False)
    if hasattr(mt5_if.check_daily_drawdown, "_startup_synced"):
        delattr(mt5_if.check_daily_drawdown, "_startup_synced")

    close_mock = MagicMock()
    monkeypatch.setattr(mt5_if, "close_all_positions", close_mock)

    fake_mt5 = MagicMock()
    fake_mt5.TIMEFRAME_D1 = 16408
    fake_mt5.DEAL_ENTRY_OUT = 1
    fake_mt5.TRADE_ACTION_REMOVE = 8
    fake_mt5.TRADE_RETCODE_DONE = 10009
    fake_mt5.TRADE_RETCODE_PLACED = 10008
    fake_mt5.ORDER_TYPE_BUY_LIMIT = 2
    fake_mt5.ORDER_TYPE_SELL_STOP = 5
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=10000.0, balance=10000.0, login=111111)
    fake_mt5.copy_rates_from_pos.return_value = [{"time": 1700000000}]
    fake_mt5.history_deals_get.return_value = [
        types.SimpleNamespace(
            magic=mt5_if.MAGIC_NUMBER, entry=1, profit=-500.0, commission=0.0, swap=0.0, fee=0.0
        )
    ]
    fake_mt5.positions_get.return_value = []
    fake_mt5.order_send.return_value = types.SimpleNamespace(retcode=10009)

    bot_limit = types.SimpleNamespace(ticket=101, magic=mt5_if.MAGIC_NUMBER, symbol=mt5_if.SYMBOL, type=2, price_open=2640.0)
    bot_stop = types.SimpleNamespace(ticket=102, magic=mt5_if.MAGIC_NUMBER, symbol=mt5_if.SYMBOL, type=5, price_open=2660.0)
    other_order = types.SimpleNamespace(ticket=999, magic=123456, symbol=mt5_if.SYMBOL, type=2, price_open=2630.0)
    fake_mt5.orders_get.return_value = [bot_limit, bot_stop, other_order]
    mt5_if._pending_order_birth[101] = 1700000000
    mt5_if._pending_order_birth[102] = 1700000000

    monkeypatch.setattr(mt5_if, "mt5", fake_mt5)

    # 1. Breach triggers DAILY_LIMIT_REACHED and cancels only bot's pending orders
    assert mt5_if.check_daily_drawdown() is False
    assert state_mgr.get("is_halted") is True
    close_mock.assert_called_once_with(reason="DAILY_LIMIT_REACHED")
    cancelled_tickets = [c.args[0]["order"] for c in fake_mt5.order_send.call_args_list]
    assert cancelled_tickets == [101, 102]
    assert 101 not in mt5_if._pending_order_birth
    assert 102 not in mt5_if._pending_order_birth

    # 2. Surviving pending order on next halted tick triggers HALT_RECONCILIATION cancellation
    fake_mt5.order_send.reset_mock()
    fake_mt5.orders_get.return_value = [bot_limit, other_order]
    assert mt5_if.check_daily_drawdown() is False
    assert [c.args[0]["order"] for c in fake_mt5.order_send.call_args_list] == [101]

    # 3. manage_pending_orders() immediately cancels matching pending orders while halted
    fake_mt5.order_send.reset_mock()
    fake_mt5.orders_get.return_value = [bot_stop, other_order]
    mt5_if.manage_pending_orders()
    assert [c.args[0]["order"] for c in fake_mt5.order_send.call_args_list] == [102]

    # 4. Direct execute_trade() call while is_halted=True is rejected immediately
    fake_mt5.order_send.reset_mock()
    fake_mt5.positions_get.reset_mock()
    assert mt5_if.execute_trade("BUY", 2640.0, 2.0, comment_text="DIRECT_CALL") is False
    fake_mt5.order_send.assert_not_called()
    fake_mt5.positions_get.assert_not_called()


def test_dd_halt_hud_tag_stacking_and_transition_logging(monkeypatch, tmp_path):
    # 23:20 UTC -> rollover active + news blackout active + DD halt active
    h = _setup_goldvx_harness(
        monkeypatch,
        tmp_path,
        check_dd_side_effect=[False, False, True, True],
        fixed_utc_now=datetime(2026, 11, 17, 23, 20, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(h["goldvx"], "is_news_blackout", MagicMock(return_value=True))
    monkeypatch.setattr(
        h["goldvx"],
        "get_active_blackout",
        MagicMock(
            return_value={
                "title": "FOMC",
                "currency": "USD",
                "end": datetime(2026, 11, 17, 23, 45, 0, tzinfo=timezone.utc),
            }
        ),
    )

    h["goldvx"].main(max_iterations=4)

    out = "".join(h["hud_writes"])
    assert out.count("Daily Drawdown Protection Active") == 1
    assert out.count("Daily Drawdown Protection Cleared") == 1
    assert "[DD-HALT] [NEWS-HOLD->23:45 UTC]" in h["bot_state_updates"][0]["regime"]
    assert "[DD-HALT]" not in h["bot_state_updates"][2]["regime"]
    assert "[NEWS-HOLD->23:45 UTC]" in h["bot_state_updates"][2]["regime"]

    # Check [DD-HALT] + [WARMUP-ROLL] stacking (no news)
    h2 = _setup_goldvx_harness(
        monkeypatch,
        tmp_path / "roll",
        check_dd_side_effect=[False],
        fixed_utc_now=datetime(2026, 11, 17, 23, 20, 0, tzinfo=timezone.utc),
    )
    h2["goldvx"].main(max_iterations=1)
    assert "[DD-HALT] [WARMUP-ROLL]" in h2["bot_state_updates"][0]["regime"]

    # Check [DD-HALT] + [WARMUP] stacking (off-hours 21:00 UTC)
    h3 = _setup_goldvx_harness(
        monkeypatch,
        tmp_path / "off",
        check_dd_side_effect=[False],
        fixed_utc_now=datetime(2026, 11, 17, 21, 0, 0, tzinfo=timezone.utc),
    )
    h3["goldvx"].main(max_iterations=1)
    assert "[DD-HALT] [WARMUP]" in h3["bot_state_updates"][0]["regime"]


def test_dd_halt_recovery_during_secondary_observe_condition_stays_observe_only(monkeypatch, tmp_path):
    payload = {
        "signal": "BUY",
        "sl": 2674.0,
        "tp_price": 2692.0,
        "confidence": 0.90,
        "comment": "ROLLOVER_RECOVERY",
        "features": [0.1],
    }
    # 23:20 UTC -> rollover stays True across both ticks while DD halt clears on tick 2
    h = _setup_goldvx_harness(
        monkeypatch,
        tmp_path,
        check_dd_side_effect=[False, True],
        strategy_payload=payload,
        time_step=0.5,
        fixed_utc_now=datetime(2026, 11, 17, 23, 20, 0, tzinfo=timezone.utc),
    )

    h["goldvx"].main(max_iterations=2)

    assert len(h["bot_state_updates"]) == 2
    assert h["bot_state_updates"][1]["drawdown_halted"] is False
    assert h["bot_state_updates"][1]["observe_only"] is True
    assert "[WARMUP-ROLL]" in h["bot_state_updates"][1]["regime"]
    # Tick 1 fetched 4 TFs; tick 2 (0.5s later, same bar) must NOT wake_from_observe because rollover is active
    assert h["get_market_data"].call_count == 4
    assert h["execute_trade"].call_count == 0


def test_check_daily_drawdown_exception_is_caught_and_skips_trading(monkeypatch, tmp_path):
    payload = {
        "signal": "BUY",
        "sl": 2674.0,
        "tp_price": 2692.0,
        "confidence": 0.90,
        "comment": "SHOULD_NOT_RUN",
        "features": [0.1],
    }
    h = _setup_goldvx_harness(
        monkeypatch,
        tmp_path,
        check_dd_func=MagicMock(side_effect=RuntimeError("MT5 IPC Timeout")),
        strategy_payload=payload,
    )

    h["goldvx"].main(max_iterations=2)

    out = "".join(h["hud_writes"])
    assert "Loop Error: MT5 IPC Timeout" in out
    assert h["sleep_calls"] == [2, 4]
    assert h["execute_trade"].call_count == 0
    assert all(len(s.calls) == 0 for s in h["strategies"].values())


def test_observe_only_sleep_cadence_with_and_without_open_positions(monkeypatch, tmp_path):
    h_no_pos = _setup_goldvx_harness(
        monkeypatch,
        tmp_path / "no_pos",
        check_dd_side_effect=[False, False],
        positions_total_val=0,
    )
    h_no_pos["goldvx"].main(max_iterations=2)
    assert h_no_pos["sleep_calls"] == [2.0, 2.0]

    h_with_pos = _setup_goldvx_harness(
        monkeypatch,
        tmp_path / "with_pos",
        check_dd_side_effect=[False, False],
        positions_total_val=1,
    )
    h_with_pos["goldvx"].main(max_iterations=2)
    assert h_with_pos["sleep_calls"] == [0.5, 0.5]


def test_telegram_bot_state_defaults_and_status_command(monkeypatch):
    import engine.telegram_notifier as tg

    # Default keys exist in _bot_state
    assert "drawdown_halted" in tg._bot_state
    assert "observe_only" in tg._bot_state

    orig_state = tg._bot_state.copy()
    try:
        tg.update_bot_state(
            regime="[RANGE] [DD-HALT]",
            adx=18.5,
            rsi=45.0,
            price=2650.0,
            trend="FLAT",
            engines="RSI_REVERSION",
            paused=False,
            drawdown_halted=True,
            observe_only=True,
        )
        status_halt = tg._handle_command("/status")
        assert "DD-HALT" in status_halt
        assert "[RANGE] [DD-HALT]" in status_halt

        tg.update_bot_state(drawdown_halted=False, observe_only=True, regime="[RANGE] [WARMUP]")
        status_obs = tg._handle_command("/status")
        assert "OBSERVE ONLY" in status_obs

        tg.update_bot_state(drawdown_halted=False, observe_only=False, regime="[RANGE]")
        status_act = tg._handle_command("/status")
        assert "ACTIVE" in status_act
    finally:
        with tg._bot_state_lock:
            tg._bot_state.clear()
            tg._bot_state.update(orig_state)


@pytest.mark.parametrize(
    "equity,expected",
    [
        (0, 15.0),
        (0.01, 15.0),
        (100.0, 15.0),
        (499.99, 15.0),
        (-100.0, 15.0),
        (500.0, 10.0),
        (500.01, 10.0),
        (2500.0, 10.0),
        (4999.0, 10.0),
        (4999.99, 10.0),
        (5000.0, 4.5),
        (5000.01, 4.5),
        (10000.0, 4.5),
        (100000.0, 4.5),
        (float("inf"), 4.5),
        (float("-inf"), 4.5),
        (float("nan"), 4.5),
        (None, 4.5),
        ("abc", 4.5),
        ("499.99", 15.0),
        ("500", 10.0),
        ("5000", 4.5),
        (True, 4.5),
        (False, 4.5),
    ],
)
def test_get_daily_loss_limit_pct_tiers_boundaries_and_edge_inputs(equity, expected):
    from decimal import Decimal
    from core.config import get_daily_loss_limit_pct

    assert get_daily_loss_limit_pct(equity) == expected
    assert get_daily_loss_limit_pct(Decimal("499.99")) == 15.0
    assert get_daily_loss_limit_pct(Decimal("499.999999999999999999")) == 15.0
    assert get_daily_loss_limit_pct(Decimal("500.00")) == 10.0
    assert get_daily_loss_limit_pct(Decimal("4999.999999999999999999")) == 10.0
    assert get_daily_loss_limit_pct(Decimal("5000.00")) == 4.5
    assert get_daily_loss_limit_pct(Decimal("NaN")) == 4.5
    assert get_daily_loss_limit_pct(np.bool_(True)) == 4.5
    assert get_daily_loss_limit_pct(np.bool_(False)) == 4.5


@pytest.mark.parametrize(
    "start_equity,safe_pnl,breach_pnl,recover_pnl",
    [
        (499.99, -74.99, -75.00, -74.50),    # < $500 -> 15.0%
        (500.0, -49.99, -50.00, -49.50),      # $500-$4,999 -> 10.0%
        (4999.99, -499.99, -500.00, -490.00), # $4,999.99 -> 10.0%
        (5000.0, -224.99, -225.00, -220.00),  # >= $5,000 -> 4.5%
        (10000.0, -449.99, -450.00, -440.00), # $10,000 -> 4.5%
    ],
)
def test_check_daily_drawdown_enforces_each_tier_threshold(
    monkeypatch, tmp_path, start_equity, safe_pnl, breach_pnl, recover_pnl
):
    import engine.mt5_interface as mt5_if
    from core.state_manager import StateManager

    state_mgr = StateManager(
        state_file=str(tmp_path / "nexus_tier.json"),
        lock_file=str(tmp_path / "nexus_tier.lock"),
    )
    monkeypatch.setattr(mt5_if, "nexus_state", state_mgr)
    if hasattr(mt5_if.check_daily_drawdown, "_startup_synced"):
        delattr(mt5_if.check_daily_drawdown, "_startup_synced")
    close_mock = MagicMock()
    monkeypatch.setattr(mt5_if, "close_all_positions", close_mock)
    monkeypatch.setattr(mt5_if, "cancel_all_pending_orders", MagicMock(return_value=0))

    fake_mt5 = MagicMock()
    fake_mt5.TIMEFRAME_D1 = 16408
    fake_mt5.DEAL_ENTRY_OUT = 1
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=start_equity, login=777)
    fake_mt5.copy_rates_from_pos.return_value = [{"time": 1700000000}]
    fake_mt5.positions_get.return_value = []
    fake_mt5.orders_get.return_value = []
    monkeypatch.setattr(mt5_if, "mt5", fake_mt5)

    # Just below limit -> passes
    fake_mt5.history_deals_get.return_value = [
        types.SimpleNamespace(magic=mt5_if.MAGIC_NUMBER, entry=1, profit=safe_pnl, commission=0.0, swap=0.0, fee=0.0)
    ]
    assert mt5_if.check_daily_drawdown() is True
    assert state_mgr.get("is_halted") is False

    # At exact breach threshold -> halts
    fake_mt5.history_deals_get.return_value = [
        types.SimpleNamespace(magic=mt5_if.MAGIC_NUMBER, entry=1, profit=breach_pnl, commission=0.0, swap=0.0, fee=0.0)
    ]
    assert mt5_if.check_daily_drawdown() is False
    assert state_mgr.get("is_halted") is True
    close_mock.assert_called_once_with(reason="DAILY_LIMIT_REACHED")

    # Still at breach threshold -> remains halted
    assert mt5_if.check_daily_drawdown() is False
    assert state_mgr.get("is_halted") is True

    # Recovers below tier threshold -> clears halt
    fake_mt5.history_deals_get.return_value = [
        types.SimpleNamespace(magic=mt5_if.MAGIC_NUMBER, entry=1, profit=recover_pnl, commission=0.0, swap=0.0, fee=0.0)
    ]
    assert mt5_if.check_daily_drawdown() is True
    assert state_mgr.get("is_halted") is False


def test_check_daily_drawdown_pnl_aggregation_lifecycle_and_fail_closed(monkeypatch, tmp_path):
    import engine.mt5_interface as mt5_if
    from core.state_manager import StateManager

    state_mgr = StateManager(
        state_file=str(tmp_path / "nexus_life.json"),
        lock_file=str(tmp_path / "nexus_life.lock"),
    )
    monkeypatch.setattr(mt5_if, "nexus_state", state_mgr)
    close_mock = MagicMock()
    monkeypatch.setattr(mt5_if, "close_all_positions", close_mock)
    monkeypatch.setattr(mt5_if, "cancel_all_pending_orders", MagicMock(return_value=0))

    fake_mt5 = MagicMock()
    fake_mt5.TIMEFRAME_D1 = 16408
    fake_mt5.DEAL_ENTRY_OUT = 1
    monkeypatch.setattr(mt5_if, "mt5", fake_mt5)

    # 1. Fresh startup with existing losing closed deals already in MT5 history today -> immediately halts on call 1!
    if hasattr(mt5_if.check_daily_drawdown, "_startup_synced"):
        delattr(mt5_if.check_daily_drawdown, "_startup_synced")
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=10000.0, login=1001)
    fake_mt5.copy_rates_from_pos.return_value = [{"time": 1700000000}]
    fake_mt5.history_deals_get.return_value = [
        types.SimpleNamespace(magic=mt5_if.MAGIC_NUMBER, entry=1, profit=-460.0, commission=0.0, swap=0.0, fee=0.0)
    ]
    fake_mt5.positions_get.return_value = []
    fake_mt5.orders_get.return_value = []
    assert mt5_if.check_daily_drawdown() is False
    assert state_mgr.get("is_halted") is True

    # 2. Combined closed (profit+commission+swap+fee) + floating open PnL, filtering non-matching magic & DEAL_ENTRY_IN
    state_mgr.set("is_halted", False)
    fake_mt5.history_deals_get.return_value = [
        types.SimpleNamespace(magic=mt5_if.MAGIC_NUMBER, entry=1, profit=-200.0, commission=-10.0, swap=-5.0, fee=-5.0),
        types.SimpleNamespace(magic=mt5_if.MAGIC_NUMBER, entry=0, profit=-9999.0, commission=0.0, swap=0.0, fee=0.0),
        types.SimpleNamespace(magic=555555, entry=1, profit=-9999.0, commission=0.0, swap=0.0, fee=0.0),
    ]
    fake_mt5.positions_get.return_value = [
        types.SimpleNamespace(symbol=mt5_if.SYMBOL, magic=mt5_if.MAGIC_NUMBER, profit=-220.0, commission=-5.0, swap=-5.0),
        types.SimpleNamespace(symbol=mt5_if.SYMBOL, magic=555555, profit=-9999.0, commission=0.0, swap=0.0),
    ]
    # Total matching PnL = -220 + -230 = -450 on $10,000 (4.5%) -> halts!
    assert mt5_if.check_daily_drawdown() is False

    # 3. Profit day -> True
    fake_mt5.history_deals_get.return_value = [
        types.SimpleNamespace(magic=mt5_if.MAGIC_NUMBER, entry=1, profit=300.0, commission=-5.0, swap=0.0, fee=0.0)
    ]
    fake_mt5.positions_get.return_value = []
    assert mt5_if.check_daily_drawdown() is True

    # 4. D1 rollover tier change ($10,000 4.5% -> $4,800 10.0%): -8.33% ($400 loss) passes on $4,800 day
    fake_mt5.copy_rates_from_pos.return_value = [{"time": 1700086400}]
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=4800.0, login=1001)
    assert mt5_if.check_daily_drawdown() is True
    fake_mt5.history_deals_get.return_value = [
        types.SimpleNamespace(magic=mt5_if.MAGIC_NUMBER, entry=1, profit=-400.0, commission=0.0, swap=0.0, fee=0.0)
    ]
    assert mt5_if.check_daily_drawdown() is True

    # 5. Account switch ($4,800 login 1001 -> $300 login 2002, 15.0% tier): -$40 (13.33%) passes, -$45 halts
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=300.0, login=2002)
    fake_mt5.history_deals_get.return_value = [
        types.SimpleNamespace(magic=mt5_if.MAGIC_NUMBER, entry=1, profit=-40.0, commission=0.0, swap=0.0, fee=0.0)
    ]
    assert mt5_if.check_daily_drawdown() is True
    fake_mt5.history_deals_get.return_value = [
        types.SimpleNamespace(magic=mt5_if.MAGIC_NUMBER, entry=1, profit=-45.0, commission=0.0, swap=0.0, fee=0.0)
    ]
    assert mt5_if.check_daily_drawdown() is False

    # 6. Corrupt daily_start_equity (None, 0, NaN) falls back cleanly to account.equity; <=0 account.equity fails closed
    state_mgr.set("is_halted", False)
    fake_mt5.history_deals_get.return_value = []
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=1000.0, login=2002)
    for bad_val in (None, 0, -50.0, float("nan"), "corrupt", True):
        state_mgr.set("daily_start_equity", bad_val)
        assert mt5_if.check_daily_drawdown() is True

    state_mgr.set("daily_start_equity", None)
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=0.0, login=2002)
    assert mt5_if.check_daily_drawdown() is False

    # 7. MT5 failure (account_info None, copy_rates_from_pos None/[]) -> fail-closed False
    fake_mt5.account_info.return_value = None
    assert mt5_if.check_daily_drawdown() is False
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=1000.0, login=2002)
    fake_mt5.copy_rates_from_pos.return_value = None
    assert mt5_if.check_daily_drawdown() is False
    fake_mt5.copy_rates_from_pos.return_value = []
    assert mt5_if.check_daily_drawdown() is False


def test_execute_trade_enforces_tiered_daily_loss_cap(monkeypatch, tmp_path):
    import core.config as cfg
    import engine.mt5_interface as mt5_if
    from core.state_manager import StateManager

    state_mgr = StateManager(
        state_file=str(tmp_path / "nexus_exec.json"),
        lock_file=str(tmp_path / "nexus_exec.lock"),
    )
    state_mgr.set("is_halted", False)
    monkeypatch.setattr(mt5_if, "nexus_state", state_mgr)
    monkeypatch.setattr(mt5_if, "DRY_RUN", False)
    monkeypatch.setattr(cfg, "MANUAL_FIXED_LOT", 1.0)
    monkeypatch.setattr(cfg, "MAX_SINGLE_TRADE_RISK_PCT", 5.5)

    fake_mt5 = MagicMock()
    fake_mt5.ORDER_TYPE_BUY = 0
    fake_mt5.ORDER_TYPE_SELL = 1
    fake_mt5.TRADE_ACTION_DEAL = 1
    fake_mt5.positions_get.return_value = []
    fake_mt5.orders_get.return_value = []
    fake_mt5.symbol_info_tick.return_value = types.SimpleNamespace(ask=2650.0, bid=2649.8)
    fake_mt5.symbol_info.return_value = types.SimpleNamespace(
        point=0.01, trade_contract_size=100.0, volume_min=0.01, volume_max=100.0, volume_step=0.01
    )
    # $10,000 prop-firm account (4.5% daily limit): 1.0 lot * 6.0 SL * 100 = $600 = 6.0% > 4.5% -> REJECTED!
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=10000.0, balance=10000.0, login=999)
    monkeypatch.setattr(mt5_if, "mt5", fake_mt5)
    monkeypatch.setattr(mt5_if, "calculate_position_size", lambda *a, **kw: 1.0)

    assert mt5_if.execute_trade("BUY", 2644.0, 2.0, comment_text="TIER_CAP_TEST", ai_conf=0.80) is False
    fake_mt5.order_send.assert_not_called()


def _make_liq_sweep_df(mode="bsl_confirmed"):
    """Build a deterministic 115-bar M5 DF containing clean EQH/EQL fractal pools and a Bar-0 sweep."""
    n = 115
    start_dt = datetime(2026, 11, 17, 6, 0, 0, tzinfo=timezone.utc)
    times = [start_dt + timedelta(minutes=5 * i) for i in range(n)]

    if mode.startswith("bsl"):
        # Mildly declining price so ema_21 < ema_50 (avoids bullish ribbon veto)
        closes = [2658.0 - 0.04 * i for i in range(n)]
        opens = [c + 0.1 for c in closes]
        highs = [max(o, c) + 0.3 for o, c in zip(opens, closes)]
        lows = [min(o, c) - 0.3 for o, c in zip(opens, closes)]
        # Two EQH fractals inside last-100 window (indices -100.. -> df indices 15..114):
        # Put fractal highs at df index 55 and 75 with high=2660.0
        for idx in (55, 75):
            opens[idx] = 2656.0
            closes[idx] = 2656.5
            highs[idx] = 2660.0
            lows[idx] = 2655.0
        if mode == "bsl_confirmed":
            # Closed bar (-2, idx 113): spikes above 2660.0 and closes red below ema_21 with strong upper wick
            opens[-2] = 2658.0
            highs[-2] = 2663.0
            lows[-2] = 2651.0
            closes[-2] = 2651.5
        else:
            # Unconfirmed: spikes above 2660.0 and closes green (fails Bar-0 red check)
            opens[-2] = 2657.0
            highs[-2] = 2661.5
            lows[-2] = 2656.5
            closes[-2] = 2659.5
        opens[-1] = closes[-2]
        closes[-1] = closes[-2]
        highs[-1] = closes[-2] + 0.2
        lows[-1] = closes[-2] - 0.2
    else:
        # Mildly rising price so ema_21 > ema_50 (avoids bearish ribbon veto)
        closes = [2642.0 + 0.04 * i for i in range(n)]
        opens = [c - 0.1 for c in closes]
        highs = [max(o, c) + 0.3 for o, c in zip(opens, closes)]
        lows = [min(o, c) - 0.3 for o, c in zip(opens, closes)]
        # Two EQL fractals at df index 55 and 75 with low=2640.0
        for idx in (55, 75):
            opens[idx] = 2644.0
            closes[idx] = 2643.5
            lows[idx] = 2640.0
            highs[idx] = 2645.0
        if mode == "ssl_confirmed":
            # Closed bar (-2): spikes below 2640.0 and closes green above ema_21 with strong lower wick
            opens[-2] = 2642.0
            lows[-2] = 2637.0
            highs[-2] = 2649.0
            closes[-2] = 2648.5
        else:
            # Unconfirmed: spikes below 2640.0 and closes red (fails Bar-0 green check)
            opens[-2] = 2643.0
            lows[-2] = 2638.5
            highs[-2] = 2643.5
            closes[-2] = 2640.5
        opens[-1] = closes[-2]
        closes[-1] = closes[-2]
        highs[-1] = closes[-2] + 0.2
        lows[-1] = closes[-2] - 0.2

    return pd.DataFrame(
        {
            "time": pd.to_datetime(times),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": [150] * n,
        }
    )


@pytest.mark.parametrize(
    "mode,expected_tag",
    [
        ("bsl_confirmed", "[BSL SWEPT: 2660.00 (OBS)]"),
        ("bsl_unconfirmed", "[BSL SWEPT: 2660.00 (OBS / WAIT RECLAIM)]"),
        ("ssl_confirmed", "[SSL SWEPT: 2640.00 (OBS)]"),
        ("ssl_unconfirmed", "[SSL SWEPT: 2640.00 (OBS / WAIT RECLAIM)]"),
    ],
)
def test_liquidity_sweep_observe_only_no_arming_mutation_or_post_halt_trade(monkeypatch, mode, expected_tag):
    from strategies.liquidity_sweep import LiquiditySweepStrategy

    df_m5 = _make_liq_sweep_df(mode)
    is_bsl = mode.startswith("bsl")
    df_h1 = _make_ohlcv_df(60, base_price=2660.0 if is_bsl else 2640.0, step=-0.2 if is_bsl else 0.2, freq_minutes=60)
    df_h4 = _make_ohlcv_df(60, base_price=2660.0 if is_bsl else 2640.0, step=-0.2 if is_bsl else 0.2, freq_minutes=240)
    df_adx = _make_ohlcv_df(60, base_price=2650.0, step=0.05, freq_minutes=15)
    df_adx["adx"] = 18.0

    strat = LiquiditySweepStrategy(ai_model=None)
    stale_time = pd.Timestamp(df_m5.iloc[-2]["time"]) - pd.Timedelta(hours=30)
    initial_traded_zones = [{"price": 2500.0, "time": stale_time, "direction": "SELL", "reentries": 0}]
    strat.traded_zones = list(initial_traded_zones)

    curr_price = float(df_m5.iloc[-1]["close"])
    res = strat.evaluate(df_m5, df_h1, df_h4, df_adx, curr_price, 2.0, 2.5, ai_mode=None, observe_only=True)

    assert res["payload"] is None
    assert res["ui"] == expected_tag
    assert "(ARMED / WAIT RECLAIM)" not in res["ui"]
    assert strat.armed_sweeps == []
    assert strat.last_traded_sweep_time is None
    assert strat.traded_zones == initial_traded_zones

    # Transition to next M5 bar with observe_only=False: no delayed Phase 1 trade from [DD-HALT] sweep
    next_bar = df_m5.iloc[-1].copy()
    next_bar["time"] = pd.Timestamp(df_m5.iloc[-1]["time"]) + pd.Timedelta(minutes=5)
    next_bar["open"] = curr_price
    next_bar["close"] = curr_price - 0.5 if is_bsl else curr_price + 0.5
    next_bar["high"] = max(next_bar["open"], next_bar["close"]) + 0.3
    next_bar["low"] = min(next_bar["open"], next_bar["close"]) - 0.3
    df_next = pd.concat([df_m5, pd.DataFrame([next_bar])], ignore_index=True)

    res_next = strat.evaluate(df_next, df_h1, df_h4, df_adx, float(next_bar["close"]), 2.0, 2.5, ai_mode=None, observe_only=False)
    assert res_next["payload"] is None


def test_liquidity_sweep_pre_halt_armed_sweep_expires_during_observe_only():
    from strategies.liquidity_sweep import LiquiditySweepStrategy

    df_m5 = _make_liq_sweep_df("bsl_unconfirmed")
    # Remove the sweep on bar -2 so Phase 2 won't re-detect it; keep neutral price inside pool
    df_m5.loc[df_m5.index[-2], ["open", "high", "low", "close"]] = [2655.0, 2655.5, 2654.5, 2655.0]
    df_h1 = _make_ohlcv_df(60, base_price=2660.0, step=-0.2, freq_minutes=60)
    df_h4 = _make_ohlcv_df(60, base_price=2660.0, step=-0.2, freq_minutes=240)
    df_adx = _make_ohlcv_df(60, base_price=2650.0, step=0.05, freq_minutes=15)
    df_adx["adx"] = 18.0

    strat = LiquiditySweepStrategy(ai_model=None)
    strat.last_closed_candle_time = df_m5.iloc[-2]["time"]
    strat.armed_sweeps = [
        {
            "direction": "SELL",
            "zone": 2660.0,
            "sweep_ext": 2662.0,
            "armed_time": df_m5.iloc[-2]["time"],
            "bars_aged": 0,
            "is_double_sweep": False,
            "matching_tz": None,
        }
    ]

    # Step 4 new closed bars while observe_only=True -> bars_aged 1, 2, 3, then >3 pruned to []
    df_curr = df_m5.copy()
    for step_idx in range(1, 5):
        bar = df_curr.iloc[-1].copy()
        bar["time"] = pd.Timestamp(df_curr.iloc[-1]["time"]) + pd.Timedelta(minutes=5)
        bar["open"] = 2655.0
        bar["high"] = 2655.4
        bar["low"] = 2654.6
        bar["close"] = 2655.0
        df_curr = pd.concat([df_curr, pd.DataFrame([bar])], ignore_index=True)

        res = strat.evaluate(df_curr, df_h1, df_h4, df_adx, 2655.0, 2.0, 2.5, ai_mode=None, observe_only=True)
        assert res["payload"] is None
        if step_idx <= 3:
            assert len(strat.armed_sweeps) == 1
            assert strat.armed_sweeps[0]["bars_aged"] == step_idx
        else:
            assert strat.armed_sweeps == []


def test_vwap_reversion_observe_only_false_rr_check_has_no_nameerror(monkeypatch):
    import strategies.vwap_reversion as vwap_mod
    from strategies.vwap_reversion import VWAPReversionStrategy

    fake_nexus = MagicMock()
    fake_nexus.get.return_value = False
    monkeypatch.setattr(vwap_mod, "nexus_state", fake_nexus)
    monkeypatch.setitem(
        sys.modules,
        "engine.feature_extractor_v3",
        types.SimpleNamespace(extract_v3_features=lambda *a, **kw: pd.DataFrame([[0.1] * 10])),
    )

    df_m5 = _make_ohlcv_df(120, base_price=2650.0, step=0.0, freq_minutes=5, oscillate=True)
    df_h1 = _make_ohlcv_df(60, base_price=2660.0, step=-0.2, freq_minutes=60)
    df_adx = _make_ohlcv_df(60, base_price=2660.0, step=-0.2, freq_minutes=15)
    df_adx["adx"] = 18.0

    strat = VWAPReversionStrategy()
    # Price stretched far above VWAP (+2 SD) so SELL signal reaches line 178 R:R check
    res = strat.evaluate(df_m5, df_h1, None, df_adx, 2675.0, 2.0, 2.5, observe_only=False)
    assert res["payload"] is not None
    assert res["payload"]["signal"] == "SELL"


def test_check_daily_drawdown_mt5_none_while_halted_never_unhalts(monkeypatch, tmp_path):
    import engine.mt5_interface as mt5_if
    from core.state_manager import StateManager

    state_mgr = StateManager(
        state_file=str(tmp_path / "nexus_none_halt.json"),
        lock_file=str(tmp_path / "nexus_none_halt.lock"),
    )
    state_mgr.set("daily_start_equity", 10000.0)
    state_mgr.set("last_d1_time", 1700000000)
    state_mgr.set("last_account_login", 111)
    state_mgr.set("is_halted", True)
    monkeypatch.setattr(mt5_if, "nexus_state", state_mgr)
    if hasattr(mt5_if.check_daily_drawdown, "_startup_synced"):
        delattr(mt5_if.check_daily_drawdown, "_startup_synced")

    fake_mt5 = MagicMock()
    fake_mt5.TIMEFRAME_D1 = 16408
    fake_mt5.DEAL_ENTRY_OUT = 1
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=10000.0, login=111)
    fake_mt5.copy_rates_from_pos.return_value = [{"time": 1700000000}]
    monkeypatch.setattr(mt5_if, "mt5", fake_mt5)

    for deals_val, pos_val in [(None, []), ([], None), (None, None)]:
        fake_mt5.history_deals_get.return_value = deals_val
        fake_mt5.positions_get.return_value = pos_val
        assert mt5_if.check_daily_drawdown() is False
        assert state_mgr.get("is_halted") is True


def test_check_daily_drawdown_startup_resume_corrupt_state_and_bad_equity(monkeypatch, tmp_path):
    import engine.mt5_interface as mt5_if
    from core.state_manager import StateManager

    state_mgr = StateManager(
        state_file=str(tmp_path / "nexus_startup_corrupt.json"),
        lock_file=str(tmp_path / "nexus_startup_corrupt.lock"),
    )
    monkeypatch.setattr(mt5_if, "nexus_state", state_mgr)

    fake_mt5 = MagicMock()
    fake_mt5.TIMEFRAME_D1 = 16408
    fake_mt5.DEAL_ENTRY_OUT = 1
    fake_mt5.copy_rates_from_pos.return_value = [{"time": 1700000000}]
    fake_mt5.history_deals_get.return_value = []
    fake_mt5.positions_get.return_value = []
    monkeypatch.setattr(mt5_if, "mt5", fake_mt5)

    # Fresh startup resume (same D1 day, same login) with corrupt daily_start_equity values
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=1000.0, login=111)
    for bad_start in ("corrupt", "bad", None, float("nan"), -50.0, 0, True):
        if hasattr(mt5_if.check_daily_drawdown, "_startup_synced"):
            delattr(mt5_if.check_daily_drawdown, "_startup_synced")
        state_mgr.set("last_d1_time", 1700000000)
        state_mgr.set("last_account_login", 111)
        state_mgr.set("daily_start_equity", bad_start)
        state_mgr.set("is_halted", False)
        assert mt5_if.check_daily_drawdown() is True
        assert state_mgr.get("daily_start_equity") == 1000.0

    # Valid daily_start_equity=1000.0, but account.equity invalid/non-positive -> fail-closed False
    for bad_eq in (0.0, -500.0, float("nan"), None):
        state_mgr.set("daily_start_equity", 1000.0)
        fake_mt5.account_info.return_value = types.SimpleNamespace(equity=bad_eq, login=111)
        assert mt5_if.check_daily_drawdown() is False

    # Fresh startup resume with account.equity=None and daily_start_equity=None -> fail-closed False (no TypeError)
    if hasattr(mt5_if.check_daily_drawdown, "_startup_synced"):
        delattr(mt5_if.check_daily_drawdown, "_startup_synced")
    state_mgr.set("daily_start_equity", None)
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=None, login=111)
    assert mt5_if.check_daily_drawdown() is False


def test_check_daily_drawdown_runtime_d1_rollover_checks_carried_floating_loss(monkeypatch, tmp_path):
    import engine.mt5_interface as mt5_if
    from core.state_manager import StateManager

    state_mgr = StateManager(
        state_file=str(tmp_path / "nexus_rollover_float.json"),
        lock_file=str(tmp_path / "nexus_rollover_float.lock"),
    )
    monkeypatch.setattr(mt5_if, "nexus_state", state_mgr)
    if hasattr(mt5_if.check_daily_drawdown, "_startup_synced"):
        delattr(mt5_if.check_daily_drawdown, "_startup_synced")

    close_mock = MagicMock()
    cancel_mock = MagicMock(return_value=0)
    monkeypatch.setattr(mt5_if, "close_all_positions", close_mock)
    monkeypatch.setattr(mt5_if, "cancel_all_pending_orders", cancel_mock)

    fake_mt5 = MagicMock()
    fake_mt5.TIMEFRAME_D1 = 16408
    fake_mt5.DEAL_ENTRY_OUT = 1
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=5000.0, login=333)
    fake_mt5.copy_rates_from_pos.return_value = [{"time": 1700000000}]
    fake_mt5.history_deals_get.return_value = []
    fake_mt5.positions_get.return_value = []
    fake_mt5.orders_get.return_value = []
    monkeypatch.setattr(mt5_if, "mt5", fake_mt5)

    # Day 1 healthy
    assert mt5_if.check_daily_drawdown() is True
    assert state_mgr.get("is_halted") is False

    # Day 2 rollover carrying a -10% (-$500 on $5,000, >= 4.5% limit) floating open position:
    # Tick 1 on rollover MUST immediately return False and halt!
    fake_mt5.copy_rates_from_pos.return_value = [{"time": 1700086400}]
    fake_mt5.positions_get.return_value = [
        types.SimpleNamespace(symbol=mt5_if.SYMBOL, magic=mt5_if.MAGIC_NUMBER, profit=-500.0, commission=0.0, swap=0.0)
    ]
    assert mt5_if.check_daily_drawdown() is False
    assert state_mgr.get("is_halted") is True
    close_mock.assert_called_once_with(reason="DAILY_LIMIT_REACHED")
    cancel_mock.assert_called_once_with(reason="DAILY_LIMIT_REACHED")


def test_execute_trade_fail_closed_on_missing_or_invalid_account_and_symbol(monkeypatch, tmp_path):
    import core.config as cfg
    import engine.mt5_interface as mt5_if
    from core.state_manager import StateManager

    state_mgr = StateManager(
        state_file=str(tmp_path / "nexus_exec_fc.json"),
        lock_file=str(tmp_path / "nexus_exec_fc.lock"),
    )
    state_mgr.set("is_halted", False)
    monkeypatch.setattr(mt5_if, "nexus_state", state_mgr)
    monkeypatch.setattr(mt5_if, "DRY_RUN", False)
    monkeypatch.setattr(cfg, "MANUAL_FIXED_LOT", 0.10)
    monkeypatch.setattr(mt5_if, "calculate_position_size", lambda *a, **kw: 0.10)

    fake_mt5 = MagicMock()
    fake_mt5.ORDER_TYPE_BUY = 0
    fake_mt5.ORDER_TYPE_SELL = 1
    fake_mt5.TRADE_ACTION_DEAL = 1
    fake_mt5.positions_get.return_value = []
    fake_mt5.orders_get.return_value = []
    fake_mt5.symbol_info_tick.return_value = types.SimpleNamespace(ask=2650.0, bid=2649.8)
    valid_symbol = types.SimpleNamespace(
        point=0.01, trade_contract_size=100.0, volume_min=0.01, volume_max=100.0, volume_step=0.01
    )
    monkeypatch.setattr(mt5_if, "mt5", fake_mt5)

    # 1. equity <= 0 while balance=1000.0 -> rejected
    for bad_eq in (-50.0, 0.0, float("nan")):
        fake_mt5.symbol_info.return_value = valid_symbol
        fake_mt5.account_info.return_value = types.SimpleNamespace(equity=bad_eq, balance=1000.0, login=999)
        assert mt5_if.execute_trade("BUY", 2640.0, 2.0, comment_text="BAD_EQ", ai_conf=0.80) is False

    # 2. balance=NaN, equity=NaN -> rejected
    fake_mt5.symbol_info.return_value = valid_symbol
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=float("nan"), balance=float("nan"), login=999)
    assert mt5_if.execute_trade("BUY", 2640.0, 2.0, comment_text="NAN_BAL", ai_conf=0.80) is False

    # 3. account_info() -> None -> rejected
    fake_mt5.symbol_info.return_value = valid_symbol
    fake_mt5.account_info.return_value = None
    assert mt5_if.execute_trade("BUY", 2640.0, 2.0, comment_text="NONE_ACC", ai_conf=0.80) is False

    # 4. symbol_info() -> None -> rejected
    fake_mt5.symbol_info.return_value = None
    fake_mt5.account_info.return_value = types.SimpleNamespace(equity=1000.0, balance=1000.0, login=999)
    assert mt5_if.execute_trade("BUY", 2640.0, 2.0, comment_text="NONE_SYM", ai_conf=0.80) is False

    fake_mt5.order_send.assert_not_called()


def test_cancel_all_pending_orders_edge_paths_and_telegram_dd_halt_paused(monkeypatch):
    import engine.mt5_interface as mt5_if
    import engine.telegram_notifier as tg

    # 1. DRY_RUN=True returns 0 without calling MT5
    fake_mt5 = MagicMock()
    monkeypatch.setattr(mt5_if, "mt5", fake_mt5)
    monkeypatch.setattr(mt5_if, "DRY_RUN", True)
    assert mt5_if.cancel_all_pending_orders() == 0
    fake_mt5.orders_get.assert_not_called()

    # 2. orders_get() -> None returns 0 cleanly
    monkeypatch.setattr(mt5_if, "DRY_RUN", False)
    fake_mt5.orders_get.return_value = None
    assert mt5_if.cancel_all_pending_orders() == 0
    fake_mt5.order_send.assert_not_called()

    # 3. Partial order_send failure (None and non-DONE retcode) preserves failed tickets
    fake_mt5.TRADE_ACTION_REMOVE = 8
    fake_mt5.TRADE_RETCODE_DONE = 10009
    fake_mt5.TRADE_RETCODE_PLACED = 10008
    o1 = types.SimpleNamespace(ticket=201, magic=mt5_if.MAGIC_NUMBER)
    o2 = types.SimpleNamespace(ticket=202, magic=mt5_if.MAGIC_NUMBER)
    o3 = types.SimpleNamespace(ticket=203, magic=mt5_if.MAGIC_NUMBER)
    fake_mt5.orders_get.return_value = [o1, o2, o3]
    fake_mt5.order_send.side_effect = [
        None,
        types.SimpleNamespace(retcode=10013),
        types.SimpleNamespace(retcode=10009),
    ]
    mt5_if._pending_order_birth[201] = 1700000000
    mt5_if._pending_order_birth[202] = 1700000000
    mt5_if._pending_order_birth[203] = 1700000000
    cancelled = mt5_if.cancel_all_pending_orders(reason="PARTIAL_TEST")
    assert cancelled == 1
    assert 201 in mt5_if._pending_order_birth
    assert 202 in mt5_if._pending_order_birth
    assert 203 not in mt5_if._pending_order_birth
    mt5_if._pending_order_birth.pop(201, None)
    mt5_if._pending_order_birth.pop(202, None)

    # 4. Telegram /status displays DD-HALT (PAUSED) when both drawdown_halted and paused are True
    orig_state = tg._bot_state.copy()
    try:
        tg.update_bot_state(
            regime="[RANGE] [DD-HALT]",
            adx=18.5,
            rsi=45.0,
            price=2650.0,
            trend="FLAT",
            engines="RSI_REVERSION",
            paused=True,
            drawdown_halted=True,
            observe_only=True,
        )
        status_halt_paused = tg._handle_command("/status")
        assert "DD-HALT (PAUSED)" in status_halt_paused
    finally:
        with tg._bot_state_lock:
            tg._bot_state.clear()
            tg._bot_state.update(orig_state)


def test_liquidity_sweep_none_df_and_backtester_stringio_import(monkeypatch):
    from strategies.liquidity_sweep import LiquiditySweepStrategy

    strat = LiquiditySweepStrategy(ai_model=None)
    assert strat.evaluate(None, None, None, None, 2650.0, 2.0, 2.5, observe_only=True) == {
        "ui": "",
        "payload": None,
    }

    monkeypatch.setattr(sys, "stdout", io.StringIO())
    import src.backtester as bt

    reloaded = importlib.reload(bt)
    assert hasattr(reloaded, "get_daily_loss_limit_pct")
