# Parallax — Quantitative Multi-Timeframe Execution & Risk Engine (`XAU/USD`)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MetaTrader 5](https://img.shields.io/badge/MetaTrader-5-green.svg)](https://www.metatrader5.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Purged%20Walk--Forward%20CV-orange.svg)](https://scikit-learn.org/)
[![Risk Engine](https://img.shields.io/badge/Risk%20Engine-Prop%20Firm%20Compliant-success.svg)](core/config.py)
[![Tests](https://img.shields.io/badge/Tests-57%20Passing-brightgreen.svg)](tests/test_goldvx_drawdown_observation.py)

**Parallax** (built as the core execution engine of **NEXUS Quant Technologies**) is the **public, open-architecture release of a live production XAU/USD algorithmic trading platform that has been in continuous operation and development for over a year**.

Rather than publishing raw indicator scripts, this repository exposes the **institutional engineering layer** honed across live market regimes: deterministic multi-timeframe state synchronization (`M5 / M15 / H1 / H4`), passive market observation during drawdown/macro halts, 3-tier macroeconomic event blackouts, Purged & Embargoed cross-validation, tiered prop-firm drawdown circuit breakers, and thread-safe MetaTrader 5 IPC execution.

> **Public Architecture vs. Private Production Alpha Notice:**  
> All execution guardrails, risk mathematics, regime orchestrator (`src/goldvx.py`), ML feature schema, Purged-CV training pipeline (`src/auto_retrain.py`), and the 57-test automated suite are included here in full. To protect commercial IP of the live product, `strategies/` ships clean **Public Reference Adapters** while tuned production alpha strategies, client operational logs, and trained `.joblib` weights remain in the private production repository.

---

## System Architecture

Parallax continuously ingests tick and multi-timeframe OHLCV telemetry, validates daily drawdown and macroeconomic news gates, checks structural displacement velocity, ranks eligible engines dynamically by live performance, and serializes orders cleanly through `mt5_lock`.

```mermaid
flowchart TD
    Tick["Tick Telemetry (XAUUSD M5 / M15 / H1 / H4)"] --> DDCheck{"Tiered Daily Drawdown Breached?"}
    DDCheck -- "Yes (DD-HALT)" --> PassiveObs["Passive Observation Mode (observe_only=True)"]
    DDCheck -- "No (Healthy)" --> News{"Macro News Blackout Active?"}

    News -- "Yes (In Window)" --> PostWhip{"Guard 3: News Whipsaw Detected? (Range >= $10 & Body < 45%)"}
    PostWhip -- "Yes" --> ExtendNews["Extend Blackout to 45m (Anti-Whipsaw)"]
    PostWhip -- "No" --> PassiveObs

    PassiveObs --> RegimeRouter["Evaluate Session & M15 ADX Regime (HUD / Telemetry Only)"]

    News -- "No (Clear to Trade)" --> VolGuard{"Volatility Guard ATR Ratio > 4.5x?"}
    VolGuard -- "Yes" --> PruneRev["Prune Mean-Reversion Engines (High-Risk Velocity)"]
    VolGuard -- "No" --> DispCheck{"Displacement Velocity Override? (2-bar >= 2.0x ATR or 3-bar >= 2.5x ATR)"}

    DispCheck -- "Yes" --> ForceTrend["Fast-Track to TREND-VEL Regime"]
    DispCheck -- "No" --> RegimeRouter

    RegimeRouter --> RegimeActive{"Active Regime"}
    RegimeActive -- "TREND (ADX > 30)" --> StratTrend["SMC_OB | LIQUIDITY_SWEEP | BB_BREAKOUT | RSI_REVERSION"]
    RegimeActive -- "RANGE (ADX < 20)" --> StratRange["RSI_REVERSION | LIQUIDITY_SWEEP"]
    RegimeActive -- "ASIAN-MATRIX" --> StratAsia["RSI_REVERSION | LIQUIDITY_SWEEP"]
    RegimeActive -- "NEUTRAL (ADX 20-30)" --> StratNeutral["SMC_OB | RSI_REVERSION | LIQUIDITY_SWEEP | BB_BREAKOUT"]

    ForceTrend --> StratTrend

    StratTrend --> LeagueRanking["Performance League (Sort by Live Win Rate & Net PnL)"]
    StratRange --> LeagueRanking
    StratAsia --> LeagueRanking
    StratNeutral --> LeagueRanking

    LeagueRanking --> StratEval["Strategy Evaluates Signals (Priority Order)"]
    StratEval --> PostNewsGuard{"Post-News Guard Window (< 45m)?"}
    PostNewsGuard -- "Spread > Limit" --> BlockSpread["BLOCKED: Guard 1 Spread Gate"]
    PostNewsGuard -- "Counter-Trend to News" --> BlockDir["BLOCKED: Guard 2 One-Way Lock"]
    PostNewsGuard -- "Valid" --> AIGate{"Dual ML & Veto Gate Passed?"}

    AIGate -- "Denied" --> Ledger["Log Rejection to Decision Ledger"]
    AIGate -- "Approved" --> RiskEngine["Tiered Risk Sizing & Anti-Chaser Limit Guards"]

    RiskEngine --> ExecuteMT5["Execute Order on MT5 Terminal (mt5_lock)"]
    ExecuteMT5 --> TelemetrySync["Broadcast to Cloud Telemetry, Telegram & Ledger CSV"]
```

---

## Market Regimes & Dynamic Orchestration

The core orchestrator ([`src/goldvx.py`](src/goldvx.py)) classifies market state every iteration across `M5`, `M15`, `H1`, and `H4`, routing execution priority across eligible engines:

| Regime State | Trigger Conditions | Active Engines | Operational Focus |
| :--- | :--- | :--- | :--- |
| **`[TREND]`** | `ADX > 30.0` or sustained HTF alignment | `SMC_OB`, `LIQUIDITY_SWEEP`, `BB_BREAKOUT`, `RSI_REVERSION` | Participates in directional expansions; mean-reversion switches cleanly to trend-pullback confirmation. |
| **`[TREND-VEL]`** | Displacement spike (`>= 2.0x ATR` over 2 bars or `>= 2.5x ATR` over 3 bars) | `SMC_OB`, `LIQUIDITY_SWEEP`, `BB_BREAKOUT` | Bypasses slow ADX lag during institutional displacement impulses. |
| **`[RANGE]`** | `ADX < 20.0` during London / New York | `RSI_REVERSION`, `LIQUIDITY_SWEEP` | Targets bounded range liquidity pools and statistical extremes. |
| **`[ASIAN-MATRIX]`** | Asian Session (`00:00 - 07:00 UTC`) | `RSI_REVERSION`, `LIQUIDITY_SWEEP` | Suppresses low-volume breakout chasing during Asia consolidation. |
| **`[ASIA-BREAKOUT]`** | Asian Session with volatility expansion / velocity | `SMC_OB`, `LIQUIDITY_SWEEP`, `RSI_REVERSION`, `BB_BREAKOUT` | Unlocks expansion engines if Asian volume breaks structure early. |
| **`[ROLLOVER]`** | Broker rollover window (`23:15 - 00:30 UTC`) | *Passive Observation (`observe_only=True`)* | Halts new exposure during illiquid spread widening while maintaining market visibility. |
| **`[DD-HALT]`** | Daily loss `>= Tiered Limit (15% / 10% / 4.5%)` | *Passive Observation (`observe_only=True`)* | Liquidates open exposure, locks execution until daily reset, and keeps live HUD/dashboard telemetry active. |

---

## Pluggable Strategy Adapter Architecture

The framework cleanly decouples execution orchestration from signal research behind [`strategies/base.py`](strategies/base.py), ensuring every signal engine honors `observe_only=True` and passes immutable structural payloads downstream:

```mermaid
flowchart LR
    subgraph S1 ["1. Order Block Engine (smc_orderblock.py)"]
        OB1["Swing Fractal Detection"] --> OB2["Zone & FVG Imbalance Filtering"]
        OB2 --> OB3["Mitigation Evaluation (Directional R:R)"]
    end

    subgraph S2 ["2. RSI Mean Reversion (rsi_reversion.py)"]
        R1["Range Mode Extremes (ADX < 22)"]
        R2["Trend Mode Filter (ADX >= 22)"]
        R3["NaN & Candle Body Safety Guards"]
        R1 --- R3
        R2 --- R3
    end

    subgraph S3 ["3. Liquidity Sweep Engine (liquidity_sweep.py)"]
        LQ1["Equal Highs / Lows (BSL / SSL Tracking)"] --> LQ2["Wick Penetration & Reclaim Check"]
        LQ2 --> LQ3["Strict No-Mutation under observe_only"]
    end

    subgraph S4 ["4. Volatility Breakout (bb_breakout.py)"]
        BB1["20-Period Bollinger Envelope"] --> BB2["Band Breakout & ADX Filter"]
        BB2 --> BB3["ATR-Buffered Stop & Target Sizing"]
    end
```

*(Note: Files in `strategies/` contain clean **Public Reference Adapters** demonstrating interfaces and `observe_only` safety guarantees; proprietary production alpha parameters are withheld.)*

---

## Tiered Macro News Filter & Post-News Safety Guards

Trading unguarded across Tier-1 releases (`CPI`, `NFP`, `FOMC`) exposes accounts to extreme spread widening and whipsaw slippage. [`engine/news_filter.py`](engine/news_filter.py) implements a 3-tier economic calendar blackout coupled with 4 post-news re-engagement gates:

- **Tier 1 (`USD-HIGH`):** `CPI`, `NFP`, `PPI`, `PCE`, `FOMC`, `GDP`, `Retail Sales` — `15m` pre / `15m` post blackout.
- **Tier 2 (`USD-SPEECH`):** Federal Reserve speeches & testimonies — `10m` pre / `15m` post blackout.
- **Tier 3 (`NON-USD-HIGH`):** Major `EUR`, `GBP`, `JPY`, `NZD`, `AUD`, `CAD` red-folder events — `10m` pre / `10m` post blackout.

```mermaid
flowchart TD
    NewsRelease["High-Impact Macro Event Released (CPI / NFP / FOMC)"] --> T15["Post-News Window Reached"]

    T15 --> G3{"Guard 3: Anti-Whipsaw Filter (Range >= $10 & Body < 45%?)"}
    G3 -- "Yes (Two-Way Whip)" --> ExtendLock["Extend Blackout to 45m (Safety Lockdown)"]
    G3 -- "No (Clean Displacement)" --> StrategyActive["Resume Signal Evaluation"]

    StrategyActive --> Signal{"Signal Generated?"}
    Signal -- "Yes" --> G1{"Guard 1: Live Spread <= Limit?"}

    G1 -- "No (Wide Spread)" --> BlockSpread["BLOCKED: Spread Gate"]
    G1 -- "Yes" --> G2{"Guard 2: Aligned With News Impulse (< 30m)?"}

    G2 -- "No (Counter-Trend)" --> BlockCounter["BLOCKED: One-Way Directional Lock"]
    G2 -- "Yes" --> G4{"Guard 4: Risk & Tiered Daily Cap Check"}

    G4 -- "Exceeded" --> BlockRisk["BLOCKED: Stop Exceeds Risk Cap"]
    G4 -- "Valid" --> ExecuteTrade["Execute via MT5 Risk Layer"]
```

---

## Tiered Risk Management & Execution Safeguards

Capital protection and in-trade execution management live in [`core/config.py`](core/config.py), [`core/performance_tracker.py`](core/performance_tracker.py), and [`engine/mt5_interface.py`](engine/mt5_interface.py):

```mermaid
flowchart LR
    subgraph Capital ["Capital & Prop-Firm Protection"]
        C1["Tiered Daily DD Limit: 15% (<$500) | 10% ($500-$4,999) | 4.5% (>=$5k)"]
        C2["Broker magic=0 SL/TP Exit Position-ID Reconciliation"]
        C3["2-Loss Global & Strategy Cooldown Circuit Breakers (7-Day Deal Resolution)"]
        C4["Symmetric BUY & SELL Limit Anti-Chaser Guards"]
        C5["Passive Observation Mode ([DD-HALT]) Instead of Loop Freeze"]
    end

    subgraph Trade ["In-Trade Dynamic Management"]
        T1["Break-Even Cushion (+0.85 pt) at BE_TRIGGER_PCT"]
        T2["50% Partial Profit Close (volume >= 0.02)"]
        T3["ATR-Scaled Dynamic Trailing Activation"]
        T4["Living Shadow Structural M5 Swing Trailing"]
    end
```

---

## Machine Learning & Purged Walk-Forward Validation

- **21-Dimensional Multi-Timeframe Features ([`engine/feature_extractor_v3.py`](engine/feature_extractor_v3.py)):** Computes normalized stationary features across `M5`, `H1`, and `H4` (EMA distance ratios, ATR volatility ratios, M15 ADX state, RSI dynamics, and candle wick/body microstructure).
- **Purged & Embargoed Cross-Validation ([`src/auto_retrain.py`](src/auto_retrain.py)):** Implements temporal fold splitting with explicit purge and embargo intervals between training and validation folds (López de Prado methodology) to prevent overlapping-horizon label leakage.
- **Live Feedback Veto Trainer ([`src/evolve.py`](src/evolve.py)):** Retrains a linear meta-veto filter from closed live MT5 executions once sufficient two-class sample volume (`>= 80` trades) has accumulated.

---

## Repository Structure

```text
parallax/
├── core/
│   ├── config.py                # Tiered drawdown thresholds, session schedules, risk params (.env-driven)
│   ├── indicators.py            # ATR, ADX, RSI, VWAP, and Volume-at-Price (POC) calculations
│   ├── license_manager.py       # Session verification helper
│   ├── performance_tracker.py   # Live league ranking & 7-day-resolved 2-loss cooldowns
│   ├── state_manager.py         # Atomic file-locked JSON persistence (portalocker)
│   └── utils.py                 # Broker DST & UTC session clock alignment
├── engine/
│   ├── mt5_interface.py         # Thread-safe MT5 IPC (mt5_lock), sizing, DD halt, Living Shadow SL
│   ├── feature_extractor_v3.py  # 21-feature multi-timeframe ML feature builder
│   ├── news_filter.py           # ForexFactory 3-tier macro blackout engine + stale cache fallback
│   ├── copier.py                # Lock-guarded local JSONL trade copier queue
│   ├── supabase_pusher.py       # Thread-safe cloud telemetry & open-order reconciliation
│   ├── telegram_notifier.py     # Bot commands (/status, /today, /kill, /pause) & channel alerts
│   └── trade_logger.py          # Automated monthly ledger & drawdown analytics exporter
├── src/
│   ├── goldvx.py                # Main orchestrator loop, regime router & passive observation mode
│   ├── auto_retrain.py          # Purged & embargoed walk-forward ML training pipeline
│   ├── evolve.py                # Live feedback veto trainer
│   └── backtester.py            # Vectorized historical replay harness
├── strategies/                  # Pluggable BaseStrategy SDK & Public Reference Adapters
├── tests/
│   └── test_goldvx_drawdown_observation.py  # 57 automated tests covering halts, tiers & edge cases
├── .env.example
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

---

## Setup & Verification

The automated test suite mocks MetaTrader 5 IPC and runs completely offline on Windows, macOS, or Linux:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment variables (only required for live broker/cloud runs)
cp .env.example .env

# 3. Run the 57-test verification suite
python -m pytest tests/ -v
```
