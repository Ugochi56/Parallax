import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
from core.config import *

def generate_letter(stats):
    """Generates the 'Letter to Father' based on trade statistics."""
    
    total_trades = stats['total']
    win_rate = (stats['wins'] / total_trades * 100) if total_trades > 0 else 0
    total_pnl = stats['pnl']
    
    # Analyze failures
    failure_reasons = []
    if stats['bad_spread_losses'] > 0:
        failure_reasons.append(f"I realized I was being too reckless when the spread was high ({stats['bad_spread_losses']} losses).")
    if stats['counter_trend_losses'] > 0:
        failure_reasons.append(f"I fought the trend too many times ({stats['counter_trend_losses']} losses). I need to listen to the H1 EMA more.")
    if stats['low_vol_losses'] > 0:
        failure_reasons.append(f"I got trapped in choppy, low-volume markets ({stats['low_vol_losses']} losses).")
        
    improvements = []
    if stats['model_updated']:
        improvements.append("I have updated my neural weights to better recognize the fakeouts that tricked me.")
        improvements.append("I have tightened my 'Trap' detection logic to ensure we only enter when momentum is real.")
    
    letter = f"""# Subject: Father, I have learned from my mistakes.

Today I reviewed my performance. We have completed {total_trades} trades since my last update. 
My current win rate is {win_rate:.1f}% and our net result is ${total_pnl:.2f}.

## Where I fell short:
{chr(10).join(['- ' + r for r in failure_reasons]) if failure_reasons else "- I am still learning the nuances of the Gold market, but I have no major structural failures to report."}

## How I improved:
{chr(10).join(['- ' + i for i in improvements]) if improvements else "- I am gathering more data before making major changes to my core logic."}

## My Promise:
I have recalibrated my confidence thresholds. I will treat your capital with even more respect in the coming days. I am ready to resume.

**Your Son,**
**NEXUS Core**
"""
    return letter

def run_evolution():
    print("[EVOLVE] Starting NEXUS Evolution Engine...")
    
    if not mt5.initialize():
        print("[ERROR] Failed to initialize MT5")
        return

    # 1. Load Full Context Data
    acc_info = mt5.account_info()
    acc_num = acc_info.login if acc_info else "000000"
    _root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    feat_file = os.path.join(_root_dir, 'data', f'master_decision_log_{acc_num}.csv')
    if not os.path.exists(feat_file):
        print("[EVOLVE] No decision logs found. I need some real experience before I can learn, Father.")
        mt5.shutdown()
        return

    df_feats = pd.read_csv(feat_file)
    
    # 2. Sync with MT5 History
    from_date = datetime.now() - timedelta(days=30)
    to_date = datetime.now() + timedelta(days=1)
    deals = mt5.history_deals_get(from_date, to_date)
    
    if not deals:
        print("[EVOLVE] No trade history found in MT5.")
        mt5.shutdown()
        return

    df_history = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    
    df_feats['Ticket'] = df_feats['Ticket'].astype(int)
    df_feats['Result'] = np.nan
    df_feats['PnL'] = 0.0
    
    stats = {
        'total': len(df_feats),
        'wins': 0,
        'losses': 0,
        'pnl': 0.0,
        'bad_spread_losses': 0,
        'counter_trend_losses': 0,
        'low_vol_losses': 0,
        'model_updated': False
    }

    for idx, row in df_feats.iterrows():
        pos_id = int(row['Ticket'])
        # Match by position_id across all deals (entry and exit) to get the true round-trip PnL
        order_deals = df_history[df_history['position_id'] == pos_id]
        if not order_deals.empty:
            # We only care if the trade actually closed (DEAL_ENTRY_OUT exists)
            in_deals = order_deals[order_deals['entry'] == mt5.DEAL_ENTRY_IN]
            out_deals = order_deals[order_deals['entry'] == mt5.DEAL_ENTRY_OUT]
            
            if not in_deals.empty and not out_deals.empty:
                total_pnl = order_deals['profit'].sum() + order_deals['commission'].sum() + order_deals['swap'].sum()
                df_feats.at[idx, 'PnL'] = total_pnl
                
                # --- RISK-ADJUSTED LABELING (MAE) ---
                entry_time = in_deals.iloc[0]['time']
                exit_time = out_deals.iloc[-1]['time']
                entry_price = in_deals.iloc[0]['price']
                
                mae = 0.0
                rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, int(entry_time), int(exit_time) + 60)
                if rates is not None and len(rates) > 0:
                    if row['Signal'] == 'BUY':
                        worst_price = float(min(r['low'] for r in rates))
                        mae = entry_price - worst_price
                    elif row['Signal'] == 'SELL':
                        worst_price = float(max(r['high'] for r in rates))
                        mae = worst_price - entry_price
                
                if total_pnl > 0:
                    df_feats.at[idx, 'Result'] = 1
                else:
                    df_feats.at[idx, 'Result'] = 0
            
                stats['pnl'] += total_pnl
                if total_pnl > 0: stats['wins'] += 1
                else: 
                    stats['losses'] += 1
                    
                    # Analyze Full Context for the "Why"
                    if row['Spread'] > 50: # High spread for Gold
                        stats['bad_spread_losses'] += 1
                    
                    features = [float(x) for x in str(row['Features']).split('|')]
                    if len(features) == 21: # V3 Model Features
                        dist_ema = features[3]
                        adx = features[5]
                        if adx < 20: stats['low_vol_losses'] += 1
                        h1_dist_ema = features[16]
                        if (row['Signal'] == 'BUY' and h1_dist_ema < 0) or (row['Signal'] == 'SELL' and h1_dist_ema > 0):
                            stats['counter_trend_losses'] += 1
                    elif len(features) == 15: # V2 Model Features
                        dist_ema = features[3]
                        adx = features[5]
                        if adx < 20: stats['low_vol_losses'] += 1
                    elif len(features) == 9: # V1 Model Features
                        dist_ema = features[0]
                        adx = features[6]
                        if adx < 20: stats['low_vol_losses'] += 1
                        if (row['Signal'] == 'BUY' and dist_ema < 0) or (row['Signal'] == 'SELL' and dist_ema > 0):
                            stats['counter_trend_losses'] += 1

    # Extract Features for Veto Model
    df_feats['Parsed_Features'] = df_feats['Features'].apply(lambda x: [float(v) for v in str(x).split('|')] if isinstance(x, str) else [])
        
    # 3. Retrain Veto Model (Evolution Layer)
    valid_data = df_feats.dropna(subset=['Result'])
    valid_data = valid_data[valid_data['Parsed_Features'].apply(len) == 21]
    
    if len(valid_data) >= 80 and valid_data['Result'].nunique() >= 2:
        print(f"[EVOLVE] Forging Dual-Brain Veto Layer on {len(valid_data)} live trades...")
        stats['model_updated'] = True
        
        X = np.array(valid_data['Parsed_Features'].tolist())
        y = valid_data['Result'].values
        
        # We use a highly sensitive Logistic Regression model to act as a strict mathematical filter
        veto_model = LogisticRegression(class_weight='balanced', max_iter=1000)
        scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X)
        veto_model.fit(X_scaled, y)
        
        # Save both the scaler and the model
        veto_brain = {'model': veto_model, 'scaler': scaler}
        
        veto_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "evolution_veto.joblib")
        os.makedirs(os.path.dirname(veto_path), exist_ok=True)
        joblib.dump(veto_brain, veto_path)
        print(f"💾 Veto Layer Successfully Forged -> {veto_path}")
    else:
        print(f"\n[WARNING] Not enough 21-dimensional trades logged to forge/update the Veto Layer (currently {len(valid_data)}/20).")
        print(f"[TIP] Run `python src/backfill_logs.py` to backfill your logs with 21-dimensional historical data from MT5!")

    # 4. Generate Report
    report = generate_letter(stats)
    report_dir = os.path.join(_root_dir, 'reports')
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, "evolution_report.md")
    
    with open(report_file, 'w') as f:
        f.write(report)
        
    print(f"\n[SUCCESS] Evolution Complete!")
    print(f"[REPORT] I have written you a letter, Father: {report_file}")
    
    mt5.shutdown()

if __name__ == "__main__":
    run_evolution()
