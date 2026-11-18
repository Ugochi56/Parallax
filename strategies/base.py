class BaseStrategy:
    def __init__(self, name="BASE"):
        self.name = name

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, **kwargs):
        """
        Evaluates market data and returns a trading signal dictionary.
        Must be implemented by child strategies.
        
        Returns:
            dict: {
                'signal': 'BUY' | 'SELL' | 'FLAT',
                'sl': float,
                'tp': float (optional, if calculated inside strategy),
                'confidence': float,
                'comment': str
            }
        """
        raise NotImplementedError("Strategy must implement evaluate() method")

    def get_trend_direction(self, df_h1, df_h4=None):
        """
        Calculates higher-timeframe trend direction using confirmed (closed) candles.
        Prevents intra-candle live tick wicks from misclassifying macro trend.
        """
        if df_h1 is None or len(df_h1) < 50: return 'FLAT'
        
        # Use confirmed closed candle (iloc[-2]) if available, to avoid live-tick noise
        h1_idx = -2 if len(df_h1) >= 2 else -1
        h1_ema_50 = df_h1['close'].ewm(span=50, adjust=False).mean().iloc[h1_idx]
        h1_close = df_h1.iloc[h1_idx]['close']
        
        if df_h4 is not None and len(df_h4) >= 50:
            h4_idx = -2 if len(df_h4) >= 2 else -1
            h4_ema_50 = df_h4['close'].ewm(span=50, adjust=False).mean().iloc[h4_idx]
            h4_close = df_h4.iloc[h4_idx]['close']
            if h1_close > h1_ema_50 and h4_close > h4_ema_50: return 'UP'
            if h1_close < h1_ema_50 and h4_close < h4_ema_50: return 'DOWN'
            return 'FLAT'
            
        if h1_close > h1_ema_50: return 'UP'
        if h1_close < h1_ema_50: return 'DOWN'
        return 'FLAT'

    def reset(self):
        """
        Flushes all active algorithmic memory (strats/arrays).
        """
        pass

