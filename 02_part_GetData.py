import requests
import pandas as pd
import talib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Binance API에서 데이터를 가져오는 함수
def get_binance_data(symbol, interval='1h', limit=1000):
    url = f'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # 데이터프레임으로 변환
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # 시계열 데이터에 timestamp를 datetime 형태로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 필요한 열만 추출하고 형 변환
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.astype({'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32'})
    
    return df

# MACD, RSI, 볼린저 밴드, ADX 계산하는 함수
def add_technical_indicators(df):
    # MACD 계산 (최소 26개 이상의 데이터가 필요)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    # RSI 계산 (최소 14개의 데이터가 필요)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    
    # 볼린저 밴드 계산
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    
    # ADX 계산 (최소 14개의 데이터가 필요)
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    
    # 초기 데이터에 대해 NaN 처리된 값을 반환
    return df

# 전처리 및 시퀀스 데이터 생성 함수
def preprocess_data(symbol, interval='1h', limit=1000, scaler=None):
    df = get_binance_data(symbol, interval, limit)
    df = add_technical_indicators(df)
    
    # 최소 26개의 데이터가 있어야 MACD가 계산 가능
    if len(df) < 26:
        return None, None, None  # 데이터가 부족하면 None을 반환
    
    # MACD 계산 후, 처음 26개의 데이터는 NaN이므로 이를 제외한 데이터만 사용
    df = df.dropna()
    
    # 스케일링 (MinMaxScaler 또는 다른 방식)
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume', 'macd', 'rsi', 'bb_upper', 'bb_middle', 'bb_lower', 'adx']])
    else:
        df_scaled = scaler.transform(df[['open', 'high', 'low', 'close', 'volume', 'macd', 'rsi', 'bb_upper', 'bb_middle', 'bb_lower', 'adx']])
    
    # 시퀀스 데이터로 변환 (예시: 60시간의 데이터를 하나의 입력으로 사용)
    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(df_scaled)):
        X.append(df_scaled[i-sequence_length:i])  # 60시간 데이터
        y.append(df_scaled[i, 3])  # 종가 (예시)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler

# 실행 예시
symbol = 'BTCUSDT'
X, y, scaler = preprocess_data(symbol)
