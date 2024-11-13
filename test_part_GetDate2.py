import pandas as pd
import numpy as np
import talib

def generate_sample_data(n=500):
    """
    임의의 가격 데이터를 생성합니다.
    Args:
        n (int): 생성할 데이터 포인트 수
    Returns:
        DataFrame: 시가, 고가, 저가, 종가, 거래량 데이터
    """
    np.random.seed(42)
    date_range = pd.date_range(start='2023-01-01', periods=n, freq='H')
    open_prices = np.random.randint(5, 10, n)
    high_prices = open_prices + np.random.randint(1, 3, n)
    low_prices = open_prices - np.random.randint(0, 3, n)
    close_prices = open_prices + np.random.randint(0, 5, n)
    volume = np.random.randint(0, 10, n)

    df = pd.DataFrame({
        'timestamp': date_range,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    df.set_index('timestamp', inplace=True)
    return df

def preprocess(df, scaling='standard'):
    """
    기술적 지표를 계산하고 데이터 전처리를 수행합니다.
    Args:
        df (DataFrame): 입력 데이터
        scaling (str): 'normalize' 또는 'standard' 선택 가능
    Returns:
        DataFrame: 기술적 지표와 전처리가 완료된 데이터프레임
    """
    # 기술적 지표 계산
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    # 결측값 제거 (가장 긴 지표로 인해 발생)
    # df.dropna(inplace=True)

    # 전처리
    features = ['open', 'high', 'low', 'close', 'volume', 'MACD', 'MACD_signal', 'RSI',
                'Upper_BB', 'Middle_BB', 'Lower_BB', 'ADX']

    if scaling == 'normalize':
        df[features] = (df[features] - df[features].min()) / (df[features].max() - df[features].min())
    elif scaling == 'standard':
        df[features] = (df[features] - df[features].mean()) / df[features].std()
    elif scaling == 'none':
        return df
    return df[features]

# 테스트 실행
df = generate_sample_data()
print(f'df.shape:{df.shape}')
print("df (Head):\n",df.head(40))
print("df (Head):\n",df.tail())
df.to_csv('df.csv')

processed_data = preprocess(df, scaling='none')
print(f'processed_data.shape : {processed_data.shape}')
print("Processed Data (Head):\n", processed_data.head(40))
print("\nProcessed Data (Tail):\n", processed_data.tail())
processed_data.to_csv('processed_data.csv')
