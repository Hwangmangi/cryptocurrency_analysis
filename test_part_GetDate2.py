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
    date_range = pd.date_range(start='2023-01-01', periods=n, freq='h')
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

#이진 레이블링 (lookahead_hours구간 동안 수익실현지점이 있었다면 1, 아니라면 0)
def label_profit_possible(df, lookahead_hours=5):
    """
    시퀀스가 끝난 후, 주어진 시간 구간(lookahead_hours) 내에 
    종가보다 더 높은 가격을 기록한 시점이 있었는지 여부로 레이블을 붙여주는 함수.
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임 (시간별 OHLCV + 기술적 지표 포함)
        lookahead_hours (int): 수익 실현 가능 여부를 확인할 시간 구간 (예: 5)
        
    Returns:
        pd.DataFrame: 'label' 컬럼이 추가된 데이터프레임
    """
    df = df.copy()  # 원본을 변경하지 않기 위해 복사본 사용

    # 'label' 열을 기본값 NaN으로 초기화 (기본은 수익 실현 불가능)
    df['label'] = pd.NA

    # 시퀀스 끝에서 이후의 가격을 기준으로 수익 실현 여부를 체크
    for i in range(len(df) - lookahead_hours):
        start_price = df['close'].iloc[i]  # 시퀀스 끝의 가격 (현재 가격)
        
        # lookahead_hours 기간 동안 종가 확인
        future_prices = df['close'].iloc[i + 1:i + lookahead_hours + 1]

        # 수익 실현 가능 여부 판단: lookahead_hours 내에 더 높은 가격이 있었다면 수익 실현 가능
        if any(future_prices > start_price):
            df.loc[i, 'label'] = 1  # 수익 실현 가능 -> 1로 레이블링
        else:
            df.loc[i, 'label'] = 0  # 수익 실현 불가능 -> 0으로 레이블링
    df.dropna(subset=['label'], inplace=True)
    return df
    #return df[['close', 'label']]  # 'close'와 'label'만 반환

# 테스트 실행
df = generate_sample_data()
print(f'df.shape:{df.shape}')
#print("df (Head):\n",df.head(40))
# print("df (Head):\n",df.tail())
# df.to_csv('df.csv')

processed_data = preprocess(df, scaling='none')
print(f'processed_data.shape : {processed_data.shape}')
# print("Processed Data (Head):\n", processed_data.head(40))
# print("\nProcessed Data (Tail):\n", processed_data.tail())
# processed_data.to_csv('processed_data.csv')

rawdata = label_profit_possible(processed_data)
print(f'dfl.shape:{rawdata.shape}')
