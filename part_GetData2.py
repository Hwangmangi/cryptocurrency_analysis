import pandas as pd
import numpy as np
import talib
from binance.client import Client

# Binance API Client 설정 (자신의 API 키와 시크릿 키를 여기에 입력)
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
client = Client(api_key, api_secret)

def fetch_all_klines(symbol, interval, start_str):
    """
    특정 코인 심볼의 모든 OHLCV 데이터를 Binance에서 가져옵니다.
    
    Args:
        symbol (str): 코인 심볼 (예: 'BTCUSDT')
        interval (str): 데이터 간격 (예: '1h')
        start_str (str): 데이터 시작 날짜 (예: '1 Jan 2017')
    
    Returns:
        DataFrame: OHLCV 데이터 전체 기간의 데이터프레임
    """
    klines = []
    limit = 1000  # Binance는 한 번에 최대 1000개 데이터 반환
    start_time = start_str

    while True:
        # 새로운 데이터를 요청하고 가져옴
        new_klines = client.get_historical_klines(symbol, interval, start_time, limit=limit)
        if not new_klines:
            break  # 데이터가 없으면 종료

        klines.extend(new_klines)  # 데이터를 klines 리스트에 누적
        start_time = new_klines[-1][0]  # 시작 시간을 마지막 데이터 이후로 설정

    # 데이터프레임 생성 및 반환
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                                       'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 
                                       'taker_buy_quote', 'ignore'])
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)

def fetch_and_preprocess(symbol, scaling='standard'):
    """
    특정 코인 심볼의 전체 기간의 1시간봉 데이터를 불러와 기술적 지표를 계산하고 정규화 또는 표준화를 진행합니다.
    
    Args:
        symbol (str): 코인 심볼 (예: 'BTCUSDT')
        scaling (str): 전처리 방식, 'standard', 'normalize', 'none' 중 선택 가능
    
    Returns:
        DataFrame: 기술적 지표와 전처리가 완료된 데이터프레임
    """
    
    # 전체 기간 1시간봉 데이터 가져오기
    df = fetch_all_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "1 Jan 2017")
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # 기술적 지표 계산
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    # 가장 긴 기간을 가진 기술적 지표로 인한 결측값 제거
    df.dropna(inplace=True)

    # 전처리
    features = ['open', 'high', 'low', 'close', 'volume', 'MACD', 'MACD_signal', 'RSI', 
                'Upper_BB', 'Middle_BB', 'Lower_BB', 'ADX']
    
    # ReturnTransform 이나 LogReturnTransform을 
    if scaling == 'normalize':
        df[features] = (df[features] - df[features].min()) / (df[features].max() - df[features].min())

    elif scaling == 'standard':
        df[features] = (df[features] - df[features].mean()) / df[features].std()

    elif scaling == 'ReturnTransform':
        df[features] = df[features].pct_change()  # 변화율 계산
        df.dropna(inplace=True)  # 변화율 계산으로 생긴 NaN 제거

    elif scaling == 'DifferencingTransform':
        df[features] = df[features].diff()  # 각 feature의 차분 계산
        df.dropna(inplace=True)  # 첫 번째 NaN 행 제거

    elif scaling == 'LogReturnTransform':
        df[features] = np.log(df[features] / df[features].shift(1))  # 로그 수익률 계산
        df.dropna(inplace=True)  # NaN 값 제거

    elif scaling == 'EMA_Normalize': #EMA를 이용하면 최근 데이터에 더 큰 가중치를 두는 방식으로 정규화
        for feature in features:
            df[feature] = (df[feature] - df[feature].ewm(span=10).mean()) / df[feature].ewm(span=10).std()

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
# df : [open, high, low, close, volumem, MACD, MACD_signal, RSI, Upper_BB, Middle_BB. Lower_BB, ADX, label]
def save_data_to_csv(df, filename):
    """
    데이터를 CSV 파일로 저장합니다.

    Args:
        df (pd.DataFrame): 저장할 데이터프레임
        filename (str): 저장할 파일 경로 및 이름
    """
    df.to_csv(filename)
    print(f"Data saved to {filename}")

# 예시 호출
data = fetch_and_preprocess('BTCUSDT', scaling='standard')
print(data)

# 모든 코인 심볼 얻기
tickers = client.get_all_tickers()
symbols = [ticker['symbol'] for ticker in tickers]

def save_data_to_csv(df, filename):
    """
    데이터를 CSV 파일로 저장합니다.

    Args:
        df (pd.DataFrame): 저장할 데이터프레임
        filename (str): 저장할 파일 경로 및 이름
    """
    df.to_csv(filename)
    print(f"Data saved to {filename}")

#==============================================레이블링 관련 함수====================================================================
#이진 레이블링 (상승/하락)
def label_binary(df, horizon=1):
    """
    현재 가격 대비 horizon 시간 후 가격 변화에 따라 상승(1), 하락(0) 레이블 생성
    
    Args:
        df (pd.DataFrame): 코인 데이터 프레임, 'close' 열이 필요함
        horizon (int): 레이블링을 위한 시간 차이 (시간 단위)
    
    Returns:
        pd.DataFrame: 상승/하락 레이블이 추가된 데이터 프레임
    """
    df['future_price'] = df['close'].shift(-horizon)  # horizon 시간 후 가격
    df['label'] = (df['future_price'] > df['close']).astype(int)
    df.dropna(inplace=True)  # future_price의 NaN 제거
    return df[['close', 'label']]


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

'''
class NormalizeByFirstRow(tf.keras.layers.Layer):  # 단순 나누기 연산 : 현재값 / 첫 타입스텝값
    def __init__(self, epsilon=1e-10, **kwargs):
        super(NormalizeByFirstRow, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        first_row = inputs[:, 0, :]  # shape: (batch_size, feature_length)
        first_row = tf.maximum(first_row, self.epsilon)  # 작은 값으로 대체
        normalized = inputs / tf.expand_dims(first_row, axis=1)
        return normalized

'''


'''
class NormalizeByFirstStep(tf.keras.layers.Layer):  # 변화율 연산 : (현재값 - 첫 타입스텝값) / 첫 타임스텝값
    def __init__(self, epsilon=1e-10, **kwargs):
        """
        첫 번째 타임스텝을 기준으로 시퀀스 데이터의 변화율을 계산하는 레이어.
        
        Args:
            epsilon (float): 작은 값으로 0으로 나누는 오류를 방지.
        """
        super(NormalizeByFirstStep, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        """
        시퀀스 데이터에서 첫 번째 타임스텝을 기준으로 변화율을 계산.

        Args:
            inputs (tensor): (batch_size, time_steps, features) 형태의 시퀀스 데이터

        Returns:
            tensor: 변화율로 정규화된 (batch_size, time_steps, features) 텐서
        """
        # 첫 번째 타임스텝의 값을 가져옴 (배치 크기, 피처 수에 대해 첫 번째 시퀀스 값)
        first_step_values = inputs[:, 0, :]  # shape: (batch_size, features)
        
        # 작은 값으로 나누는 오류 방지
        first_step_values = tf.maximum(first_step_values, self.epsilon)

        # 각 시퀀스 샘플의 첫 번째 타임스텝을 기준으로 변화율 계산
        normalized = (inputs - tf.expand_dims(first_step_values, axis=1)) / tf.expand_dims(first_step_values, axis=1)

        return normalized
'''