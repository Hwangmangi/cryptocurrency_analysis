import pandas as pd
import numpy as np
import talib
from binance.client import Client
from datetime import datetime, timedelta
import ccxt

# Binance API Client 설정 (자신의 API 키와 시크릿 키를 여기에 입력)
api_key = 'dQe5j00uyrvcyeJRGXQHRflYqCRZR3KTMBsVsKivpE8COOxN2RwxFyfFbZrFD6OZ'
api_secret = 'kCPemcQpcvw9L1DhH4bIQXtNJASR5mLQT8KtJNb39PNGrjh7Hr8HYB4xd2ncIuH2'
client = Client(api_key, api_secret)
'''
# 분봉
Client.KLINE_INTERVAL_1MINUTE  # '1m' : 1분봉
Client.KLINE_INTERVAL_3MINUTE  # '3m' : 3분봉
Client.KLINE_INTERVAL_5MINUTE  # '5m' : 5분봉
Client.KLINE_INTERVAL_15MINUTE # '15m': 15분봉
Client.KLINE_INTERVAL_30MINUTE # '30m': 30분봉

# 시간봉
Client.KLINE_INTERVAL_1HOUR    # '1h' : 1시간봉
Client.KLINE_INTERVAL_2HOUR    # '2h' : 2시간봉
Client.KLINE_INTERVAL_4HOUR    # '4h' : 4시간봉
Client.KLINE_INTERVAL_6HOUR    # '6h' : 6시간봉
Client.KLINE_INTERVAL_8HOUR    # '8h' : 8시간봉
Client.KLINE_INTERVAL_12HOUR   # '12h': 12시간봉

# 일봉
Client.KLINE_INTERVAL_1DAY     # '1d' : 1일봉
Client.KLINE_INTERVAL_3DAY     # '3d' : 3일봉

# 주봉
Client.KLINE_INTERVAL_1WEEK    # '1w' : 1주봉

# 월봉
Client.KLINE_INTERVAL_1MONTH   # '1M' : 1개월봉
'''

def fetch_listing_date(client, symbol, interval):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=1)
    if klines:
        first_kline_timestamp = klines[0][0]  # 첫 번째 Kline의 타임스탬프
        return first_kline_timestamp
    return None

def fetch_all_klines_from_listing(client, symbol, interval):
    # 초기 시작 날짜를 10년 전으로 설정
    start_date = datetime.now() - timedelta(days=3650)  # 10년 전
    start_time = start_date.strftime('%d %b %Y %H:%M:%S')

    # 데이터를 받아올 리스트
    klines = []
    limit = 1000  # Binance API 요청당 최대 데이터 수

    while True:
        try:
            # 데이터를 요청
            new_klines = client.get_historical_klines(symbol, interval, start_time, limit=limit)

            if not new_klines:
                break

            klines.extend(new_klines)  # 누적 저장
            # 마지막 클로즈 시간 이후로 시작 시간 설정
            start_time = new_klines[-1][6] + 1  # 마지막 close_time + 1ms

            # 요청한 데이터가 1000개보다 적으면 종료
            if len(new_klines) < limit:
                break

        except Exception as e:
            # 에러가 발생하면 날짜를 하루씩 앞당기며 시도
            print(f"에러 발생: {e}")
            start_date += timedelta(days=1)
            start_time = start_date.strftime('%d %b %Y %H:%M:%S')
            print(f"다시 시도하는 날짜: {start_time}")

    # 데이터프레임 생성
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    # 필요한 컬럼만 반환하며 숫자 데이터로 변환
    df = df[['timestamp', 'volume', 'open', 'high', 'low', 'close']].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # timestamp를 datetime으로 변환

    return df



def fetch_and_preprocess(symbol, scaling='none'):
    """
    특정 코인 심볼의 전체 기간의 1시간봉 데이터를 불러와 기술적 지표를 계산하고 정규화 또는 표준화를 진행합니다.
    
    Args:
        symbol (str): 코인 심볼 (예: 'BTCUSDT')
        scaling (str): 전처리 방식, 'standard', 'normalize', 'none' 중 선택 가능
    
    Returns:
        DataFrame: 기술적 지표와 전처리가 완료된 데이터프레임
    """
    
    # 전체 기간 1시간봉 데이터 가져오기
    df = fetch_all_klines_from_listing(client, symbol, Client.KLINE_INTERVAL_1HOUR)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # 기술적 지표 계산
     # 이동 평균 추가
    # df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    # df['MACD_diff'] = df['MACD'] - df['MACD_signal']

    df['SMA5'] = df['close'].rolling(window=5).mean()  # 5시간 단순이동평균
    df['SMA20'] = df['close'].rolling(window=20).mean()  # 20시간 단순이동평균
    df['SMA50'] = df['close'].rolling(window=50).mean()  # 50시간 단순이동평균
    df['SMA144'] = df['close'].rolling(window=144).mean()  # 144시간 단순이동평균
    df['EMA5'] = talib.EMA(df['close'], timeperiod=5)  # 20시간 지수이동평균
    df['EMA20'] = talib.EMA(df['close'], timeperiod=20)  # 20시간 지수이동평균
    df['EMA50'] = talib.EMA(df['close'], timeperiod=50)  # 50시간 지수이동평균
    df['EMA144'] = talib.EMA(df['close'], timeperiod=144)  # 144시간 지수이동평균

    # df['RSI6'] = talib.RSI(df['close'], timeperiod=6)
    # df['RSI12'] = talib.RSI(df['close'], timeperiod=12)
    # df['RSI24'] = talib.RSI(df['close'], timeperiod=24)

    # df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    # df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    # 가장 긴 기간을 가진 기술적 지표로 인한 결측값 제거
    df.dropna(inplace=True)

    # 전처리
    # features = ['open', 'high', 'low', 'close', 'volume','MACD', 'MACD_signal','MACD_diff', 'SMA5', 'SMA20', 'SMA50', 'SMA144', 'EMA5','EMA20', 'EMA50', 'EMA144',  'RSI6','RSI12','RSI24', 
    #             'Upper_BB', 'Middle_BB', 'Lower_BB', 'ADX']
    features = ['volume','open', 'high', 
                'low', 'close', 
                 'SMA5', 'SMA20', 'SMA50', 'SMA144', 'EMA5','EMA20',
                   'EMA50', 'EMA144']
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
def add_labels(df, lookahead_steps=5):
    """
    종가를 기준으로 이후 5시간 동안의 종가보다 더 높은 값이 하나라도 있으면 1, 아니면 0을 라벨로 추가.
    
    Args:
        df (pd.DataFrame): 종가를 포함하는 DataFrame (timestamp, close 등)
        lookahead_steps (int): 라벨링을 위해 고려할 시간 범위 (기본값은 5)
        
    Returns:
        pd.DataFrame: 라벨이 추가된 DataFrame
    """
    labels = []

    for i in range(len(df)):
        current_close = df.iloc[i]['close']
        
        # 이후 5시간 동안의 종가
        future_close = df.iloc[i+1:i+1+lookahead_steps]['close'] if i+1+lookahead_steps <= len(df) else [] # 여기까지는 맞는듯함
        
        # 종가가 더 큰 값이 있으면 1, 없으면 0
        if len(future_close) > 0 and any(future_close > current_close):
            labels.append(1)
        else:
            labels.append(0)

    # 라벨 추가
    df['label'] = labels  # 인덱스를 따로 지정할 필요 없음

    # 마지막 lookahead_steps 만큼 데이터 자르기
    df = df.iloc[:-lookahead_steps] 
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

print("Fetching and preprocessing data...")
df = fetch_and_preprocess("BTCUSDT", scaling='none')  # 정규화를 수행 (scaling='normalize')

# # 수익 실현 가능 여부 레이블링
# print("Labeling data...")
df = add_labels(df)  # 5시간 내 수익 실현 여부를 레이블링

# 데이터프레임 정보 출력
print("Processed Data Sample:")
# 상위 20개 행 출력
print(df.head(20))

# 하위 20개 행 출력
print(df.tail(20))

df_values = df.values
print(type)
# 텍스트 파일로 저장 (탭 구분)
file_path = 'C:/code/python/autohunting/dataset_test/test.txt'

# 텍스트 파일로 저장
with open(file_path, 'w') as f:
    for row in df_values:
        f.write('\t'.join(map(str, row)) + '\n')

print(f"파일이 {file_path}에 저장되었습니다.")

data = np.loadtxt(file_path, delimiter='\t')

# [104, 3] 위치의 데이터 출력 (0-based index)
print(data.shape,type(data),data[103, 2])
