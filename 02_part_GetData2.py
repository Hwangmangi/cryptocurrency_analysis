import pandas as pd
import numpy as np
import talib
from binance.client import Client

# Binance API Client 설정 (자신의 API 키와 시크릿 키를 여기에 입력)
api_key = 'dQe5j00uyrvcyeJRGXQHRflYqCRZR3KTMBsVsKivpE8COOxN2RwxFyfFbZrFD6OZ'
api_secret = 'kCPemcQpcvw9L1DhH4bIQXtNJASR5mLQT8KtJNb39PNGrjh7Hr8HYB4xd2ncIuH2'
client = Client(api_key, api_secret)
#    df = fetch_all_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "1 Jan 2017")

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
data = fetch_and_preprocess('BTCUSDT', scaling='none')
#print(data)

# 모든 코인 심볼 얻기
tickers = client.get_all_tickers()
symbols = [ticker['symbol'] for ticker in tickers]

i=1
for ticker in symbols:
    df = fetch_and_preprocess(ticker, scaling='none')
    print(f'{i}번째 : ticker, df.shape:{df.shape}')
    df = label_profit_possible(df, lookahead_hours=5)
    print(f'{i}번째 : ticker, df_lable.shape:{df.shape}')
    save_data_to_csv(df, ticker)
    i+=1
    break