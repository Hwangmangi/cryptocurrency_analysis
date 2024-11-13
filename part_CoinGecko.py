import requests
import pandas as pd
from datetime import datetime

# CoinGecko API URL
BASE_URL = "https://api.coingecko.com/api/v3"

# 1. 코인 목록 가져오기 함수
def get_coin_list():
    url = f"{BASE_URL}/coins/list"
    response = requests.get(url)
    if response.status_code == 200:
        coin_list = response.json()  # 리스트 형태로 반환
        return coin_list
    else:
        print(f"Error fetching coin list: {response.status_code}")
        return None

a=get_coin_list()
print(a[1],len(a))

def get_coin_ticker_data(coin_id, vs_currency='usd', days=30):
    # vs_currency: 대상 통화 (기본값: USD)
    # days: 가져올 데이터의 기간 (기본값: 30일)
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': vs_currency,
        'days': days,
        'interval': 'daily'  # daily 간격으로 데이터 요청
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        # OHLC 데이터를 가져옴 (시가, 고가, 저가, 종가, 거래량)
        prices = data.get('prices', [])
        market_caps = data.get('market_caps', [])
        total_volumes = data.get('total_volumes', [])
        
        # 데이터를 pandas DataFrame으로 변환
        ohlc_data = []
        for i in range(len(prices)):
            timestamp = prices[i][0] / 1000  # 밀리초 -> 초로 변환
            date = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')
            open_price = market_caps[i][1] / 1e6  # market cap을 기준으로 시가 계산
            close_price = prices[i][1]
            high_price = max(prices[i][1], market_caps[i][1] / 1e6)  # 단순 예시로 고가와 시가 비교
            low_price = min(prices[i][1], market_caps[i][1] / 1e6)   # 단순 예시로 저가와 시가 비교
            volume = total_volumes[i][1]
            
            ohlc_data.append([date, open_price, close_price, high_price, low_price, volume])
        
        # pandas DataFrame으로 반환
        df = pd.DataFrame(ohlc_data, columns=['Date', 'Open', 'Close', 'High', 'Low', 'Volume'])
        return df
    else:
        print(f"Error fetching data for {coin_id}: {response.status_code}")
        return None
coin_list = get_coin_list()
i=1
if coin_list:
    print(f'{i}th')
    print(f"Total coins found: {len(coin_list)}")
        # 예시로 첫 번째 코인의 ID를 가져옴
    coin_id = coin_list[0]['id']  # 첫 번째 코인의 ID (예: 'bitcoin')

        # 2. 비트코인 시세 데이터 가져오기 (30일 기간)
    coin_data = get_coin_ticker_data(coin_id, vs_currency='usd', days=30)
    if coin_data is not None:
        print(coin_data.head())  # 첫 5일 데이터 출력
    else:
        print("none data")
    i+=1