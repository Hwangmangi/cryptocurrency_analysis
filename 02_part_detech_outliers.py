import os
import numpy as np
from binance.client import Client

api_key = 'dQe5j00uyrvcyeJRGXQHRflYqCRZR3KTMBsVsKivpE8COOxN2RwxFyfFbZrFD6OZ'
api_secret = 'kCPemcQpcvw9L1DhH4bIQXtNJASR5mLQT8KtJNb39PNGrjh7Hr8HYB4xd2ncIuH2'

client = Client(api_key, api_secret)
data_path = r'C:\code\python\autohunting\dataset_raw_1hour23feature'

exchange_info = client.get_exchange_info()

def detect_outliers_iqr(features):
    outliers = []
    for feature_idx in range(features.shape[1]):
        Q1 = np.percentile(features[:, feature_idx], 25)
        Q3 = np.percentile(features[:, feature_idx], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 이상치가 있는 인덱스를 찾기
        outlier_indices = np.where((features[:, feature_idx] < lower_bound) | (features[:, feature_idx] > upper_bound))[0]
        
        if len(outlier_indices) > 0:
            outliers.append((feature_idx, outlier_indices))  # (특징 인덱스, 이상치 인덱스)

    return outliers

iteration = 1
for s in exchange_info['symbols']:
    symbol = s['symbol']
    file_path = os.path.join(data_path, f'{symbol}.txt')

    if not os.path.exists(file_path):
        print(f'File not found: {file_path}')
        continue

    # 데이터 로드
    try:
        data = np.loadtxt(file_path, delimiter='\t')
    except Exception as e:
        print(f'Error reading {file_path}: {e}')
        continue

    # 마지막 열을 label로 분리
    features = data[:, :-1]  # 모든 열 중 마지막 제외

    # 이상치 탐지
    outliers = detect_outliers_iqr(features)

    # 이상치가 있으면 출력
    if outliers:
        print(f"File: {file_path}")
        for feature_idx, outlier_indices in outliers:
            print(f"Feature {feature_idx} contains outliers at indices: {outlier_indices}")
