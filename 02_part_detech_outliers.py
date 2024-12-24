import os
import numpy as np
from binance.client import Client

api_key = 'dQe5j00uyrvcyeJRGXQHRflYqCRZR3KTMBsVsKivpE8COOxN2RwxFyfFbZrFD6OZ'
api_secret = 'kCPemcQpcvw9L1DhH4bIQXtNJASR5mLQT8KtJNb39PNGrjh7Hr8HYB4xd2ncIuH2'

client = Client(api_key, api_secret)
data_path = r'C:\code\python\autohunting\dataset_raw_1hour38feature'

exchange_info = client.get_exchange_info()

def detect_outliers_z_score(features, threshold=3):
    outliers = []
    for feature_idx in range(features.shape[1]):
        mean = np.mean(features[:, feature_idx])
        std = np.std(features[:, feature_idx])
        z_scores = np.abs((features[:, feature_idx] - mean) / std)
        
        # 이상치가 있는 인덱스를 찾기
        outlier_indices = np.where(z_scores > threshold)[0]
        
        if len(outlier_indices) > 0:
            outliers.append((feature_idx, outlier_indices, mean, std))  # (특징 인덱스, 이상치 인덱스, 평균, 표준편차)

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
    features = data[:, :-1]

    # 이상치 탐지
    outliers = detect_outliers_z_score(features)

    # 이상치가 있으면 출력
    if outliers:
        print(f"File: {file_path}")
        for feature_idx, outlier_indices, mean, std in outliers:
            outlier_values = features[outlier_indices, feature_idx]
            print(f"Feature {feature_idx} contains outliers at indices: {outlier_indices}")
            print(f"Outlier values: {outlier_values}")
            print(f"Mean: {mean}, Standard Deviation: {std}")

    iteration += 1