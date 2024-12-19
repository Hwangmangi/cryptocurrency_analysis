import os
import numpy as np
import tensorflow as tf
from binance.client import Client
#============================================[ 사용자 설정 파라미터 ]==================================================================================
tfrecord_filename = 'TFRecord.1hour22feature'
sequence_length = 30  # 시퀀스 길이
feature_dim = 22  # 한 샘플의 특성 수 (레이블 제외)
normalization1 = 'standard' # 전체 데이터 정규화 'min-max', 'standard', 'none'
normalization2 = 'none' # 시퀀스 정규화: 'min-max', 'standard', 'change-rate', 'none'
#=============================================[ 주요 파라미터 ]========================================================================
# Binance API 설정
api_key = 'dQe5j00uyrvcyeJRGXQHRflYqCRZR3KTMBsVsKivpE8COOxN2RwxFyfFbZrFD6OZ'
api_secret = 'kCPemcQpcvw9L1DhH4bIQXtNJASR5mLQT8KtJNb39PNGrjh7Hr8HYB4xd2ncIuH2'

client = Client(api_key, api_secret)
# 데이터 경로 및 파라미터 설정
data_path = r'C:\code\python\autohunting\dataset_raw_1hour22feature'
output_dir = r'C:\code\python\autohunting\dataset_TFRecord'
tfrecord_path = os.path.join(output_dir, tfrecord_filename)
#============================================[ 주요 함수 ]============================================================================
# TFRecord 직렬화 함수
def serialize_example(features, label):
    feature = {
        'features': tf.train.Feature(float_list=tf.train.FloatList(value=features)),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()
# TFRecord 파싱 함수
def parse_function(proto):
    # TFRecord 데이터에서 사용할 특성 정의
    features = {
        'features': tf.io.FixedLenFeature([sequence_length * feature_dim], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(proto, features)
    features_reshaped = tf.reshape(parsed_features['features'], [sequence_length, feature_dim])

    return features_reshaped, parsed_features['label']

def normalize_features(features, normalization_type='none'):
    """
    주어진 특성을 특정 정규화 방법으로 변환합니다.

    Parameters:
        features (numpy.ndarray): 정규화할 2D 배열 형태의 데이터 (shape: [num_samples, num_features])
        normalization_type (str): 정규화 방법 ('min-max', 'standard', 'none')

    Returns:
        numpy.ndarray: 정규화된 2D 배열 형태의 데이터
    """
    if normalization_type == 'min-max':
        # Min-Max Scaling: 모든 값을 0~1 사이로 스케일링
        feature_min = np.min(features, axis=0)
        feature_max = np.max(features, axis=0)
        normalized_features = (features - feature_min) / (feature_max - feature_min + 1e-8)

    elif normalization_type == 'standard':
        # Standardization: 평균이 0, 표준편차가 1이 되도록 스케일링
        feature_mean = np.mean(features, axis=0)
        feature_std = np.std(features, axis=0)
        normalized_features = (features - feature_mean) / (feature_std + 1e-8)

    elif normalization_type == 'none':
        # 정규화를 하지 않음
        normalized_features = features

    else:
        raise ValueError(f"Unsupported normalization type: {normalization_type}. Use 'min-max', 'standard', or 'none'.")

    return normalized_features

def normalize_sequence(sequence, normalization_type='none'):
    """
    주어진 시퀀스를 특정 정규화 방법으로 변환합니다.

    Parameters:
        sequence (numpy.ndarray): 정규화할 2D 배열 형태의 데이터 (shape: [sequence_length, num_features])
        normalization_type (str): 정규화 방법 ('min-max', 'standard', 'none', 'change-rate')

    Returns:
        numpy.ndarray: 정규화된 시퀀스
    """
    if normalization_type == 'min-max':
        # Min-Max Scaling: 각 시퀀스에서 값을 0~1 사이로 스케일링
        feature_min = np.min(sequence, axis=0)
        feature_max = np.max(sequence, axis=0)
        normalized_sequence = (sequence - feature_min) / (feature_max - feature_min + 1e-8)

    elif normalization_type == 'standard':
        # Standardization: 각 시퀀스에서 평균이 0, 표준편차가 1이 되도록 스케일링
        feature_mean = np.mean(sequence, axis=0)
        feature_std = np.std(sequence, axis=0)
        normalized_sequence = (sequence - feature_mean) / (feature_std + 1e-8)

    elif normalization_type == 'change-rate':
        # Change-rate Normalization: 첫 번째 타임스텝 기준 등락률 계산
        normalized_sequence = (sequence / sequence[0]) - 1

    elif normalization_type == 'none':
        # 정규화를 하지 않음
        normalized_sequence = sequence

    else:
        raise ValueError(f"Unsupported normalization type: {normalization_type}. Use 'min-max', 'standard', 'change-rate', or 'none'.")

    return normalized_sequence
#==============================================[ 실행 코드 ]===================================================================================
# 데이터 처리 및 TFRecord 저장
with tf.io.TFRecordWriter(tfrecord_path) as writer:
    exchange_info = client.get_exchange_info()
    iteration=1
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
        labels = data[:, -1].astype(int)  # 마지막 열 (정수형 변환)
        features = normalize_features(features, normalization_type=normalization1) # 전체 데이터 정규화 'min-max', 'standard', 'none'
        # 시퀀스 데이터 생성
        for i in range(len(features) - sequence_length + 1):
            sequence = features[i:i + sequence_length]  # 시퀀스 길이로 자르기
            sequence = normalize_sequence(sequence, normalization_type=normalization2)  # 시퀀스 정규화: 'min-max', 'standard', 'change-rate', 'none'
            sequence = sequence.flatten()            
            label = labels[i + sequence_length - 1]  # 시퀀스 끝의 label 사용
            
            # TFRecord로 저장
            example = serialize_example(sequence, label)
            writer.write(example)
        
        print(f'{iteration}.Successfully processed and saved {symbol}.txt to TFRecord.')
        iteration+=1

print(f'TFRecord saved to {tfrecord_path}')
#===========================================================================================================================================
