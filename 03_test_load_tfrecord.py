import os
import numpy as np
import tensorflow as tf
from binance.client import Client
#============================================[ 사용자 설정 파라미터 ]==================================================================================
tfrecord_filename = 'test.tfrecord'
sequence_length = 30  # 시퀀스 길이
feature_dim = 23  # 한 샘플의 특성 수 (레이블 제외)
all_normalization = 'none' # 전체 데이터 정규화 'min-max', 'standard', 'patial-norstand', 'none'
sequence_normalization = 'partial-normalization' # 시퀀스 정규화: 'min-max', 'standard', 'patial-norstand', 'change-rate', 'none'
#=============================================[ 주요 파라미터 ]========================================================================
# Binance API 설정
api_key = 'dQe5j00uyrvcyeJRGXQHRflYqCRZR3KTMBsVsKivpE8COOxN2RwxFyfFbZrFD6OZ'
api_secret = 'kCPemcQpcvw9L1DhH4bIQXtNJASR5mLQT8KtJNb39PNGrjh7Hr8HYB4xd2ncIuH2'

client = Client(api_key, api_secret)
# 데이터 경로 및 파라미터 설정
data_path = r'C:\code\python\autohunting\dataset_raw_1hour38feature'
output_dir = r'C:\code\python\autohunting\dataset_test'
tfrecord_path = os.path.join(output_dir, tfrecord_filename)

def parse_function(proto):
    # TFRecord 데이터에서 사용할 특성 정의
    features = {
        'features': tf.io.FixedLenFeature([sequence_length * feature_dim], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(proto, features)
    features_reshaped = tf.reshape(parsed_features['features'], [sequence_length, feature_dim])

    return features_reshaped, parsed_features['label']


# TFRecordDataset 로드
raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

# 데이터 파싱
dataset = raw_dataset.map(parse_function)

# # 첫 번째, 두 번째, 세 번째 샘플 출력
# for i, (features, label) in enumerate(dataset.take(3)):  # 처음 3개 샘플만 가져옴
#     print(f"Sample {i + 1}:")
#     print(f"Features{features.shape}{type(features)}: {features.numpy()}")
#     print(f"Label{label.shape}{type(label)}: {label.numpy()}\n")


# 전체 데이터셋을 리스트로 변환 (메모리에 로드)
all_samples = list(dataset)

# # 처음 3개 샘플 출력
# print("First 3 samples:")
# for i in range(3):
#     features, label = all_samples[i]
#     print(f"Sample {i + 1}:")
#     print(f"Features{features.shape}{type(features)}: {features.numpy()}")
#     print(f"Label{label.shape}{type(label)}: {label.numpy()}\n")

# 마지막 3개 샘플 출력
print("Last 3 samples:")
for i in range(-3, 0):
    features, label = all_samples[i]
    print(f"Sample {len(all_samples) + i + 1}:")
    print(f"Features{features.shape}{type(features)}: {features.numpy()}")
    print(f"Label{label.shape}{type(label)}: {label.numpy()}\n")

