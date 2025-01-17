import os
import numpy as np
import tensorflow as tf
from binance.client import Client
#============================================[ 사용자 설정 파라미터 ]==================================================================================
tfrecord_filename = 'test.tfrecord'
sequence_length = 50  # 시퀀스 길이
feature_dim = 38  # 한 샘플의 특성 수 (레이블 제외)
all_normalization = 'none' # 전체 데이터 정규화 'min-max', 'standard', 'patial-norstand', 'none'
sequence_normalization = 'partial-normalization' # 시퀀스 정규화: 'min-max', 'standard', 'patial-norstand', 'change-rate', 'none'
#=============================================[ 주요 파라미터 ]========================================================================
# Binance API 설정
api_key = 'dQe5j00uyrvcyeJRGXQHRflYqCRZR3KTMBsVsKivpE8COOxN2RwxFyfFbZrFD6OZ'
api_secret = 'kCPemcQpcvw9L1DhH4bIQXtNJASR5mLQT8KtJNb39PNGrjh7Hr8HYB4xd2ncIuH2'

client = Client(api_key, api_secret)
# 데이터 경로 및 파라미터 설정
data_path = r'C:\code\python\autohunting\dataset_raw_1day38feature'
output_dir = r'C:\code\python\autohunting\dataset_test'
tfrecord_path = os.path.join(output_dir, tfrecord_filename)
#============================================[ 주요 함수 ]============================================================================
# TFRecord 직렬화 함수
def serialize_example(features, label):
    feature = {
        'features': tf.train.Feature(float_list=tf.train.FloatList(value=features)),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized_example = example.SerializeToString()

    # 바이트 형식 확인
    if not isinstance(serialized_example, bytes):
        raise ValueError("Serialized example is not bytes")
    
    return serialized_example
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

    elif normalization_type == 'partial-norstand':
        # 특정 열에 대해 정규화와 표준화 분리
        normalized_features = np.zeros_like(features)
        # ['volume', 'open', 'high', 'low', 'close', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'SMA5', 'SMA20', 'SMA50', 'SMA144', 'EMA5', 'EMA20', 'EMA50', 'EMA144',
        #  'MACD', 'MACD_signal', 'MACD_diff', 'RSI6', 'RSI12', 'RSI24', 'ADX', 'SAR', 'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI', 'OBV', 'Chaikin_Osc', 'Momentum',
        #   'ROC', 'ATR', 'STDDEV', 'VWAP', 'Pivot', 'Resistance1', 'Support1', 'label']
        # 정규화할 열
        normalize_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 21, 22]
        for col in normalize_columns:
            feature_min = np.min(features[:, col])
            feature_max = np.max(features[:, col])
            normalized_features[:, col] = (features[:, col] - feature_min) / (feature_max - feature_min + 1e-8)

        # 표준화할 열
        standardize_columns = [13, 14, 15]
        for col in standardize_columns:
            feature_mean = np.mean(features[:, col])
            feature_std = np.std(features[:, col])
            normalized_features[:, col] = (features[:, col] - feature_mean) / (feature_std + 1e-8)

    else:
        raise ValueError(f"Unsupported normalization type: {normalization_type}. Use 'min-max', 'standard', 'partial-norstand' or 'none'.")

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

    elif normalization_type == 'partial-normalization':
        # 특정 열에 대해 정규화와 표준화 분리
        # ['volume','open', 'high', 'low', 'close', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'SMA5', 'SMA20', 'SMA50', 'SMA144', 'EMA5','EMA20', 'EMA50', 'EMA144', 
        #  [ 0]      [1       2       3          4         5             6      7         8        9       10       11        12     13       14       15 ]
        #'MACD', 'MACD_signal','MACD_diff', 'RSI6','RSI12','RSI24', 'ADX' 
        # [ 16         17            18  ]   [19     20      21]     [22] 
        #() :min-max, []:standard,  {}:zero center  , ||:change-rate,

        normalized_sequence = np.zeros_like(sequence)
        
        feature_mean = np.mean(sequence[:, 0])
        feature_std = np.std(sequence[:, 0])
        normalized_sequence[:, 0] = (sequence[:, 0] - feature_mean) / (feature_std + 1e-8)

        feature_mean = np.mean(sequence[:, 1:16])
        feature_std = np.std(sequence[:, 1:16])
        normalized_sequence[:, 1:16] = (sequence[:, 1:16] - feature_mean) / (feature_std + 1e-8)

        feature_mean = np.mean(sequence[:, 16:19])
        feature_std = np.std(sequence[:, 16:19])
        normalized_sequence[:, 16:19] = (sequence[:, 16:19] - feature_mean) / (feature_std + 1e-8)

        feature_mean = np.mean(sequence[:, 19:22])
        feature_std = np.std(sequence[:, 19:22])
        normalized_sequence[:, 19:22] = (sequence[:, 19:22] - feature_mean) / (feature_std + 1e-8)


        for col in range(22, 38):
            feature_mean = np.mean(sequence[:, col])
            feature_std = np.std(sequence[:, col])
            normalized_sequence[:, col] = (sequence[:, col] - feature_mean) / (feature_std + 1e-8)


    elif normalization_type == 'change-rate':
        # Change-rate Normalization: 첫 번째 타임스텝 기준 등락률 계산
        normalized_sequence = (sequence / sequence[0]) - 1

    elif normalization_type == 'none':
        # 정규화를 하지 않음
        normalized_sequence = sequence

    else:
        raise ValueError(f"Unsupported normalization type: {normalization_type}. Use 'min-max', 'standard', 'partial-norstand', 'change-rate', or 'none'.")

    return normalized_sequence
#===========================================================================================================================================
# 데이터 처리 및 TFRecord 저장
print('here1')

with tf.io.TFRecordWriter(tfrecord_path) as writer:
    print('here2')
    file_path = os.path.join(data_path, f'BCCBTC.txt')
    data = np.loadtxt(file_path, delimiter='\t')

    # 마지막 열을 label로 분리
    features = data[:, :feature_dim]  # 모든 열 중 마지막 제외
    labels = data[:, -1].astype(int)  # 마지막 열 (정수형 변환)
    features = normalize_features(features, normalization_type='none') # 전체 데이터 정규화 'min-max', 'standard', 'none'
    print(f'DEBUG : feature.shape : {features.shape}')

    for i in range(len(features) - sequence_length + 1):
        try:
            sequence = features[i:i + sequence_length]
            sequence = normalize_sequence(sequence, normalization_type=sequence_normalization)
            sequence = sequence.flatten()
            label = labels[i + sequence_length - 1]
            example = serialize_example(sequence, label)
            writer.write(example)
        except Exception as e:
            print(f'Error processing record at index {i}: {e}')
            continue        
    print(f'Successfully processed and saved BTCUSDT.txt to TFRecord.')

print(f'TFRecord saved to {tfrecord_path}')
#===========================================================================================================================================
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

with open("1day_38feature_none_partialstandard.txt", "w") as txtfile:
    for i in range(-3, 0):  # 마지막 3개 반복
        features, label = all_samples[i]
        txtfile.write(f"Sample {len(all_samples) + i + 1}:\n")
        txtfile.write(f"Features: {features.numpy().tolist()}\n\n")

print("Last 3 features saved as 'last_3_features.txt'")