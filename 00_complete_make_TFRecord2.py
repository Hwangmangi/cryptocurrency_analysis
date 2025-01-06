import os
import numpy as np
import tensorflow as tf
from binance.client import Client

#============================================[ 사용자 설정 파라미터 ]==================================================================================
tfrecord_train_filename = '1day30seq23feature_TRAIN.tfrecord'
tfrecord_val_filename = '1day30seq23feature_VAL.tfrecord'
data_path = r'C:\code\python\autohunting\dataset_raw_1day38feature'
output_dir = r'C:\code\python\autohunting\dataset_TFRecord'
sequence_length = 30  # 시퀀스 길이
feature_dim = 23  # 한 샘플의 특성 수 (레이블 제외)
all_normalization = 'none' # 전체 데이터 정규화 'min-max', 'standard', 'partial-normalization', 'none'
sequence_normalization = 'partial-normalization' # 시퀀스 정규화: 'min-max', 'standard', 'partial-normalization', 'change-rate', 'none'
#=============================================[ 주요 파라미터 ]========================================================================
# Binance API 설정
api_key = 'dQe5j00uyrvcyeJRGXQHRflYqCRZR3KTMBsVsKivpE8COOxN2RwxFyfFbZrFD6OZ'
api_secret = 'kCPemcQpcvw9L1DhH4bIQXtNJASR5mLQT8KtJNb39PNGrjh7Hr8HYB4xd2ncIuH2'

client = Client(api_key, api_secret)
# 데이터 경로 및 파라미터 설정

tfrecord_train_path = os.path.join(output_dir, tfrecord_train_filename)
tfrecord_val_path = os.path.join(output_dir, tfrecord_val_filename)
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

    elif normalization_type == 'partial-normalization':
        # 특정 열에 대해 정규화와 표준화 분리
        # ['volume','open', 'high', 'low', 'close', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'SMA5', 'SMA20', 'SMA50', 'SMA144', 'EMA5','EMA20', 'EMA50', 'EMA144', 
        #  { 0}       {1       2       3          4         5             6      7         8        9       10       11        12     13       14       15 
        #'MACD', 'MACD_signal','MACD_diff', 'RSI6','RSI12','RSI24', 'ADX','SAR', 'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI', 'OBV', 'Chaikin_Osc', 'Momentum', 
        # { 16         17            18  } {  19     20      21}      22    23     24         25          26           27      28     29           30        
        # 'ROC', 'ATR', 'STDDEV', 'VWAP', 'Pivot', 'Resistance1', 'Support1']
        #  31     32      33       34      35        36             37
        #() :min-max, []:standard,  {}:zero center  , ||:change-rate,
        normalized_features = np.zeros_like(features)
        
        max_abs_value = np.max(np.abs(features[:, 0]))  # 최대 절대값
        normalized_features[:, 0] = features[:, 0] / (max_abs_value +1e-8) 

        max_abs_value = np.max(np.abs(features[:, 1:16]))  # 최대 절대값
        normalized_features[:, 1:16] = features[:, 1:16] / (max_abs_value +1e-8) 

        max_abs_value = np.max(np.abs(features[:,16:19]))  # 최대 절대값
        normalized_features[:, 16:19] = features[:, 16:19] / (max_abs_value +1e-8) 

        max_abs_value = np.max(np.abs(features[:, 19:22]))  # 최대 절대값
        normalized_features[:, 19:22] = features[:, 19:22] / (max_abs_value +1e-8) 

        max_abs_value = np.max(np.abs(features[:, 22]))  # 최대 절대값
        normalized_features[:, 22] = features[:, 22] / (max_abs_value +1e-8) 


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
        
        feature_min = np.min(sequence[:, 0])
        feature_max = np.max(sequence[:, 0])
        normalized_sequence[:, 0] = (sequence[:, 0] - feature_min) / (feature_max - feature_min + 1e-8)  
        
        feature_min = np.min(sequence[:, 1:16])
        feature_max = np.max(sequence[:, 1:16])
        normalized_sequence[:, 1:16] = (sequence[:, 1:16] - feature_min) / (feature_max - feature_min + 1e-8)  

        feature_min = np.min(sequence[:, 16:19])
        feature_max = np.max(sequence[:, 16:19])
        normalized_sequence[:, 16:19] = 2*((sequence[:, 16:19] - feature_min) / (feature_max - feature_min + 1e-8)) -1  

        feature_min = np.min(sequence[:, 19:22])
        feature_max = np.max(sequence[:, 19:22])
        normalized_sequence[:, 19:22] = (sequence[:, 19:22] - feature_min) / (feature_max - feature_min + 1e-8)  

        feature_min = np.min(sequence[:, 22])
        feature_max = np.max(sequence[:, 22])
        normalized_sequence[:, 22] = (sequence[:, 22] - feature_min) / (feature_max - feature_min + 1e-8)            

        # fixedScale_columns = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
        # for col in fixedScale_columns:
        #     max_abs_value = np.max(np.abs(sequence[:, col]))  # 최대 절대값
        #     normalized_sequence[:, col] = sequence[:, col] / max_abs_value 

    elif normalization_type == 'change-rate':
        # Change-rate Normalization: 첫 번째 타임스텝 기준 등락률 계산
        normalized_sequence = (sequence / sequence[0]) - 1
        # normalized_sequence = (sequence[:,col] / sequence[0,col]) - 1

    elif normalization_type == 'none':
        # 정규화를 하지 않음
        normalized_sequence = sequence

    else:
        raise ValueError(f"Unsupported normalization type: {normalization_type}. Use 'min-max', 'standard', 'partial-norstand', 'change-rate', or 'none'.")

    return normalized_sequence
#==============================================[ 실행 코드 ]===================================================================================
# 데이터 처리 및 TFRecord 저장

# 데이터 처리 및 TFRecord 저장
with tf.io.TFRecordWriter(tfrecord_train_path) as train_writer, tf.io.TFRecordWriter(tfrecord_val_path) as val_writer:
    exchange_info = client.get_exchange_info()
    iteration = 1
    error_count =0
    for s in exchange_info['symbols']:
        print(f'{iteration}.Processing {s["symbol"]}:')
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
        features = data[:, : feature_dim]  # 모든 열 중 마지막 제외
        labels = data[:, -1].astype(int)  # 마지막 열 (정수형 변환)
        features = normalize_features(features, normalization_type=all_normalization)  # 전체 데이터 정규화

        # 데이터셋 분할 (80% Train, 20% Validation)
        split_idx = int(len(features) * 0.8)
        train_features, val_features = features[:split_idx], features[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]

        # 훈련 데이터 TFRecord 저장
        for i in range(len(train_features) - sequence_length + 1):
            try:
                sequence = train_features[i:i + sequence_length]
                sequence = normalize_sequence(sequence, normalization_type=sequence_normalization)
                sequence = sequence.flatten()
                label = train_labels[i + sequence_length - 1]
                example = serialize_example(sequence, label)
                train_writer.write(example)
            except Exception as e:
                print(f'Error processing record for {symbol} at index {i}: {e}')
                error_count += 1
                continue

        # 검증 데이터 TFRecord 저장
        for i in range(len(val_features) - sequence_length + 1):
            try:
                sequence = val_features[i:i + sequence_length]
                sequence = normalize_sequence(sequence, normalization_type=sequence_normalization)
                sequence = sequence.flatten()
                label = val_labels[i + sequence_length - 1]
                example = serialize_example(sequence, label)
                val_writer.write(example)
            except Exception as e:
                print(f'Error processing record for {symbol} at index {i}: {e}')
                error_count += 1
                continue

        print(f'-> Successfully processed and saved {symbol}.txt to TFRecord (Train + Validation).')
        iteration += 1

print(f'Error count: {error_count}')
print(f'Training TFRecord saved to {tfrecord_train_path}')
print(f'Validation TFRecord saved to {tfrecord_val_path}')