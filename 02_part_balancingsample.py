import tensorflow as tf
import numpy as np
import random

# 기존 TFRecord 파일 경로
tfrecord_train_path = r'C:\code\python\autohunting\dataset_TFRecord\1day50seq38feature_TRAIN.tfrecord'
tfrecord_val_path = r'C:\code\python\autohunting\dataset_TFRecord\1day50seq38feature_VAL.tfrecord'

# 새로운 TFRecord 파일 경로
tfrecord_train_path_new = r'C:\code\python\autohunting\dataset_TFRecord\1day50seq38feature_TRAIN2.tfrecord'
tfrecord_val_path_new = r'C:\code\python\autohunting\dataset_TFRecord\1day50seq38feature_VAL2.tfrecord'

sequence_length = 50
feature_dim = 38

def parse_function(proto):
    # TFRecord 데이터에서 사용할 특성 정의
    features = {
        'features': tf.io.FixedLenFeature([sequence_length * feature_dim], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(proto, features)
    features_reshaped = tf.reshape(parsed_features['features'], [sequence_length, feature_dim])
    return features_reshaped, parsed_features['label']

def serialize_example(features, label):
    feature = {
        'features': tf.train.Feature(float_list=tf.train.FloatList(value=features.flatten())),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def balance_dataset(tfrecord_path, tfrecord_path_new):
    # TFRecord 파일 읽기
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_function)

    # 라벨이 0인 데이터와 1인 데이터를 분리
    label_0_samples = []
    label_1_samples = []

    for features, label in dataset:
        if label.numpy() == 0:
            label_0_samples.append((features.numpy(), label.numpy()))
        else:
            label_1_samples.append((features.numpy(), label.numpy()))

    # 라벨이 1인 데이터 샘플을 랜덤하게 선택하여 라벨이 0인 데이터 샘플 수와 비슷하게 맞춤
    random.shuffle(label_1_samples)
    label_1_samples = label_1_samples[:len(label_0_samples)]

    # 새로운 TFRecord 파일에 저장
    with tf.io.TFRecordWriter(tfrecord_path_new) as writer:
        for features, label in label_0_samples + label_1_samples:
            example = serialize_example(features, label)
            writer.write(example)

    print(f"New TFRecord file saved at {tfrecord_path_new}")

# 훈련 데이터셋 균형 맞추기
balance_dataset(tfrecord_train_path, tfrecord_train_path_new)

# 검증 데이터셋 균형 맞추기
balance_dataset(tfrecord_val_path, tfrecord_val_path_new)