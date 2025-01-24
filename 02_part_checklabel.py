import tensorflow as tf
import os  
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

def count_labels(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_function)

    label_counts = {0: 0, 1: 0}
    count = 0
    for _, label in dataset:
        label_value = label.numpy()
        if label_value in label_counts:
            label_counts[label_value] += 1
        else:
            label_counts[label_value] = 1
        print("Count: ", count)
        count += 1

    return label_counts

# TFRecord 파일 경로
tfrecord_train_path = r'C:\code\python\autohunting\dataset_TFRecord\1day50seq38feature_TRAIN.tfrecord'
tfrecord_val_path = r'C:\code\python\autohunting\dataset_TFRecord\1day50seq38feature_VAL.tfrecord'

# 라벨 데이터 분포 확인
train_label_counts = count_labels(tfrecord_train_path)
val_label_counts = count_labels(tfrecord_val_path)

print("Training label distribution:", train_label_counts)
print("Validation label distribution:", val_label_counts)