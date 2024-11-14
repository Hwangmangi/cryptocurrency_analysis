import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 데이터셋 경로
dataset_path = "C:\\code\\python\\autohunting\\dataset"
tfrecord_save_path = "C:\\code\\python\\autohunting\\dataset_TFrecord"
os.makedirs(tfrecord_save_path, exist_ok=True)

# 시퀀스 길이 설정
window_size = 30

def load_and_preprocess_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def create_sequences(data, window_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        seq = data.iloc[i:i + window_size].drop(columns=['label']).values
        label = data.iloc[i + window_size - 1]['label']
        sequences.append(seq)
        labels.append(label)
    return sequences, labels

def build_dataset(file_paths, window_size):
    all_sequences = []
    all_labels = []
    for file_path in file_paths:
        data = load_and_preprocess_csv(file_path)
        sequences, labels = create_sequences(data, window_size)
        all_sequences.extend(sequences)
        all_labels.extend(labels)
    return all_sequences, all_labels

# 모든 CSV 파일 가져오기
csv_files = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith('.csv')]

# 시퀀스 생성
sequences, labels = build_dataset(csv_files, window_size)

# 넘파이 배열로 변환
sequences = tf.constant(sequences, dtype=tf.float32)
labels = tf.constant(labels, dtype=tf.float32)

# 데이터셋 분리 (8:2 비율)
train_sequences, val_sequences, train_labels, val_labels = train_test_split(sequences, labels, test_size=0.2, random_state=42)

def serialize_example(sequence, label):
    feature = {
        'sequence': tf.train.Feature(float_list=tf.train.FloatList(value=sequence.flatten())),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecord(filename, sequences, labels):
    with tf.io.TFRecordWriter(filename) as writer:
        for seq, lbl in zip(sequences, labels):
            example = serialize_example(seq, lbl)
            writer.write(example)

# TFRecord 파일 저장
write_tfrecord(os.path.join(tfrecord_save_path, "train.tfrecord"), train_sequences, train_labels)
write_tfrecord(os.path.join(tfrecord_save_path, "val.tfrecord"), val_sequences, val_labels)

print("데이터가 TFRecord 파일로 저장되었습니다.")
