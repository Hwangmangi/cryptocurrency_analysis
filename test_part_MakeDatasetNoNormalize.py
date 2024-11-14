import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 데이터셋 경로 설정
dataset_path = "C:\\code\\python\\autohunting\\dataset"
tfrecord_save_path = "C:\\code\\python\\autohunting\\dataset_TFrecord"
os.makedirs(tfrecord_save_path, exist_ok=True)

# 시퀀스 길이 설정 (사용자가 지정)
window_size = 30

def load_and_preprocess_csv(file_path):
    """CSV 파일을 로드하고 데이터프레임으로 변환."""
    df = pd.read_csv(file_path)
    print(f"[INFO] 파일 로드 완료: {file_path}, 데이터 크기: {df.shape}")
    return df

def create_sequences(data, window_size, filename):
    """시퀀스를 생성하고 마지막 행의 label을 타겟으로 설정."""
    sequences = []
    labels = []
    sources = []
    for i in range(len(data) - window_size):
        seq = data.iloc[i:i + window_size].drop(columns=['label']).values
        label = data.iloc[i + window_size - 1]['label']
        sequences.append(seq)
        labels.append(label)
        sources.append(filename)
    return sequences, labels, sources

def build_dataset(file_paths, window_size):
    """여러 CSV 파일로부터 시퀀스를 생성."""
    all_sequences = []
    all_labels = []
    all_sources = []
    for file_path in file_paths:
        data = load_and_preprocess_csv(file_path)
        sequences, labels, sources = create_sequences(data, window_size, os.path.basename(file_path))
        
        # 디버그 출력: 시퀀스와 레이블 일부 출력
        if sequences:
            print(f"[DEBUG] 첫 시퀀스 데이터 예시: {sequences[0][:5]}")
            print(f"[DEBUG] 첫 시퀀스 레이블: {labels[0]}, 출처: {sources[0]}")
        
        all_sequences.extend(sequences)
        all_labels.extend(labels)
        all_sources.extend(sources)
    
    print(f"[INFO] 전체 데이터셋 크기: {len(all_sequences)} 시퀀스, {len(all_labels)} 레이블")
    return all_sequences, all_labels, all_sources

# CSV 파일 목록 가져오기
csv_files = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith('.csv')]

# 시퀀스 생성 및 디버그 출력
sequences, labels, sources = build_dataset(csv_files, window_size)

# 넘파이 배열로 변환
sequences = tf.constant(sequences, dtype=tf.float32)
labels = tf.constant(labels, dtype=tf.float32)
sources = tf.constant(sources)

# 데이터셋 분리 (8:2 비율)
train_sequences, val_sequences, train_labels, val_labels, train_sources, val_sources = train_test_split(
    sequences, labels, sources, test_size=0.2, random_state=42)

print(f"[INFO] 훈련 데이터셋 크기: {train_sequences.shape}, 검증 데이터셋 크기: {val_sequences.shape}")

def serialize_example(sequence, label, source):
    """시퀀스와 라벨, 파일명을 TFRecord 형식으로 직렬화."""
    feature = {
        'sequence': tf.train.Feature(float_list=tf.train.FloatList(value=sequence.flatten())),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
        'source': tf.train.Feature(bytes_list=tf.train.BytesList(value=[source.numpy()]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecord(filename, sequences, labels, sources):
    """TFRecord 파일로 시퀀스 데이터 저장."""
    with tf.io.TFRecordWriter(filename) as writer:
        for idx, (seq, lbl, src) in enumerate(zip(sequences, labels, sources)):
            example = serialize_example(seq, lbl, src)
            writer.write(example)
            
            # 디버그: 저장된 데이터 일부 확인
            if idx < 3:  # 첫 3개의 데이터만 확인
                print(f"[DEBUG] 저장된 시퀀스 {idx}: {seq.numpy().flatten()[:5]}...")
                print(f"[DEBUG] 저장된 레이블 {idx}: {lbl.numpy()}, 출처: {src.numpy().decode()}")

# TFRecord 파일 저장
train_tfrecord_path = os.path.join(tfrecord_save_path, "train.tfrecord")
val_tfrecord_path = os.path.join(tfrecord_save_path, "val.tfrecord")

write_tfrecord(train_tfrecord_path, train_sequences, train_labels, train_sources)
write_tfrecord(val_tfrecord_path, val_sequences, val_labels, val_sources)
print(f"[INFO] TFRecord 저장 완료: {train_tfrecord_path}, {val_tfrecord_path}")

def parse_tfrecord(example_proto):
    """TFRecord 파일에서 데이터를 파싱."""
    feature_description = {
        'sequence': tf.io.FixedLenFeature([window_size * train_sequences.shape[2]], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.float32),
        'source': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    sequence = tf.reshape(parsed_example['sequence'], (window_size, -1))
    label = parsed_example['label']
    source = parsed_example['source']
    return sequence, label, source

def load_tfrecord_dataset(filename, batch_size=32):
    """TFRecord 파일을 로드하여 Dataset 생성."""
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# TFRecord 파일 로드 및 확인
train_dataset = load_tfrecord_dataset(train_tfrecord_path)
val_dataset = load_tfrecord_dataset(val_tfrecord_path)

print("[INFO] TFRecord 데이터셋 로드 완료")

# 디버그: 로드된 데이터 확인
for seq, lbl, src in train_dataset.take(1):
    print(f"[DEBUG] 로드된 시퀀스 shape: {seq.shape}")
    print(f"[DEBUG] 로드된 시퀀스 데이터: {seq[0, :5]}")
    print(f"[DEBUG] 로드된 레이블: {lbl[0]}, 출처: {src[0].numpy().decode()}")
