import os
import numpy as np
import tensorflow as tf

#============================================[ 사용자 설정 파라미터 ]==================================================================================
batch = 128
lr = 0.001

#===========================================[ 주요 파라미터 ]============================================================================

data_path = r'C:\code\python\autohunting\dataset_raw_1hour22feature'
output_dir = r'C:\code\python\autohunting\dataset_TFRecord'
tfrecord_filename = 'TFRecord.1hour22feature.tfrecord'
tfrecord_path = os.path.join(output_dir, tfrecord_filename)
sequence_length = 30  # 시퀀스 길이
feature_dim = 22  # 한 샘플의 특성 수 (레이블 제외)
#===========================================[ 주요 함수 ]============================================================================
def parse_function(proto):
    # TFRecord 데이터에서 사용할 특성 정의
    features = {
        'features': tf.io.FixedLenFeature([sequence_length * feature_dim], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(proto, features)
    features_reshaped = tf.reshape(parsed_features['features'], [sequence_length, feature_dim])

    return features_reshaped, parsed_features['label']


# 모델 정의: Transformer 기반
def create_transformer_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Transformer 블록
    x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-Forward Network
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # 출력층
    output = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    
    return model

# 모델 컴파일 및 학습
def train_model():
    # TFRecordDataset 로드
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    # 데이터 파싱
    dataset = raw_dataset.map(parse_function)

    # 전체 데이터셋 크기 계산
    total_samples = sum(1 for _ in dataset)

    # 훈련 데이터와 검증 데이터 비율 설정
    train_size = int(0.8 * total_samples)  # 80% 훈련 데이터
    val_size = total_samples - train_size  # 나머지는 검증 데이터

    # 데이터셋 셔플
    dataset = dataset.shuffle(buffer_size=total_samples, reshuffle_each_iteration=True)

    # 훈련 데이터와 검증 데이터 나누기
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    # 배치 처리
    batch_size = batch
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    # 데이터셋의 prefetch 사용
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # 모델 생성
    model = create_transformer_model(input_shape=(sequence_length, feature_dim), num_classes=1)  # 이진 분류
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # EarlyStopping 콜백
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    print(model.summary())
    # 모델 학습
    model.fit(train_dataset, epochs=20, validation_data=val_dataset, callbacks=[early_stopping])
#==========================================[ 실행 코드 ]=====================================================================================
train_model()
