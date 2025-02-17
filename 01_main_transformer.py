import os
import numpy as np
import tensorflow as tf

#============================================[ 사용자 설정 파라미터 ]==================================================================================
batch = 64
lr = 0.001
#===========================================[ 주요 파라미터 ]============================================================================
output_dir = r'C:\code\python\autohunting\dataset_TFRecord'
train_tfrecord_filename = '1day50seq38feature_TRAIN.tfrecord'
val_tfrecord_filename = '1day50seq38feature_VAL.tfrecord'

# TFRecord 파일 경로
train_tfrecord_path = os.path.join(output_dir, train_tfrecord_filename)
val_tfrecord_path = os.path.join(output_dir, val_tfrecord_filename)

sequence_length = 50  # 시퀀스 길이
feature_dim = 38  # 한 샘플의 특성 수 (레이블 제외)
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

def positional_encoding(sequence_length, d_model):
    positions = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]  # (30, 1)의 shape
    dimensions = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]  # (1, d_model)의 shape

    angle_rates = 1 / tf.pow(10000.0, (2 * (dimensions // 2)) / tf.cast(d_model, tf.float32))  # (1, d_model)의 shape
    angle_rads = tf.cast(positions, tf.float32) * angle_rates   #(30,22)의 shape  
    # [[0.00, 0.00, 0.000],
    #  [1.00, 1.00, 0.010],
    #  [2.00, 2.00, 0.020],
    #  [3.00, 3.00, 0.030]]  -> angle_rads
    sines = tf.sin(angle_rads[:, 0::2]) # 짝수 인덱스에서 사인값을 계산 
    cosines = tf.cos(angle_rads[:, 1::2]) # 홀수 인덱스에서 코사인값을 계산

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    # [[[0.0, 1.0, 0.0],
    #  [0.841, 0.540, 0.010],
    #  [0.909, 0.415, 0.020],
    #  [0.141, 0.989, 0.030]]] -> pos_encoding
    return pos_encoding
class PositionEncoding(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-10, **kwargs):
        super(PositionEncoding, self).__init__(**kwargs)
        self.epsilon = epsilon
    def get_config(self):
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
        })
        return config
    def call(self, inputs):
        sequence_length = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        pos_encoding = positional_encoding(sequence_length, d_model)
        # 1. 내 좆대로 스케일링
        # scaled_pos_encoding = pos_encoding
        # scaled_pos_encoding = pos_encoding * 0.1
        # scaled_pos_encoding = pos_encoding * 0.01
        # 2.표준편차로 스케일링
        input_std = tf.math.reduce_std(inputs)
        scaled_pos_encoding = pos_encoding * input_std # input_std는 1에 가까운값
        x = inputs + scaled_pos_encoding
        return x


# Transformer 인코더 레이어
def transformer_encoder(inputs):
    # 포지셔널 인코딩 추가
    sequence_length = tf.shape(inputs)[1]
    d_model = tf.shape(inputs)[2]
    pos_encoding = positional_encoding(sequence_length, d_model)
    # 위치값 너무 크면 안됨 1
    scaled_pos_encoding = pos_encoding * 0.01
    # # 위치값 너무 크면 안됨2
    # input_std = tf.math.reduce_std(inputs)
    # scaled_pos_encoding = pos_encoding * input_std # input_std는 1에 가까운값
    x = inputs + scaled_pos_encoding

    return x

def transformer_lstm_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = PositionEncoding()(inputs)
    attn_output1 = tf.keras.layers.MultiHeadAttention(key_dim=512, num_heads=16)(x, x)
    attn_output2 = tf.keras.layers.LayerNormalization(epsilon=1e-8)(attn_output1 + x)  # 잔차 연결
    x_ff1 = tf.keras.layers.Dense(512, activation='tanh')(attn_output2) 
    x_ff2 = tf.keras.layers.Dense(38, activation='tanh')(x_ff1) 
    dense_out = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x_ff2 + attn_output1 + x)  # 잔차 연결


    f1 = tf.keras.layers.Flatten()(dense_out)
    f2 = tf.keras.layers.Dense(512, activation='tanh')(f1)
    f3 = tf.keras.layers.Dropout(0.3)(f2)
    f4 = tf.keras.layers.Dense(256, activation='tanh')(f3)
    f5 = tf.keras.layers.Dropout(0.3)(f4)
    f6 = tf.keras.layers.Dense(128, activation='tanh')(f5)
    f7 = tf.keras.layers.Dropout(0.3)(f6)
    f_out = tf.keras.layers.Dense(64, activation='tanh')(f7)
    # lstm_out = tf.keras.layers.GlobalMaxPooling1D()(dense_out)  # 시퀀스 차원을 축소
    output = tf.keras.layers.Dense(1, activation='sigmoid')(f_out)

    return tf.keras.models.Model(inputs, output)

def lstm_transformer_model(input_shape):



    inputs = tf.keras.layers.Input(shape=input_shape)

    # LSTM 계층
    x1 = tf.keras.layers.LSTM(23, return_sequences=True)(inputs)
    lstm_out = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x1)


    # Transformer 인코더 블록
    # attn_output = tf.keras.layers.MultiHeadAttention(key_dim=64, num_heads=4, dropout=0.1)(x, x)
    attn_output1 = tf.keras.layers.MultiHeadAttention(key_dim=64, num_heads=4)(lstm_out, lstm_out)
    # attn_output = tf.keras.layers.Dropout(0.3)(attn_output)
    attn_output2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attn_output1 + lstm_out)  # 잔차 연결

    # 포인트 와이즈 피드포워드 네트워크
    x_ff1 = tf.keras.layers.Dense(128, activation='relu')(attn_output2)
    x_ff2 = tf.keras.layers.Dense(23, activation='relu')(x_ff1)
    # x_ff = tf.keras.layers.Dropout(0.3)(x_ff)
    dense_out = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x_ff2 + attn_output1 + lstm_out)  # 잔차 연결

    # 출력 계층
    # x = tf.keras.layers.GlobalMaxPooling1D()(dense_out)  # 시퀀스 차원을 축소
    lstm_out2 = tf.keras.layers.LSTM(64)(dense_out)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(lstm_out2)

    return tf.keras.models.Model(inputs, outputs)

def lstm_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # LSTM layer
    # 첫 번째 LSTM 층
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)

    # 두 번째 LSTM 층
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)

    # 세 번째 LSTM 층
    x = tf.keras.layers.LSTM(256, return_sequences=True)(x)

    # 네 번째 LSTM 층
    x = tf.keras.layers.LSTM(128)(x)  # return_sequences=False로 설정하여 최종 출력만 반환

    # Classification head
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.models.Model(inputs, outputs)
    # TFRecordDataset 로드 및 파싱 함수
def load_tfrecord_dataset(tfrecord_path, batch_size):
    # 데이터 로드
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    # 데이터 파싱
    dataset = raw_dataset.map(parse_function)

    # 셔플, 배치, prefetch 처리
    # dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
# 모델 컴파일 및 학습
def train_model():
    # 훈련 및 검증 데이터셋 로드
    print('Loading training dataset...')
    train_dataset = load_tfrecord_dataset(train_tfrecord_path, batch_size=batch)

    print('Loading validation dataset...')
    val_dataset = load_tfrecord_dataset(val_tfrecord_path, batch_size=batch)

    # 모델 생성
    model = transformer_lstm_model(input_shape=(sequence_length, feature_dim))  # 이진 분류
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # EarlyStopping 콜백
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    
    # ModelCheckpoint 콜백
    checkpoint_path = r"C:\code\python\autohunting\model\model_checkpoin19.h5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    
    print(model.summary())
    print(f'callback path: {checkpoint_path}')
    # 모델 학습
    model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[early_stopping, model_checkpoint])
    
    # 학습이 끝난 후 모델 저장
    final_model_path = r"C:\code\python\autohunting\model\final_model18.h5"
    model.save(final_model_path)
# 저장된 모델을 불러와서 재학습시키는 함수



def retrain_model(saved_model_path):
    # 훈련 및 검증 데이터셋 로드
    print('Loading training dataset...')
    train_dataset = load_tfrecord_dataset(train_tfrecord_path, batch_size=batch)

    print('Loading validation dataset...')
    val_dataset = load_tfrecord_dataset(val_tfrecord_path, batch_size=batch)

    # 저장된 모델 로드
    custom_objects = {'PositionEncoding': PositionEncoding}
    model = tf.keras.models.load_model(saved_model_path, custom_objects=custom_objects)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # EarlyStopping 콜백
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    
    # ModelCheckpoint 콜백
    checkpoint_path = r"C:\code\python\autohunting\model\model_checkpoint18.h5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    
    print(model.summary())
    # 모델 재학습
    model.fit(train_dataset, epochs=500, validation_data=val_dataset, callbacks=[early_stopping, model_checkpoint])
    
    # 재학습이 끝난 후 모델 저장
    final_model_path = r"C:\code\python\autohunting\model\final_model18.h5"
    model.save(final_model_path)
# ==========================================[ 실행 코드 ]=====================================================================================
train_model()
#retrain_model(r"C:\code\python\autohunting\model\model_checkpoin18.h5")
