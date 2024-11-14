import tensorflow as tf
import pandas as pd
import glob

def create_dataset_from_csv_files(file_path_pattern, sequence_length, batch_size, num_parallel_reads=tf.data.AUTOTUNE):
    file_list = tf.data.Dataset.list_files(file_path_pattern, shuffle=True)

    def load_and_preprocess_data(file_path):
        df = pd.read_csv(file_path.numpy().decode('utf-8'))
        
        features = df[['open', 'high', 'low', 'close', 'volumem', 'MACD', 
                       'MACD_signal', 'RSI', 'Upper_BB', 'Middle_BB', 
                       'Lower_BB', 'ADX']].values
        label = df['label'].iloc[-1]

        sequences = []
        labels = []
        for i in range(len(features) - sequence_length + 1):
            sequences.append(features[i: i + sequence_length])
            labels.append(label)
        
        return tf.convert_to_tensor(sequences, dtype=tf.float32), tf.convert_to_tensor(labels, dtype=tf.float32)

    def tf_load_and_preprocess(file_path):
        sequences, labels = tf.py_function(load_and_preprocess_data, [file_path], [tf.float32, tf.float32])
        sequences.set_shape((None, sequence_length, 12))
        labels.set_shape((None,))
        return sequences, labels

    dataset = file_list.interleave(
        lambda file_path: tf.data.Dataset.from_tensor_slices(tf_load_and_preprocess(file_path)),
        cycle_length=num_parallel_reads,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
def create_lstm_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # 이진 분류를 위한 sigmoid 활성화 함수
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 모델 생성

# 데이터셋 및 모델 설정
file_path_pattern = "C:/code/python/autohunting/dataset/*.csv"
sequence_length = 30
batch_size = 32
dataset = create_dataset_from_csv_files(file_path_pattern, sequence_length, batch_size)

input_shape = (sequence_length, 12)
model = create_lstm_model(input_shape)
model.summary()
# 모델 학습
epochs = 10
model.fit(dataset, epochs=epochs)
