import tensorflow as tf
import os
import numpy as np
import pykrx as pk
from pykrx import stock
from pykrx import bond
import pandas as pd
import datetime
import matplotlib.pyplot as plt


print('-------------------------------------------------------------')
if 'VIRTUAL_ENV' in os.environ:
    print("1. 현재 가상 환경에서 실행 중입니다.")
else:
    print("1. 현재 가상 환경이 아닌 환경에서 실행 중입니다.")

print('2. 현재 사용중인 tensorflow 버젼 : ',tf.__version__)

print('2. 현재 gpu사용이 가능한지 : ',tf.config.list_physical_devices('GPU'))
print('--------------------------------------------------------------')
 

#-------------------------주요 파라미터--------------------------------------------------------------------------
now = datetime.datetime.now()
formatted_time = now.strftime("%Y%m%d_%H%M%S")

tf.keras.mixed_precision.set_global_policy('float32')
lr=0.001

# 시퀀스 길이와 feature 수 (시퀀스 길이와 feature 수는 저장할 때와 동일해야 함)
window_size = 30
num_features = 12  # feature 수는 데이터에 따라 조정

def parse_tfrecord(example_proto):
    feature_description = {
        'sequence': tf.io.FixedLenFeature([window_size * num_features], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.float32),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    sequence = tf.reshape(parsed_example['sequence'], (window_size, num_features))
    label = parsed_example['label']
    return sequence, label

def load_tfrecord_dataset(filename, batch_size=32):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# 파일 경로 설정
train_tfrecord_path = "C:\\code\\python\\autohunting\\dataset_TFrecord\\train.tfrecord"
val_tfrecord_path = "C:\\code\\python\\autohunting\\dataset_TFrecord\\val.tfrecord"

# TFRecord 파일 로드
train_dataset = load_tfrecord_dataset(train_tfrecord_path)
val_dataset = load_tfrecord_dataset(val_tfrecord_path)

print("TFRecord 파일에서 데이터셋이 로드되었습니다.")

#----------------------------------------------------------------------------------------------------------------

sequence_length = 30
feature_length = 12 #IRQ3은 등락률 특성도 학습에 포함시킨다 inputs = tf.keras.layers.Input(shape=(sequence_length, feature_length))  # 기존 입력 형태 (배치 크기 제외)

class NormalizeByFirstRow(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-10, **kwargs):
        super(NormalizeByFirstRow, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        first_row = inputs[:, 0, :]  # shape: (batch_size, feature_length)
        first_row = tf.maximum(first_row, self.epsilon)  # 작은 값으로 대체
        normalized = inputs / tf.expand_dims(first_row, axis=1)
        return normalized


inputs = tf.keras.layers.Input(shape=(sequence_length, feature_length))  # 기존 입력 형태 (배치 크기 제외)
normalized = NormalizeByFirstRow()(inputs)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(d1) 
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

print(model.summary())

#====================================================================


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.7, 
    patience=7, 
    min_lr=0.00001,
    verbose=1
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  
    patience=20,  
    mode='min',  
    verbose=1 
)

# ModelCheckpoint 
savefile=f'C:\code\python\stock\keras_model\{formatted_time}_CallbackSave'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=savefile,  
    monitor='val_loss',  
    mode='min',  
    save_best_only=True, 
    verbose=1  
      )
#--------------------------------------------------------------------------------------------------------------------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
#optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
#optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
#optimizer = tf.keras.optimizers.Adagrad (learning_rate=lr)
#optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr)
#loss=tf.keras.losses.SparseCategoricalCrossentropy()
loss=tf.keras.losses.BinaryCrossentropy()
metric=tf.keras.metrics.Accuracy()
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#--------------------------------------------------------------------------------------------------------------------------------------

print("batch, lr, conv_l1, conv_l2, lstm_l1,lstm_l2, dense_l1, dense_l2: ",batch_size,lr)


history=model.fit(train_dataset,
         batch_size=batch_size,
            validation_batch_size=batch_size,
           epochs=5000, 
           callbacks=[checkpoint_callback,reduce_lr,early_stopping_callback],
           validation_data=val_dataset)
os.chdir(r'C:\code\python\stock\keras_model')
model.save(f'{formatted_time}_EarlystoppingSave')



 #--------------------------------------------------------------------------------------------------------------------------------------
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation MAE")
plt.legend()
plt.show() 
#--------------------------------------------------------------------------------------------------------------------------------------
