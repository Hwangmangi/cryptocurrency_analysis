import tensorflow as tf

def positional_encoding(sequence_length, d_model):
    positions = tf.range(sequence_length)[:, tf.newaxis] #(30,1)의 shape
    dimensions = tf.range(d_model)[tf.newaxis, :] # (1,22)의 shape

    angle_rates = 1 / tf.pow(10000.0,(2 * tf.cast(dimensions // 2, tf.float32)) / tf.cast(d_model, tf.float32))  # (1, 22)의 shape
    angle_rads = tf.cast(positions, tf.float32) * angle_rates   #(30,22)의 shape

    sines = tf.sin(angle_rads[:, 0::2]) # 짝수 인덱스에서 사인값을 계산 
    cosines = tf.cos(angle_rads[:, 1::2]) # 홀수 인덱스에서 코사인값을 계산

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]

    return pos_encoding

# Transformer 인코더 레이어
def transformer_encoder(inputs):
    # 포지셔널 인코딩 추가
    sequence_length = tf.shape(inputs)[1]
    d_model = tf.shape(inputs)[2]
    pos_encoding = positional_encoding(sequence_length, d_model)
    x = inputs + pos_encoding # (배치 크기, 시퀀스 길이, 특성 수)


    return x
class PositionEncoding(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-10, **kwargs):
        super(PositionEncoding, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
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

    # Transformer 인코더 블록
    # x = transformer_encoder(inputs)
    x = PositionEncoding()(inputs)
    # 멀티헤드 어텐션
    x = tf.keras.layers.MultiHeadAttention(key_dim=64, num_heads=4, dropout=0.1)(x, x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # 포인트 와이즈 피드포워드 네트워크
    x_ff = tf.keras.layers.Dense(128, activation='relu')(x)
    x_ff = tf.keras.layers.Dense(64)(x_ff)
    x = tf.keras.layers.Dropout(0.3)(x_ff)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # LSTM 계층
    x = tf.keras.layers.LSTM(128)(x)

    # 출력 계층
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.models.Model(inputs, x)



model = transformer_lstm_model((30, 22))
print(model.summary())