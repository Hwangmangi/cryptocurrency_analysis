
#==============================================레이블링 관련 함수====================================================================
#이진 레이블링 (상승/하락)
def label_binary(df, horizon=1):
    """
    현재 가격 대비 horizon 시간 후 가격 변화에 따라 상승(1), 하락(0) 레이블 생성
    
    Args:
        df (pd.DataFrame): 코인 데이터 프레임, 'close' 열이 필요함
        horizon (int): 레이블링을 위한 시간 차이 (시간 단위)
    
    Returns:
        pd.DataFrame: 상승/하락 레이블이 추가된 데이터 프레임
    """
    df['future_price'] = df['close'].shift(-horizon)  # horizon 시간 후 가격
    df['label'] = (df['future_price'] > df['close']).astype(int)
    df.dropna(inplace=True)  # future_price의 NaN 제거
    return df[['close', 'label']]


#이진 레이블링 (lookahead_hours구간 동안 수익실현지점이 있었다면 1, 아니라면 0)
def label_profit_possible(df, lookahead_hours=5):
    """
    시퀀스가 끝난 후, 주어진 시간 구간(lookahead_hours) 내에 
    종가보다 더 높은 가격을 기록한 시점이 있었는지 여부로 레이블을 붙여주는 함수.
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임 (시간별 OHLCV + 기술적 지표 포함)
        lookahead_hours (int): 수익 실현 가능 여부를 확인할 시간 구간 (예: 5)
        
    Returns:
        pd.DataFrame: 'label' 컬럼이 추가된 데이터프레임
    """
    df = df.copy()  # 원본을 변경하지 않기 위해 복사본 사용

    # 'label' 열을 기본값 NaN으로 초기화 (기본은 수익 실현 불가능)
    df['label'] = pd.NA

    # 시퀀스 끝에서 이후의 가격을 기준으로 수익 실현 여부를 체크
    for i in range(len(df) - lookahead_hours):
        start_price = df['close'].iloc[i]  # 시퀀스 끝의 가격 (현재 가격)
        
        # lookahead_hours 기간 동안 종가 확인
        future_prices = df['close'].iloc[i + 1:i + lookahead_hours + 1]

        # 수익 실현 가능 여부 판단: lookahead_hours 내에 더 높은 가격이 있었다면 수익 실현 가능
        if any(future_prices > start_price):
            df.loc[i, 'label'] = 1  # 수익 실현 가능 -> 1로 레이블링
        else:
            df.loc[i, 'label'] = 0  # 수익 실현 불가능 -> 0으로 레이블링
    df.dropna(subset=['label'], inplace=True)
    return df
    #return df[['close', 'label']]  # 'close'와 'label'만 반환


class NormalizeByFirstRow(tf.keras.layers.Layer):  # 단순 나누기 연산 : 현재값 / 첫 타입스텝값
    def __init__(self, epsilon=1e-10, **kwargs):
        super(NormalizeByFirstRow, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        first_row = inputs[:, 0, :]  # shape: (batch_size, feature_length)
        first_row = tf.maximum(first_row, self.epsilon)  # 작은 값으로 대체
        normalized = inputs / tf.expand_dims(first_row, axis=1)
        return normalized





class NormalizeByFirstStep(tf.keras.layers.Layer):  # 변화율 연산 : (현재값 - 첫 타입스텝값) / 첫 타임스텝값
    def __init__(self, epsilon=1e-10, **kwargs):
        """
        첫 번째 타임스텝을 기준으로 시퀀스 데이터의 변화율을 계산하는 레이어.
        
        Args:
            epsilon (float): 작은 값으로 0으로 나누는 오류를 방지.
        """
        super(NormalizeByFirstStep, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        """
        시퀀스 데이터에서 첫 번째 타임스텝을 기준으로 변화율을 계산.

        Args:
            inputs (tensor): (batch_size, time_steps, features) 형태의 시퀀스 데이터

        Returns:
            tensor: 변화율로 정규화된 (batch_size, time_steps, features) 텐서
        """
        # 첫 번째 타임스텝의 값을 가져옴 (배치 크기, 피처 수에 대해 첫 번째 시퀀스 값)
        first_step_values = inputs[:, 0, :]  # shape: (batch_size, features)
        
        # 작은 값으로 나누는 오류 방지
        first_step_values = tf.maximum(first_step_values, self.epsilon)

        # 각 시퀀스 샘플의 첫 번째 타임스텝을 기준으로 변화율 계산
        normalized = (inputs - tf.expand_dims(first_step_values, axis=1)) / tf.expand_dims(first_step_values, axis=1)

        return normalized
