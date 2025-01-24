import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 경로
data_path = r'C:\code\python\autohunting\dataset_raw_2hour38feature'

# 등락률 저장 리스트
return_rates = []
positive_rate_ema50 = []
positive_rate_ema144 = []
negative_rate_ema50 = []
negative_rate_ema144 = []
# 데이터 파일 처리
for filename in os.listdir(data_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(data_path, filename)
        
        try:
            # 데이터 로드 (탭으로 구분된 텍스트 파일)
            data = pd.read_csv(file_path, delimiter='\t', header=None, names=[
                'volume', 'open', 'high', 'low', 'close', 
                'Upper_BB', 'Middle_BB', 'Lower_BB', 'SMA5', 'SMA20', 
                'SMA50', 'SMA144', 'EMA5', 'EMA20', 'EMA50', 'EMA144', 
                'MACD', 'MACD_signal', 'MACD_diff', 'RSI6', 'RSI12', 'RSI24', 
                'ADX', 'SAR', 'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI', 
                'OBV', 'Chaikin_Osc', 'Momentum', 'ROC', 'ATR', 'STDDEV', 
                'VWAP', 'Pivot', 'Resistance1', 'Support1', 'label'
            ])
            
            # MACD 차이값 (MACD - Signal)
            macd_diff = data['MACD_diff'].values
            # print(f'{filename}:macd_diff : {macd_diff.shape}{macd_diff[:50]}')
            close = data['close'].values
            ema50 = data['EMA50'].values
            ema144 = data['EMA144'].values
            
            # 구간 저장 리스트
            cross_periods = []
            
            # 첫 번째 골든크로스를 찾고 그 후에 데드크로스를 찾는 방식
            i = 0
            while i < len(macd_diff) - 1:
                # 골든크로스 찾기 (0 이하에서 0 초과로 전환)
                if macd_diff[i] <= 0 and macd_diff[i + 1] > 0:
                    golden_cross_index = i + 1
                    # 골든크로스가 발생하면, 그 다음 데드크로스를 찾음
                    for j in range(golden_cross_index + 1, len(macd_diff) - 1):
                        # 데드크로스 찾기 (0 이상에서 0 이하로 전환)
                        if macd_diff[j] >= 0 and macd_diff[j + 1] < 0:
                            dead_cross_index = j + 1
                            # 구간 저장
                            cross_periods.append((golden_cross_index, dead_cross_index))
                            
                            # 구간의 등락률 계산 (close 값 변화에 따라 등락률을 계산)
                            start_price = close[golden_cross_index]
                            end_price = close[dead_cross_index]
                            return_rate = (end_price - start_price) / start_price * 100  # 등락률 계산
                            if return_rate > 31 or return_rate < -31:
                                i = j + 1 
                                break
                            if(return_rate < 0):
                                gradient_ema50 = np.diff(ema50[golden_cross_index:dead_cross_index])
                                mean_gradient_ema50 = np.mean(gradient_ema50)
                                gradient_ema144 = np.diff(ema144[golden_cross_index:dead_cross_index])
                                mean_gradient_ema144 = np.mean(gradient_ema144)
                                negative_rate_ema50.append(mean_gradient_ema50)
                                negative_rate_ema144.append(mean_gradient_ema144)   
                            else:
                                gradient_ema50 = np.diff(ema50[golden_cross_index:dead_cross_index])
                                mean_gradient_ema50 = np.mean(gradient_ema50)
                                gradient_ema144 = np.diff(ema144[golden_cross_index:dead_cross_index])
                                mean_gradient_ema144 = np.mean(gradient_ema144)
                                positive_rate_ema50.append(mean_gradient_ema50)
                                positive_rate_ema144.append(mean_gradient_ema144)
                            # print(f"#{filename}Golden Cross: {golden_cross_index}, Dead Cross: {dead_cross_index}, Return Rate: {return_rate:.2f}%")
                            # print(f'골든크로스 시점.({macd_diff[golden_cross_index-1]})  ->  ({macd_diff[golden_cross_index]})\n데드크로스 시점.({macd_diff[dead_cross_index-1]})  ->  ({macd_diff[dead_cross_index]})')
                            # print(f"양수macd구간.{macd_diff[golden_cross_index]}({golden_cross_index})-> {macd_diff[dead_cross_index]}({dead_cross_index}): 가격 {start_price} -> {end_price}\n\n")
                            if return_rate > 1000:
                                print(f"{filename}Golden Cross: {golden_cross_index}, Dead Cross: {dead_cross_index}, Return Rate: {return_rate:.2f}%")
                                print(f'{macd_diff[golden_cross_index-1]}->{macd_diff[golden_cross_index]}\n{macd_diff[dead_cross_index-1]}->{macd_diff[dead_cross_index]}')
                                print(f"{macd_diff[golden_cross_index]}({golden_cross_index})-> {macd_diff[dead_cross_index]}({dead_cross_index}):{start_price} -> {end_price}\n\n")
                            return_rates.append(return_rate)
                            i = j + 1  # 데드크로스 이후 인덱스로 이동
                            break
                i += 1
            
    
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    # break
# 정상적인 범위의 등락률 필터링 (예: -70% ~ 70%)
filtered_return_rates = [rate for rate in return_rates if -30 <= rate <= 30]

# 필터링된 평균 등락률 계산
average_filtered_return_rate = np.mean(filtered_return_rates) if filtered_return_rates else 0
average_positive_rate_ema50 = np.mean(positive_rate_ema50) if positive_rate_ema50 else 0
average_positive_rate_ema144 = np.mean(positive_rate_ema144) if positive_rate_ema144 else 0
average_negative_rate_ema50 = np.mean(negative_rate_ema50) if negative_rate_ema50 else 0
average_negative_rate_ema144 = np.mean(negative_rate_ema144) if negative_rate_ema144 else 0

# 양수 및 음수 등락률 개수 계산
positive_count = len([rate for rate in filtered_return_rates if rate > 0])
negative_count = len([rate for rate in filtered_return_rates if rate < 0])

# 시각적 표현 (히스토그램, 필터링된 범위)
plt.figure(figsize=(10, 6))
plt.hist(filtered_return_rates, bins=30, range=(-20, 20), edgecolor='black', alpha=0.7)
plt.title(f"Filtered Return Rate Distribution (Average: {average_filtered_return_rate:.2f}%)")
plt.xlabel('Return Rate (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 필터링된 평균 등락률 출력
print(f"Filtered Average Return Rate: {average_filtered_return_rate:.2f}%")
print(f"Positive Return Rate Count: {positive_count}")
print(f"Negative Return Rate Count: {negative_count}")
print(f"양수 등락율 ema50평균 : {average_positive_rate_ema50:.2f}")
print(f"음수 등락율 ema50평균 : {average_negative_rate_ema50:.2f}")
print(f"양수 등락율 ema144평균 : {average_positive_rate_ema144:.2f}")
print(f"음수 등락율 ema144평균 : {average_negative_rate_ema144:.2f}")
print(f"돈 딸 확률 : {positive_count/(positive_count+negative_count)*100:.2f}%")
