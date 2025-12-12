import autorootcwd
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict, deque
import numpy as np
from src.CADA.csi_buffer_utils import RealtimeCSIBufferManager
from src.CADA.CADA_process import SlidingCadaProcessor, parse_and_normalize_payload, load_calibration_data
from demo.services.mqtt_manager import start_csi_mqtt_thread
from demo.config.settings import CSI_TOPIC, CSI_SUBCARRIERS as SUBCARRIER_NUM, CSI_INDICES_TO_REMOVE


# ----- 설정 -----
PLOT_POINTS = 300
PLOT_INTERVAL = 0.2  # 초 단위
BUFFER_SIZE = 600   
CADA_WINDOW_SIZE = 320
CADA_STRIDE = 20
CADA_SMALL_WIN_SIZE = 64

# ----- 버퍼 생성 -----
csi_buffers = RealtimeCSIBufferManager(topics=CSI_TOPIC, buffer_size=BUFFER_SIZE, window_size=CADA_SMALL_WIN_SIZE)
print("[INFO] Loading calibration data...")
load_calibration_data(CSI_TOPIC, csi_buffers.mu_bg_dict, csi_buffers.sigma_bg_dict)

cada_processor = SlidingCadaProcessor(
    topic=CSI_TOPIC[0],
    buffer_manager=csi_buffers,
    window_size=CADA_WINDOW_SIZE,
    stride=CADA_STRIDE,
    small_win_size=CADA_SMALL_WIN_SIZE
)

# 스레드 동기화를 위한 Lock
buffer_lock = threading.Lock()

# ----- 실시간 시각화 함수 -----
def plot_realtime():
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))

    # CADA 결과를 표시할 라인 생성 (plot_utils.py 스타일 참조)
    feature_line, = ax.plot([], [], color='green', linewidth=1.5, label="CADA Feature")
    threshold_line, = ax.plot([], [], color='darkred', linestyle='--', linewidth=1.5, label="Threshold")
    flag_line, = ax.plot([], [], color='red', drawstyle='steps-mid', linewidth=2, alpha=0.7, label="Activity Flag")

    ax.legend()
    ax.set_title(f"CSI Activity Detection (CADA) - Topic: {CSI_TOPIC[0]}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Feature Value")
    ax.grid(True, linestyle='--', alpha=0.6)

    # X축 시간 포맷 설정
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    while True:
        ts_list, feature_list, flag_list, threshold_list = [], [], [], []
        
        with buffer_lock:
            # CADA 결과 버퍼에서 최신 데이터 가져오기
            ts_deque = csi_buffers.timestamp_buffer[CSI_TOPIC[0]]
            feature_deque = csi_buffers.cada_feature_buffers['activity_detection'][CSI_TOPIC[0]]
            flag_deque = csi_buffers.cada_feature_buffers['activity_flag'][CSI_TOPIC[0]]
            threshold_deque = csi_buffers.cada_feature_buffers['threshold'][CSI_TOPIC[0]]

            num_samples = len(feature_deque)
            points_to_plot = min(num_samples, PLOT_POINTS)

            if points_to_plot > 0:
                # deques를 list로 변환
                feature_list = list(feature_deque)[-points_to_plot:]
                flag_list = list(flag_deque)[-points_to_plot:]
                threshold_list = list(threshold_deque)[-points_to_plot:]
                # 특징 데이터 길이에 맞춰 타임스탬프도 잘라내기
                ts_list = list(ts_deque)[-len(feature_list):]

        if ts_list and feature_list:
            # activity_flag 스케일링 (plot_utils.py 참조)
            threshold_height = np.mean(threshold_list) if threshold_list else 1.0
            scaled_flag_list = np.array(flag_list) * threshold_height

            # 데이터 설정
            feature_line.set_data(ts_list, feature_list)
            threshold_line.set_data(ts_list, threshold_list)
            flag_line.set_data(ts_list, scaled_flag_list)

            # 축 범위 자동 조절
            ax.relim()
            ax.autoscale_view()
            ax.set_xlim(ts_list[0], ts_list[-1])
            
        fig.autofmt_xdate() # X축 라벨 자동 포맷
        plt.pause(PLOT_INTERVAL)

# ----- MQTT 수신 핸들러 -----
def on_csi_message(topic, payload):
    # 1. 버퍼 저장 및 파싱
    parsed = parse_and_normalize_payload(
        payload=payload,
        topic=topic,
        subcarriers=SUBCARRIER_NUM,
        indices_to_remove=CSI_INDICES_TO_REMOVE,
        mu_bg_dict=csi_buffers.mu_bg_dict,
        sigma_bg_dict=csi_buffers.sigma_bg_dict,
    )
    if not parsed:
        return
    
    z_normalized, packet_time = parsed
    with buffer_lock:
        csi_buffers.timestamp_buffer[topic].append(packet_time)
        # z_normalized는 CADA 프로세서로 바로 전달되므로 버퍼에 중복 저장할 필요 없음

    # 2. CADA 처리
    cada_processor.push(z_normalized, packet_time)

# ----- 실행 -----
if __name__ == "__main__":
    print("[INFO] Starting CSI MQTT thread...")
    # MQTT 클라이언트는 자체 백그라운드 스레드에서 실행됩니다.
    start_csi_mqtt_thread(on_csi_message, topics=CSI_TOPIC)

    print("[INFO] Starting real-time plot...")
    # Matplotlib GUI는 메인 스레드에서 실행해야 합니다.
    plot_realtime()
