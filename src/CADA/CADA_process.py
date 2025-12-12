"""
CADA_process.py
----
Module for Wi-Fi CSI-based CADA (CSI Activity Detection Algorithm) preprocessing and activity detection.

Key Functions
----
• z_normalization function: Z-score normalization.
• filter_normalization function: Outlier removal after normalization.
• realtime_cada_pipeline / cada_pipeline functions: Real-time and offline pipeline functionalities.
• SlidingCadaProcessor class: Sliding window-based activity detection.
• parse_and_normalize_payload function: MQTT payload parsing and Z-score transformation.
"""


import autorootcwd
import os
import csv
from scipy.signal import medfilt
import numpy as np
from collections import deque
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

def load_calibration_data(topics, mu_bg_dict, sigma_bg_dict):
    try:
        CALIB_DIR = "data/calibration"
        
        for topic in topics:
            calib_file = os.path.join(CALIB_DIR, f"{topic.replace('/','_')}_bg_params.csv")
            
            if os.path.exists(calib_file):
                with open(calib_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                mu_bg = np.array([float(x) for x in rows[0]])
                sigma_bg = np.array([float(x) for x in rows[1]])
                sigma_bg[sigma_bg == 0] = 1  # 0인 σ는 1로 대체
                
                mu_bg_dict[topic] = mu_bg
                sigma_bg_dict[topic] = sigma_bg
                print(f" Loaded calibration for {topic}")
            else:
                print(f" No calibration file found for {topic}: {calib_file}")
                
    except Exception as e:
        print(f"Error loading calibration data: {e}")


def parse_custom_timestamp(ts):
    """Converts a 15-digit ESP timestamp (YYMMDDhhmmssSSS) to a datetime object."""
    ts_str = str(ts).zfill(15)
    year = 2000 + int(ts_str[0:2])
    month = int(ts_str[2:4])
    day = int(ts_str[4:6])
    hour = int(ts_str[6:8])
    minute = int(ts_str[8:10])
    second = int(ts_str[10:12])
    millisecond = int(ts_str[12:15])
    microsecond = millisecond * 1000
    return datetime(year, month, day, hour, minute, second, microsecond)

def parse_and_normalize_payload(payload: str,
                                topic: str,
                                subcarriers: int,
                                indices_to_remove: list[int] | None,
                                mu_bg_dict: dict,
                                sigma_bg_dict: dict):
    """Extracts Z-score normalized amplitude vector and timestamp from MQTT payload string.

    Returns:
        (amp_z, packet_time) or None if parsing fails.
    """
    try:
        # 1) 타임스탬프 파싱 ---------------------------------------------------
        match = re.search(r"time=(\d{15})", payload)
        if match:
            ts_str = match.group(1)
            packet_time = parse_custom_timestamp(ts_str)
        else:
            packet_time = datetime.now()

        # 2) CSI 문자열 → 복소수 배열 ------------------------------------------
        csi_data_str = payload.split("CSI values: ")[-1].strip()
        csi_values = list(map(int, csi_data_str.split()))
        if len(csi_values) < subcarriers * 2:
            return None  # 데이터 부족

        csi_complex = [csi_values[i] + 1j * csi_values[i + 1]
                       for i in range(0, len(csi_values), 2)]
        csi_complex = np.array(csi_complex)[:subcarriers]

        # 3) 노이즈 채널 제거 --------------------------------------------------
        if indices_to_remove:
            csi_complex = np.delete(csi_complex, indices_to_remove)

        csi_amplitude = np.abs(csi_complex)

        # 4) Z-score 정규화 ----------------------------------------------------
        if topic in mu_bg_dict and topic in sigma_bg_dict:
            amp_z = z_normalization(csi_amplitude,
                                    mu_bg_dict[topic],
                                    sigma_bg_dict[topic])
        else:
            amp_z = csi_amplitude  # 캘리브레이션 미존재 시 정규화 생략

        return amp_z, packet_time

    except Exception as e:
        print(f"ERROR: parse_and_normalize_payload failed for {topic}: {e}")
        return None  
    

def z_normalization(amp, mu, sigma ) : 
    '''
    Desc :
        Z-score normalizaion & visualization
        param mu_bg : mean value for each subcarrier
        param sigma_bg : standard deviation for each subcarrier
    Example : 
        mu_bg, sigma_bg = read_calibration(_, _) 
        z_normalization(amp_reduced, mu_bg, sigma_bg)
    '''
    mu_bg = mu
    sigma_bg = sigma
    amp_normalized  = (amp - mu_bg) / sigma_bg
    return amp_normalized

def filter_normalization(amp_normalized, iqr_multiplier=1.5, gap_threshold=0.2):
    ''' Outlier removal after normalization based on subcarrier mean (using IQR) '''
    # 1. 평균 계산 
    means = np.mean(amp_normalized, axis=0)
    # 2. IQR 계산
    q1 = np.percentile(means, 25) # 하위 25% : -2.11
    q3 = np.percentile(means, 75) # 상위 75% : 0.44
    iqr = q3 - q1
    upper = q3 + iqr_multiplier * iqr # 4.27
    # 3. 평균값 내림차순 정렬
    sorted_indices = np.argsort(means)[::-1]
    top1_idx = sorted_indices[0] # 4번
    top2_idx = sorted_indices[1] # 3번
    top1_val = means[top1_idx]  # 4번 평균 :13.42 
    top2_val = means[top2_idx]  # 3번 평균 :4.85 
    # 4. 최대 평균이 upper보다 크고, 다음 값과 차이가 충분히 날 경우만 제거
    if top1_val > upper and (top1_val - top2_val) > gap_threshold:
        invalid_indices = [top1_idx]
    else:
        invalid_indices = []
    amp_norm_filtered = np.delete(amp_normalized, invalid_indices, axis=1 )
    print(f"[filter_normalization] Q1 = {q1:.2f}, IQR = {iqr:.2f}, upper = {upper:.2f}")
    print(f"[filter_normalization] Top1: SC {top1_idx}, mean = {top1_val:.2f}")
    print(f"[filter_normalization] Top2: SC {top2_idx}, mean = {top2_val:.2f}")
    print(f"[filter_normalization] Removed: {invalid_indices}")
    return  amp_norm_filtered

def robust_hampel(col, window=5, n_sigma=3):
    """Remove outliers using Hampel filter"""
    median = medfilt(col, kernel_size=window)
    dev    = np.abs(col - median)
    mad    = np.median(dev)
    out    = dev > n_sigma * mad
    col[out] = median[out]
    return col

def detrending_amp(amp, historical_window=100):
    """
    Desc:
        Function to perform 2-step detrending (for batch processing)
        - 1st step: Remove frame-wise mean (centering)
        - 2nd step: Remove subcarrier baseline (current mean + historical mean)
    Parameters:
        amp : Hampel_filtered
        historical_window : initial frame count used for baseline calculation (default: 100)
    Example:
        Hampel_filtered = np.apply_along_axis(robust_hampel, 0, amp_norm_filtered)
        detrended = detrending_amp(Hampel_filtered, historical_window=100)
    """
    # 1단계: 프레임별 평균 제거
    mean_per_frame = np.mean(amp, axis=1, keepdims=True)  
    detrended_packet = amp - mean_per_frame             
    # 2단계: 기준선 제거 (시간 평균 기준)
    mean_current = np.mean(amp, axis=0)                  
    mean_historical = np.mean(amp[:historical_window], axis=0)
    combined_mean = (mean_current + mean_historical) / 2    
    # 최종 detrending
    detrended = detrended_packet - combined_mean           
    return detrended

def extract_motion_features(detrended, WIN_SIZE=64):
    """
    Desc:
        Function to extract motion features from detrended CSI data (for batch processing)
        - 1st step: Calculate frame-wise amplitude change (std)
        - 2nd step: Differentiate + absolute value → extract change amount
        - 3rd step: Apply overlap-save moving average filter
    Parameters:
        detrended : Detrended amplitude data (frames x subcarriers)
        WIN_SIZE : Moving average filter size (default: 64)
    Example:
        detrended = detrending_amp(Hampel_filtered)
        feature = extract_motion_features(detrended, WIN_SIZE=64)
    """
    # 1단계: 진폭 변화량 계산 (프레임별 표준편차)
    std_per_pkt = np.std(detrended, axis=1)
    # 2단계: 변화량 미분 후 절댓값 처리
    feature_derivative_abs = np.abs(np.diff(std_per_pkt))
    # 3단계: Overlap-save convolution 기반 이동 평균 필터
    prev_samples = np.zeros(WIN_SIZE)
    padded_signal = np.concatenate([prev_samples, feature_derivative_abs])
    window = np.ones(WIN_SIZE)
    convolved = np.convolve(padded_signal, window, mode='valid')
    # 4단계: 최신 변화량만 반환
    feature = convolved[-len(feature_derivative_abs):]
    return feature

def detect_activity_with_ewma(feature, threshold_factor=2.5):
    """
    Desc:
        Function to calculate threshold using EWMA method from batch CSI data
        and perform motion detection. Can be executed directly in batch processing
        without maintaining real-time state.
    Parameters:
        feature : Input change series (1D array-like)
        threshold_factor : Threshold multiplier relative to mean (default: 2.5)
    Example:
        activity_flag, threshold = detect_activity_with_ewma(feature)
    """
    avgSigVal = np.mean(feature)
    ewma = avgSigVal  # 상태 유지 없이 한 번에 계산
    threshold = threshold_factor * ewma
    activity_flag = (feature > threshold).astype(float)
    return activity_flag, threshold


# === 통합 파이프라인 ========================================================

def cada_pipeline(amp_normalized,
                        use_filter_normalization=True,
                        historical_window=100,
                        WIN_SIZE=64,
                        threshold_factor=2.5):
    """
    Desc:
        Function to execute CADA pipeline on Z-score normalized CSI amplitude data.
        - Z-normalization must be completed outside.
    Parameters:
        amp_normalized : Z-score normalized amplitude data (frames x subcarriers)
        use_filter_normalization : filter_normalization 적용 여부
        historical_window : Frame count used for baseline calculation in detrending
        WIN_SIZE : Moving average filter size
        threshold_factor : Threshold multiplier relative to mean (default: 2.5)
    Returns:
        Dictionary:
            'amp_filtered', 'hampel_filtered', 'detrended',
            'feature', 'activity_flag', 'threshold'
    """
    try:
        # 1. Filter normalization (선택적)
        if use_filter_normalization:
            amp_filtered = filter_normalization(amp_normalized)
        else:
            amp_filtered = amp_normalized

        # 2. Hampel 필터 적용
        hampel_filtered = np.apply_along_axis(robust_hampel, 0, amp_filtered)

        # 3. Detrending
        detrended = detrending_amp(hampel_filtered, historical_window=historical_window)

        # 4. 움직임 특징 추출
        feature = extract_motion_features(detrended, WIN_SIZE=WIN_SIZE)

        # 5. 활동 감지
        activity_flag, threshold = detect_activity_with_ewma(feature, threshold_factor=threshold_factor)

        # 6. 결과 반환
        results = {
            'amp_filtered': amp_filtered,
            'hampel_filtered': hampel_filtered,
            'detrended': detrended,
            'feature': feature,
            'activity_flag': activity_flag,
            'threshold': threshold
        }

        return results

    except Exception as e:
        print(f"Error in cada_pipeline: {e}")
        dummy_shape = amp_normalized.shape[0] - 1
        return {
            'amp_filtered': amp_normalized,
            'hampel_filtered': amp_normalized,
            'detrended': amp_normalized,
            'feature': np.zeros(dummy_shape),
            'activity_flag': np.zeros(dummy_shape),
            'threshold': 0.1
        }

# =  CADA 활동 탐지 파이프라인 클래스 ========================================================

class SlidingCadaProcessor:
    """320-frame sliding window + stride-based CADA batch processing helper class"""

    def __init__(self,
                 topic: str,
                 buffer_manager,
                 window_size: int = 320,
                 stride: int = 40,
                 small_win_size: int = 64,
                 threshold_factor: float = 2.5,
                 executor: ThreadPoolExecutor | None = None):
        self.topic = topic
        self.buffer_manager = buffer_manager
        self.window_size = window_size
        self.stride = stride
        self.small_win_size = small_win_size  # WIN_SIZE for cada_pipeline    
        self.threshold_factor = threshold_factor

        self._buf = deque(maxlen=self.window_size)
        self._ts_buf = deque(maxlen=self.window_size)
        self._counter = 0
        self._processing_running = False
        self._executor = executor or ThreadPoolExecutor(max_workers=1)

    def push(self, amp_z: np.ndarray, packet_time):
        """Adds one frame to buffer and requests asynchronous batch processing if needed."""
        self._buf.append(amp_z.copy())
        self._ts_buf.append(packet_time)
        self._counter += 1

        if (len(self._buf) == self.window_size and
                (self._counter % self.stride == 0) and
                not self._processing_running):
            window_copy = np.array(self._buf)
            ts_copy = list(self._ts_buf)
            self._processing_running = True
            self._executor.submit(self._process_window, window_copy, ts_copy)

    def _process_window(self, csi_window: np.ndarray, ts_window):
        """Runs in background thread: cada_pipeline followed by buffer_manager result push"""
        try:
            # 1. Execute CADA pipeline (only normalized data is input)
            results = cada_pipeline(
                amp_normalized=csi_window,
                use_filter_normalization=False,
                historical_window=100,
                WIN_SIZE=self.small_win_size,
                threshold_factor=self.threshold_factor,
            )

            feature = results["feature"]
            avg_sig_val = float(np.mean(feature)) if len(feature) > 0 else 0.0

            # 2. Update EWMA
            alpha = 0.01
            prev_ewma = self.buffer_manager.cada_ewma_states.get(self.topic, 0.0)
            ewma_curr = avg_sig_val if prev_ewma == 0.0 else alpha * avg_sig_val + (1 - alpha) * prev_ewma
            self.buffer_manager.cada_ewma_states[self.topic] = ewma_curr
            Th = self.threshold_factor * ewma_curr
            activity_flag = (feature > Th).astype(float)

            # 3. Save results
            frames_to_push = min(self.stride, len(feature))
            start_idx = -frames_to_push
            for i in range(frames_to_push):
                idx = start_idx + i
                self.buffer_manager.cada_feature_buffers["activity_detection"][self.topic].append(feature[idx])
                self.buffer_manager.cada_feature_buffers["activity_flag"][self.topic].append(activity_flag[idx])
                self.buffer_manager.cada_feature_buffers["threshold"][self.topic].append(Th)

        except Exception as e:
            print(f"ERROR: SlidingCadaProcessor window processing failed for {self.topic}: {e}")
        finally:
            self._processing_running = False

if __name__ == "__main__" : 
    pass