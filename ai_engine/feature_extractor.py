"""
ECG Feature Extraction for SmartCPR Guardian
Extracts clinically relevant features from raw ECG + sensor data.
Supports: MIT-BIH Arrhythmia Database, PhysioNet, and simulated streams.
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class ECGFeatureExtractor:
    """
    Extracts features from raw ECG signal and sensor data.
    Features are designed to match FOLD-RM rule conditions.
    """

    def __init__(self, sampling_rate: int = 360):
        self.fs = sampling_rate  # Hz (MIT-BIH standard)
        self.window_sec = 10     # analysis window in seconds

    def extract(self, ecg_signal: np.ndarray,
                spo2: float = None,
                systolic_bp: float = None,
                motion_accel: float = None) -> Dict[str, float]:
        """
        Extract all features from a 10-second ECG window + sensor readings.

        Returns dict of features ready for FOLD-RM classifier.
        """
        features = {}

        # ── ECG-derived features ──────────────────────────────────────
        features["heart_rate"]        = self._compute_heart_rate(ecg_signal)
        features["ecg_rr_variability"] = self._compute_rr_variability(ecg_signal)
        features["ecg_amplitude"]     = self._compute_amplitude(ecg_signal)
        features["ecg_noise"]         = self._compute_noise_level(ecg_signal)
        features["qrs_width"]         = self._compute_qrs_width(ecg_signal)
        features["st_deviation"]      = self._compute_st_deviation(ecg_signal)
        features["rhythm_regularity"] = self._compute_regularity(ecg_signal)

        # ── Sensor features ───────────────────────────────────────────
        features["spo2"]              = spo2 if spo2 is not None else 98.0
        features["systolic_bp"]       = systolic_bp if systolic_bp is not None else 120.0
        features["pulse_pressure"]    = max(0, features["systolic_bp"] - 80)
        features["motion"]            = motion_accel if motion_accel is not None else 0.1
        features["fall_detected"]     = 1.0 if (motion_accel is not None and motion_accel > 3.0) else 0.0

        return features

    def _detect_r_peaks(self, ecg: np.ndarray) -> np.ndarray:
        """Pan-Tompkins inspired R-peak detection."""
        # Bandpass filter 5–15 Hz
        b, a = signal.butter(2, [5/(self.fs/2), 15/(self.fs/2)], btype='band')
        filtered = signal.filtfilt(b, a, ecg)

        # Derivative + squaring
        diff = np.diff(filtered)
        squared = diff ** 2

        # Moving window integration (150ms)
        window = int(0.15 * self.fs)
        integrated = np.convolve(squared, np.ones(window)/window, mode='same')

        # Find peaks with minimum 300ms refractory period
        min_dist = int(0.3 * self.fs)
        threshold = 0.5 * np.max(integrated)
        peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_dist)
        return peaks

    def _compute_heart_rate(self, ecg: np.ndarray) -> float:
        """Compute heart rate in BPM from R-peaks."""
        peaks = self._detect_r_peaks(ecg)
        if len(peaks) < 2:
            return 0.0
        rr_intervals = np.diff(peaks) / self.fs  # seconds
        hr = 60.0 / np.mean(rr_intervals)
        return round(float(np.clip(hr, 0, 400)), 1)

    def _compute_rr_variability(self, ecg: np.ndarray) -> float:
        """
        Compute RR interval variability (RMSSD normalized).
        High variability (>0.8) suggests VFib.
        """
        peaks = self._detect_r_peaks(ecg)
        if len(peaks) < 3:
            return 1.0  # assume high variability if no peaks (arrest)
        rr = np.diff(peaks) / self.fs
        rmssd = np.sqrt(np.mean(np.diff(rr)**2))
        mean_rr = np.mean(rr)
        return round(float(rmssd / (mean_rr + 1e-10)), 3)

    def _compute_amplitude(self, ecg: np.ndarray) -> float:
        """ECG peak-to-peak amplitude (mV). Near 0 = asystole."""
        return round(float(np.percentile(ecg, 95) - np.percentile(ecg, 5)), 4)

    def _compute_noise_level(self, ecg: np.ndarray) -> float:
        """Estimate noise level (0=clean, 1=lead-off artifact)."""
        b, a = signal.butter(2, 40/(self.fs/2), btype='high')
        hf = signal.filtfilt(b, a, ecg)
        noise = np.std(hf) / (np.std(ecg) + 1e-10)
        return round(float(np.clip(noise, 0, 1)), 3)

    def _compute_qrs_width(self, ecg: np.ndarray) -> float:
        """QRS complex width in ms. Wide QRS suggests bundle branch block."""
        peaks = self._detect_r_peaks(ecg)
        if len(peaks) == 0:
            return 0.0
        # Estimate width from peak neighborhood
        half_win = int(0.05 * self.fs)
        widths = []
        for p in peaks[:5]:
            start = max(0, p - half_win)
            end = min(len(ecg), p + half_win)
            segment = np.abs(ecg[start:end])
            above_half = np.where(segment > 0.5 * np.max(segment))[0]
            if len(above_half) > 0:
                widths.append((above_half[-1] - above_half[0]) / self.fs * 1000)
        return round(float(np.mean(widths)) if widths else 0.0, 1)

    def _compute_st_deviation(self, ecg: np.ndarray) -> float:
        """ST segment deviation (elevation/depression) in mV."""
        peaks = self._detect_r_peaks(ecg)
        if len(peaks) == 0:
            return 0.0
        st_offset = int(0.08 * self.fs)  # 80ms after R peak
        st_vals = []
        baseline = np.median(ecg)
        for p in peaks:
            st_idx = p + st_offset
            if st_idx < len(ecg):
                st_vals.append(ecg[st_idx] - baseline)
        return round(float(np.mean(st_vals)) if st_vals else 0.0, 4)

    def _compute_regularity(self, ecg: np.ndarray) -> float:
        """
        Heart rhythm regularity score (0=irregular, 1=perfectly regular).
        Low score suggests AFib or arrest.
        """
        peaks = self._detect_r_peaks(ecg)
        if len(peaks) < 3:
            return 0.0
        rr = np.diff(peaks)
        cv = np.std(rr) / (np.mean(rr) + 1e-10)
        return round(float(1.0 / (1.0 + cv)), 3)


class PatientVitalsSimulator:
    """
    Simulate realistic patient vitals for different cardiac conditions.
    Uses real parameter ranges from clinical literature.
    """

    @staticmethod
    def normal(noise: float = 0.05, hr: float = 72, spo2: float = 99, bp: float = 120) -> Tuple[np.ndarray, Dict]:
        """Normal sinus rhythm patient."""
        fs = 360
        t = np.linspace(0, 10, 10 * fs)
        # Simulate realistic ECG with P-QRS-T waves
        target_hr = hr + np.random.normal(0, 1)
        ecg = PatientVitalsSimulator._synthetic_ecg(t, target_hr, fs, noise)
        sensors = {
            "spo2": round(spo2 + np.random.normal(0, 0.2), 1),
            "systolic_bp": round(bp + np.random.normal(0, 2), 0),
            "motion_accel": round(abs(np.random.normal(0.1, 0.02)), 2),
        }
        return ecg, sensors

    @staticmethod
    def cardiac_arrest_vfib(noise: float = 0.1, spo2: float = 75, bp: float = 40) -> Tuple[np.ndarray, Dict]:
        """Ventricular fibrillation — chaotic, high amplitude."""
        fs = 360
        t = np.linspace(0, 10, 10 * fs)
        # VFib: chaotic oscillation 150–500 Hz components
        ecg = (
            0.3 * np.sin(2*np.pi*8*t + np.random.rand()*2*np.pi)
            + 0.2 * np.sin(2*np.pi*13*t + np.random.rand()*2*np.pi)
            + 0.15 * np.sin(2*np.pi*6*t + np.random.rand()*2*np.pi)
            + noise * np.random.randn(len(t))
        )
        sensors = {
            "spo2": round(spo2 + np.random.normal(0, 2), 1),
            "systolic_bp": round(bp + np.random.normal(0, 5), 0),
            "motion_accel": round(abs(np.random.normal(0.05, 0.02)), 2),
        }
        return ecg, sensors

    @staticmethod
    def cardiac_arrest_asystole(noise: float = 0.02, spo2: float = 60, bp: float = 30) -> Tuple[np.ndarray, Dict]:
        """Asystole — flatline ECG."""
        fs = 360
        t = np.linspace(0, 10, 10 * fs)
        ecg = noise * np.random.randn(len(t))  # near-zero signal
        sensors = {
            "spo2": round(spo2 + np.random.normal(0, 2), 1),
            "systolic_bp": round(bp + np.random.normal(0, 5), 0),
            "motion_accel": 0.0,
        }
        return ecg, sensors

    @staticmethod
    def pre_arrest_bradycardia(hr: float = 18, spo2: float = 87, bp: float = 58) -> Tuple[np.ndarray, Dict]:
        """Severe bradycardia — warning state before arrest."""
        fs = 360
        t = np.linspace(0, 10, 10 * fs)
        ecg = PatientVitalsSimulator._synthetic_ecg(t, hr, fs, 0.08)
        sensors = {
            "spo2": round(spo2 + np.random.normal(0, 1), 1),
            "systolic_bp": round(bp + np.random.normal(0, 3), 0),
            "motion_accel": round(abs(np.random.normal(0.08, 0.01)), 2),
        }
        return ecg, sensors

    @staticmethod
    def heart_failure(hr: float = 105, spo2: float = 89, bp: float = 95) -> Tuple[np.ndarray, Dict]:
        """Congestive Heart Failure — low SpO2, elevated HR, low BP."""
        fs = 360
        t = np.linspace(0, 10, 10 * fs)
        # CHF often shows sinus tachycardia or low voltage
        ecg = PatientVitalsSimulator._synthetic_ecg(t, hr, fs, 0.05) * 0.7 
        sensors = {
            "spo2": round(spo2 + np.random.normal(0, 1), 1),
            "systolic_bp": round(bp + np.random.normal(0, 5), 0),
            "motion_accel": round(abs(np.random.normal(0.05, 0.01)), 2),
        }
        return ecg, sensors

    @staticmethod
    def tachycardia(hr: float = 190, spo2: float = 94, bp: float = 105) -> Tuple[np.ndarray, Dict]:
        """Supraventricular Tachycardia — extremely high HR."""
        fs = 360
        t = np.linspace(0, 10, 10 * fs)
        ecg = PatientVitalsSimulator._synthetic_ecg(t, hr, fs, 0.03)
        sensors = {
            "spo2": round(spo2 + np.random.normal(0, 1), 1),
            "systolic_bp": round(bp + np.random.normal(0, 3), 0),
            "motion_accel": round(abs(np.random.normal(0.12, 0.02)), 2),
        }
        return ecg, sensors

    @staticmethod
    def _synthetic_ecg(t: np.ndarray, hr: float,
                       fs: int, noise: float) -> np.ndarray:
        """Generate synthetic P-QRS-T ECG waveform."""
        rr = 60.0 / hr
        ecg = np.zeros(len(t))
        beat_times = np.arange(0, t[-1], rr)

        for bt in beat_times:
            idx = int(bt * fs)
            # P wave
            p_idx = np.arange(max(0, idx-int(0.08*fs)), min(len(t), idx))
            ecg[p_idx] += 0.2 * np.sin(np.linspace(0, np.pi, len(p_idx)))
            # QRS complex
            q_idx = np.arange(max(0, idx), min(len(t), idx+int(0.08*fs)))
            qrs = np.zeros(len(q_idx))
            mid = len(q_idx)//2
            qrs[:mid] = np.linspace(-0.1, 1.0, mid)
            qrs[mid:] = np.linspace(1.0, -0.2, len(q_idx)-mid)
            ecg[q_idx] += qrs
            # T wave
            t_start = idx + int(0.12*fs)
            t_end = t_start + int(0.16*fs)
            t_idx = np.arange(max(0, t_start), min(len(t), t_end))
            ecg[t_idx] += 0.3 * np.sin(np.linspace(0, np.pi, len(t_idx)))

        ecg += noise * np.random.randn(len(ecg))
        return ecg
