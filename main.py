#!/usr/bin/env python3
"""
🎯 Professional Dysfluency Detection & Speech Analysis System v4.0 - COMPLETE
All Features:
- Complete training pipeline with multiple models
- Real-time audio recording (JavaScript-based)
- Live analysis with SHAP explainability
- Comprehensive visualizations
- Model comparison and evaluation
- Professional modern UI
"""

import os
import re
import warnings
import logging
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any
import time
import tempfile
from datetime import datetime
import base64
import io

import numpy as np
import pandas as pd
#import torch

# Performance optimizations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import streamlit as st
import streamlit.components.v1 as components

# -------------------------
# Configuration
# -------------------------
class SystemConfig:
    """System configuration"""
    DATA_DIR = "segrigated_samples"
    OUTPUT_DIR = "output_results_pro"
    CACHE_DIR = "cache_features_pro"
    CLEAR_DIR = "clear_audio_pro"
    TEMP_DIR = "temp_recordings"
    
    # Audio settings
    TARGET_SR = 16000
    AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac")
    MAX_DURATION = 30
    
    # Feature dimensions
    MFCC_N = 13
    N_FFT = 512
    HOP_LENGTH = 256
    
    # Model settings
    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    def __init__(self):
        for d in [self.OUTPUT_DIR, self.CACHE_DIR, self.CLEAR_DIR, self.TEMP_DIR]:
            os.makedirs(d, exist_ok=True)

config = SystemConfig()

# Setup logging
logging.basicConfig(
    filename=os.path.join(config.OUTPUT_DIR, "system.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------
# Core Imports
# -------------------------
try:
    import librosa
    import librosa.display
    import soundfile as sf
    import noisereduce as nr
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from scipy import signal as scipy_signal
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        roc_curve, auc, precision_recall_fscore_support,
        precision_recall_curve, average_precision_score
    )
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from imblearn.over_sampling import SMOTE
    import xgboost as xgb
    import lightgbm as lgb
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    import joblib
    import shap
except ImportError as e:
    st.error(f"❌ Missing library: {e}")
    st.info("""
    Install required packages:
    ```bash
    pip install librosa soundfile scikit-learn imbalanced-learn xgboost lightgbm tensorflow transformers torch noisereduce plotly joblib streamlit shap seaborn scipy
    ```
    """)
    st.stop()

# -------------------------
# Custom Audio Recorder Component
# -------------------------
def audio_recorder_component():
    """Custom audio recorder using JavaScript"""
    
    recorder_html = """
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); border-radius: 15px;">
        <h3 style="color: white; margin: 0 0 20px 0;">🎙️ Audio Recorder</h3>
        <button id="recordBtn" onclick="toggleRecording()" 
                style="background: white; color: #fa709a; border: none; padding: 15px 30px; 
                       border-radius: 25px; font-size: 16px; font-weight: bold; cursor: pointer; 
                       box-shadow: 0 4px 15px rgba(0,0,0,0.2); transition: all 0.3s;">
            🔴 Start Recording
        </button>
        <div id="timer" style="color: white; font-size: 24px; margin-top: 15px; font-weight: bold;">00:00</div>
        <div id="status" style="color: white; margin-top: 10px; font-size: 14px;">Ready to record</div>
    </div>
    
    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let startTime;
        let timerInterval;
        
        async function toggleRecording() {
            const btn = document.getElementById('recordBtn');
            const status = document.getElementById('status');
            
            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = (event) => {
                        audioChunks.push(event.data);
                    };
                    
                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const reader = new FileReader();
                        reader.readAsDataURL(audioBlob);
                        reader.onloadend = () => {
                            const base64Audio = reader.result;
                            window.parent.postMessage({
                                type: 'streamlit:setComponentValue',
                                value: base64Audio
                            }, '*');
                        };
                    };
                    
                    mediaRecorder.start();
                    isRecording = true;
                    btn.textContent = '⏹️ Stop Recording';
                    btn.style.background = '#ff4444';
                    btn.style.color = 'white';
                    status.textContent = 'Recording...';
                    
                    startTime = Date.now();
                    timerInterval = setInterval(updateTimer, 100);
                    
                } catch (error) {
                    status.textContent = 'Error: ' + error.message;
                }
            } else {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                btn.textContent = '🔴 Start Recording';
                btn.style.background = 'white';
                btn.style.color = '#fa709a';
                status.textContent = 'Recording saved!';
                clearInterval(timerInterval);
            }
        }
        
        function updateTimer() {
            const elapsed = Date.now() - startTime;
            const seconds = Math.floor(elapsed / 1000);
            const milliseconds = Math.floor((elapsed % 1000) / 100);
            document.getElementById('timer').textContent = 
                String(seconds).padStart(2, '0') + '.' + milliseconds;
        }
    </script>
    """
    
    recorded_audio = components.html(recorder_html, height=200)
    return recorded_audio

# -------------------------
# Custom CSS - Modern Professional Design
# -------------------------
def apply_custom_css():
    st.markdown(""" 
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    /* Core Colors */
    --bg-main: #0a0a0f;           /* Dark space background */
    --text-light: #cbd5e1;        /* Soft gray text */
    --text-dark: #f8fafc;         /* White for high contrast */
    --accent-primary: #00f0ff;    /* Neon Cyan */
    --accent-secondary: #ff00ff;  /* Neon Magenta */
    --card-bg: rgba(10,10,15,0.8); /* Glass card */
    --success: #22c55e;
    --warning: #f59e0b;
    --error: #ef4444;

    --radius-lg: 18px;
    --radius-md: 12px;
    --shadow-soft: 0 12px 30px rgba(0,0,0,0.5);
    --transition: all 0.3s ease;
}

* {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body, .main {
    background: var(--bg-main);
    padding: 1rem;
    color: var(--text-light);
}

/* ===== Header Card ===== */
.header-card {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    color: var(--text-dark);
    padding: 3rem 2rem;
    border-radius: var(--radius-lg);
    text-align: center;
    box-shadow: var(--shadow-soft);
    position: relative;
    overflow: hidden;
}

.header-card::before {
    content: '';
    position: absolute;
    width: 250%;
    height: 250%;
    top: -75%;
    left: -75%;
    background: radial-gradient(circle at center, rgba(255,255,255,0.1) 0%, transparent 60%);
    animation: pulse 18s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1) rotate(15deg); }
}

.header-title {
    font-size: 3rem;
    font-weight: 700;
    position: relative;
    z-index: 1;
}

.header-subtitle {
    font-size: 1.2rem;
    font-weight: 400;
    opacity: 0.85;
    margin-top: 0.5rem;
    position: relative;
    z-index: 1;
}

/* ===== Glass Cards ===== */
.card, .metric-card, .shap-container, .info-box {
    background: var(--card-bg);
    backdrop-filter: blur(14px);
    padding: 2rem;
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-soft);
    margin-bottom: 1.5rem;
    transition: var(--transition);
    border: 1px solid rgba(255,255,255,0.15);
}

.card:hover, .metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 16px 36px rgba(0,0,0,0.7);
}

/* ===== Metrics ===== */
.metric-value {
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--accent-primary);
}

.metric-label {
    font-size: 0.9rem;
    color: var(--text-light);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ===== Prediction Card ===== */
.prediction-card {
    background: linear-gradient(135deg, var(--accent-secondary), var(--accent-primary));
    color: var(--text-dark);
    text-align: center;
    padding: 2.5rem;
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-soft);
}

.prediction-class {
    font-size: 2.5rem;
    font-weight: 700;
}

.prediction-confidence {
    font-size: 1.4rem;
    margin-top: 0.5rem;
    opacity: 0.95;
}

/* ===== Alerts ===== */
.alert {
    padding: 1rem 1.2rem;
    border-radius: var(--radius-md);
    font-weight: 500;
    margin: 1rem 0;
}

.alert-success { background: rgba(34,197,94,0.15); color: var(--success);}
.alert-warning { background: rgba(245,158,11,0.15); color: var(--warning);}
.alert-error { background: rgba(239,68,68,0.15); color: var(--error);}

/* ===== Buttons ===== */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    color: var(--text-dark);
    border-radius: var(--radius-md);
    padding: 0.8rem 2rem;
    font-weight: 600;
    border: none;
    box-shadow: 0 6px 20px rgba(0,0,0,0.6);
    transition: var(--transition);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(0,0,0,0.8);
}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(20,20,30,0.9);
    border-radius: var(--radius-lg);
    padding: 0.75rem;
    box-shadow: var(--shadow-soft);
    gap: 1rem;
}

.stTabs [data-baseweb="tab"] {
    padding: 0.6rem 1.5rem;
    border-radius: var(--radius-md);
    font-weight: 500;
    transition: var(--transition);
    color: var(--text-light);
}

.stTabs [aria-selected="true"] {
    background: var(--accent-primary);
    color: var(--text-dark);
}

/* ===== Progress Bar ===== */
.stProgress > div > div > div > div {
    background: var(--accent-primary);
    border-radius: 8px;
}

/* ===== Recording Card ===== */
.recording-card {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    color: var(--text-dark);
    padding: 2rem;
    border-radius: var(--radius-lg);
    text-align: center;
    box-shadow: var(--shadow-soft);
}

/* ===== Badge ===== */
.format-badge {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 500;
    background: rgba(0,240,255,0.15);
    color: var(--accent-primary);
    margin: 0.3rem;
}

/* ===== Training Status ===== */
.training-status {
    background: rgba(255,0,255,0.08);
    padding: 1.5rem;
    border-radius: var(--radius-md);
}

/* ===== Responsive ===== */
@media (max-width: 768px) {
    .header-title { font-size: 2rem; }
    .metric-value { font-size: 1.8rem; }
    .prediction-class { font-size: 2rem; }
}
</style>""", unsafe_allow_html=True)

# -------------------------
# Utility Functions
# -------------------------
def list_audio_files(root: str) -> List[str]:
    """List all audio files"""
    if not os.path.exists(root):
        return []
    return sorted([
        os.path.join(r, f) for r, _, fs in os.walk(root)
        for f in fs if f.lower().endswith(config.AUDIO_EXTS)
    ])

def load_audio_safe(path: str, sr: int = None) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Load audio with error handling"""
    sr = sr or config.TARGET_SR
    try:
        y, s = librosa.load(path, sr=sr, mono=True, duration=config.MAX_DURATION)
        if len(y) < 1024 or not np.isfinite(y).all():
            return None, None
        return y, s
    except Exception as e:
        logging.error(f"load_audio error {path}: {e}")
        return None, None

def load_audio_from_base64(base64_string: str, sr: int = None) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Load audio from base64 string"""
    sr = sr or config.TARGET_SR
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        audio_bytes = base64.b64decode(base64_string)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        y, s = librosa.load(tmp_path, sr=sr, mono=True, duration=config.MAX_DURATION)
        os.unlink(tmp_path)
        
        if len(y) < 1024 or not np.isfinite(y).all():
            return None, None
        
        return y, s
    except Exception as e:
        logging.error(f"load_audio_from_base64 error: {e}")
        return None, None

def detect_audio_format(file) -> Dict[str, Any]:
    """Detect audio format"""
    try:
        file.seek(0)
        info = sf.info(file)
        return {
            'format': info.format,
            'subtype': info.subtype,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'duration': info.duration,
            'frames': info.frames
        }
    except Exception as e:
        return {
            'format': 'Unknown',
            'subtype': 'Unknown',
            'sample_rate': 0,
            'channels': 0,
            'duration': 0,
            'frames': 0
        }

# -------------------------
# Audio Analysis
# -------------------------
class AudioAnalyzer:
    """Audio analysis class"""
    
    @staticmethod
    def analyze_quality(y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze audio quality"""
        metrics = {}
        
        try:
            # SNR
            frame_len = int(0.025 * sr)
            hop = max(frame_len // 4, 1)
            frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop)
            energy = np.sum(frames**2, axis=0)
            noise_mask = energy < np.percentile(energy, 25)
            
            if noise_mask.sum() > 0:
                metrics['snr_db'] = float(10.0 * np.log10(
                    np.mean(energy) / (np.mean(energy[noise_mask]) + 1e-10)
                ))
            else:
                metrics['snr_db'] = 0.0
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            metrics['zcr_mean'] = float(np.mean(zcr))
            metrics['zcr_std'] = float(np.std(zcr))
            
            # RMS
            rms = librosa.feature.rms(y=y)[0]
            metrics['rms_mean'] = float(np.mean(rms))
            metrics['rms_std'] = float(np.std(rms))
            
            # Spectral
            S = np.abs(librosa.stft(y))
            centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
            rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
            
            metrics['spectral_centroid'] = float(np.mean(centroid))
            metrics['spectral_rolloff'] = float(np.mean(rolloff))
            
            # Dynamic range
            metrics['dynamic_range_db'] = float(20 * np.log10(
                np.max(np.abs(y)) / (np.mean(np.abs(y)) + 1e-10)
            ))
            
            # Silence
            threshold = np.max(np.abs(y)) * 0.01
            silence_frames = np.abs(y) < threshold
            metrics['silence_ratio'] = float(np.mean(silence_frames))
        
        except Exception as e:
            logging.error(f"Quality analysis error: {e}")
            metrics = {
                'snr_db': 0.0,
                'zcr_mean': 0.0,
                'zcr_std': 0.0,
                'rms_mean': 0.0,
                'rms_std': 0.0,
                'spectral_centroid': 0.0,
                'spectral_rolloff': 0.0,
                'dynamic_range_db': 0.0,
                'silence_ratio': 0.0
            }
        
        return metrics
    
    @staticmethod
    def analyze_prosody(y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze prosody"""
        metrics = {}
        
        try:
            # F0
            f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) > 0:
                metrics['f0_mean'] = float(np.nanmean(f0_voiced))
                metrics['f0_std'] = float(np.nanstd(f0_voiced))
                metrics['f0_range'] = float(np.nanmax(f0_voiced) - np.nanmin(f0_voiced))
            else:
                metrics['f0_mean'] = 0.0
                metrics['f0_std'] = 0.0
                metrics['f0_range'] = 0.0
            
            # Speech rate
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            duration = len(y) / sr
            metrics['speech_rate'] = float(len(onsets) / duration) if duration > 0 else 0.0
            
            # Pause analysis
            metrics['avg_pause_duration'] = 0.0
            metrics['pause_count'] = 0
        
        except Exception as e:
            logging.error(f"Prosody error: {e}")
            metrics = {
                'f0_mean': 0.0,
                'f0_std': 0.0,
                'f0_range': 0.0,
                'speech_rate': 0.0,
                'avg_pause_duration': 0.0,
                'pause_count': 0
            }
        
        return metrics

# -------------------------
# Visualization
# -------------------------
def plot_waveform_and_spectrogram(y: np.ndarray, sr: int, title: str = "Audio Analysis"):
    """Plot waveform and spectrogram"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Waveform', 'Mel Spectrogram'),
        vertical_spacing=0.12
    )
    
    # Waveform
    time = np.linspace(0, len(y)/sr, len(y))
    fig.add_trace(
        go.Scatter(x=time, y=y, mode='lines', name='Amplitude',
                  line=dict(color='#667eea', width=1)),
        row=1, col=1
    )
    
    # Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    fig.add_trace(
        go.Heatmap(z=S_db, colorscale='Viridis', name='Mel Spectrogram'),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (frames)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Mel Frequency", row=2, col=1)
    
    fig.update_layout(
        title=title,
        height=700,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def plot_audio_quality_radar(metrics: Dict[str, float]):
    """Radar chart for quality"""
    categories = ['SNR', 'Dynamic Range', 'Speech Rate', 'RMS Energy', 'Spectral Quality']
    
    values = [
        min(max(metrics.get('snr_db', 0) * 5, 0), 100),
        min(max(metrics.get('dynamic_range_db', 0) * 3, 0), 100),
        min(max(metrics.get('speech_rate', 0) * 20, 0), 100),
        min(max(metrics.get('rms_mean', 0) * 1000, 0), 100),
        min(max(metrics.get('spectral_centroid', 0) / 50, 0), 100)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Audio Quality',
        line=dict(color='#667eea', width=2),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title="Audio Quality Analysis",
        height=500
    )
    
    return fig

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=class_names,
                    y=class_names,
                    color_continuous_scale='Viridis',
                    text_auto=True)
    
    fig.update_layout(
        title="Confusion Matrix",
        height=500
    )
    
    return fig

def plot_roc_curves(y_true, y_pred_proba, class_names):
    """Plot ROC curves"""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig = go.Figure()
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{class_names[i]} (AUC = {roc_auc:.2f})',
            line=dict(width=2)
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title="ROC Curves - Multi-class",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=600,
        template='plotly_white'
    )
    
    return fig

# -------------------------
# Feature Extraction
# -------------------------
def extract_enhanced_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract enhanced features"""
    try:
        features = []
        
        # MFCC with deltas
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.MFCC_N)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        for feat in [mfcc, delta, delta2]:
            features.extend([np.mean(feat, axis=1), np.std(feat, axis=1)])
        
        features = np.concatenate(features)
        
        # Spectral features
        S = np.abs(librosa.stft(y))
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        
        spectral_feats = []
        for feat in [centroid, rolloff, bandwidth, flatness]:
            spectral_feats.extend([np.mean(feat), np.std(feat)])
        spectral_feats.extend([np.mean(contrast), np.std(contrast)])
        
        features = np.concatenate([features, spectral_feats])
        
        # Prosodic features
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            prosodic = [np.mean(f0_clean), np.std(f0_clean),
                       np.min(f0_clean), np.max(f0_clean)]
        else:
            prosodic = [0, 0, 0, 0]
        
        # Energy & ZCR
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        prosodic.extend([np.mean(rms), np.std(rms), np.mean(zcr), np.std(zcr)])
        
        features = np.concatenate([features, prosodic])
        
        # Rhythm
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        rhythm = [np.mean(tempogram), np.std(tempogram)]
        
        features = np.concatenate([features, rhythm])
        
        return features.astype(np.float32)
    
    except Exception as e:
        logging.error(f"Feature extraction error: {e}")
        return np.zeros(90, dtype=np.float32)

# -------------------------
# Training Functions
# -------------------------
def load_and_process_dataset(progress_bar=None, status_text=None):
    """Load and process dataset"""
    
    files = list_audio_files(config.DATA_DIR)
    
    if not files:
        return None, None, None, None
    
    X_list = []
    y_list = []
    
    for i, file in enumerate(files):
        if progress_bar:
            progress_bar.progress((i + 1) / len(files))
        if status_text:
            status_text.text(f"Processing {i+1}/{len(files)}: {Path(file).name}")
        
        y_audio, sr = load_audio_safe(file)
        
        if y_audio is None:
            continue
        
        # Clean audio
        try:
            y_clean = nr.reduce_noise(y=y_audio, sr=sr, prop_decrease=0.7)
        except:
            y_clean = y_audio
        
        # Extract features
        features = extract_enhanced_features(y_clean, sr)
        
        if features is not None and len(features) > 0:
            X_list.append(features)
            label = Path(file).parent.name
            y_list.append(label)
    
    if not X_list:
        return None, None, None, None
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le, y

def train_classical_models(X_train, X_test, y_train, y_test, le, progress_callback=None):
    """Train classical ML models"""
    
    results = {}
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(config.OUTPUT_DIR, "scaler_classical.pkl"))
    
    # Models to train
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=20, 
                                              min_samples_split=5, n_jobs=-1, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1,
                                     subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=10,
                                                      learning_rate=0.1, random_state=42)
    }
    
    class_names = list(le.classes_)
    
    for i, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(f"Training {name}...", (i + 1) / (len(models) + 1))
        
        try:
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, n_jobs=-1)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': cm
            }
            
            # Save model
            if name == 'RandomForest':
                joblib.dump(model, os.path.join(config.OUTPUT_DIR, "model_rf.pkl"))
            elif name == 'XGBoost':
                joblib.dump(model, os.path.join(config.OUTPUT_DIR, "model_xgb.pkl"))
            elif name == 'GradientBoosting':
                joblib.dump(model, os.path.join(config.OUTPUT_DIR, "model_gb.pkl"))
            
            logging.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        except Exception as e:
            logging.error(f"Error training {name}: {e}")
            results[name] = None
    
    # Train LightGBM separately
    if progress_callback:
        progress_callback("Training LightGBM...", 1.0)
    
    try:
        lgb_train = lgb.Dataset(X_train_scaled, y_train)
        lgb_test = lgb.Dataset(X_test_scaled, y_test, reference=lgb_train)
        
        params = {
            'objective': 'multiclass',
            'num_class': len(class_names),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        lgb_model = lgb.train(params, lgb_train, num_boost_round=200,
                             valid_sets=[lgb_test], callbacks=[lgb.early_stopping(50)])
        
        # Predict
        y_pred_proba_lgb = lgb_model.predict(X_test_scaled, num_iteration=lgb_model.best_iteration)
        y_pred_lgb = np.argmax(y_pred_proba_lgb, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred_lgb)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_lgb, average='weighted', zero_division=0
        )
        
        cm = confusion_matrix(y_test, y_pred_lgb)
        
        results['LightGBM'] = {
            'model': lgb_model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred_lgb,
            'y_pred_proba': y_pred_proba_lgb,
            'confusion_matrix': cm
        }
        
        # Save
        lgb_model.save_model(os.path.join(config.OUTPUT_DIR, "model_lgb.txt"))
        
        logging.info(f"LightGBM - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    except Exception as e:
        logging.error(f"Error training LightGBM: {e}")
        results['LightGBM'] = None
    
    return results

# -------------------------
# Page Configuration
# -------------------------
# st.set_page_config(
#     page_title="Dysfluency Detection Pro",
#     page_icon="🎙️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

apply_custom_css()

# -------------------------
# Main Application
# -------------------------
def main():
    # Header
    st.markdown("""
    <div class="header-card">
        <h1 class="header-title">🎙️ Professional Dysfluency Detection System</h1>
        <p class="header-subtitle">Advanced Speech Analysis with AI-Powered Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ System Configuration")
        
        # st.markdown(f"""
        # <div class="info-box">
        #     <strong>🖥️ Device:</strong> {config.DEVICE}<br>
        #     <strong>📊 Sample Rate:</strong> {config.TARGET_SR} Hz<br>
        #     <strong>🎵 Max Duration:</strong> {config.MAX_DURATION}s<br>
        #     <strong>🧠 MFCC Features:</strong> {config.MFCC_N}
        # </div>
        # """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        files = list_audio_files(config.DATA_DIR)
        st.info(f"**Dataset Files:** {len(files)}")
        
        if os.path.exists(os.path.join(config.OUTPUT_DIR, "label_encoder.pkl")):
            st.success("✅ Models Trained")
        else:
            st.warning("⚠️ No trained models")
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs([
        "🎤 Analysis & Recording", 
        "🎯 Train Models", 
        "📊 Results & Insights"
    ])
    
    # =====================================================
    # TAB 1: Analysis & Recording
    # =====================================================
    with tab1:
        st.markdown("## 🎤 Audio Analysis Center")
        
        col1, col2 = st.columns([1, 1])
        
        # Initialize session state
        if 'recording_data' not in st.session_state:
            st.session_state.recording_data = None
        
        with col1:
            st.markdown("### 📤 Upload Audio File")
            
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
                help="Upload audio file for analysis"
            )
            
            if uploaded_file:
                format_info = detect_audio_format(uploaded_file)
                
                st.markdown(f"""
                <div class="info-box">
                    <strong>📁 File:</strong> {uploaded_file.name}<br>
                    <strong>🎵 Format:</strong> <span class="format-badge">{format_info['format']}</span><br>
                    <strong>📊 Sample Rate:</strong> {format_info['sample_rate']} Hz<br>
                    <strong>⏱️ Duration:</strong> {format_info['duration']:.2f}s<br>
                    <strong>📢 Channels:</strong> {format_info['channels']}
                </div>
                """, unsafe_allow_html=True)
                
                st.audio(uploaded_file, format='audio/wav')
        
        with col2:
            st.markdown("### 🎙️ Record Audio")
            
            st.markdown("""
            <div class="recording-card">
                <h3 style="color: white; margin: 0;">🔴 Live Recording</h3>
                <p style="color: white; opacity: 0.9; margin-top: 0.5rem;">
                    Click below to start/stop recording
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Audio recorder
            recorded = audio_recorder_component()
            
            if recorded and recorded != st.session_state.recording_data:
                st.session_state.recording_data = recorded
                st.success("✅ Recording captured!")
                st.info("Use recorded audio for analysis below")
        
        # Analysis Section
        audio_to_analyze = uploaded_file
        
        if st.session_state.get('recording_data'):
            use_recording = st.checkbox("📌 Use recorded audio for analysis")
            if use_recording:
                audio_to_analyze = st.session_state.recording_data
        
        if audio_to_analyze:
            st.markdown("---")
            
            if st.button("🔍 Analyze Audio", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing audio..."):
                    start_time = time.time()
                    
                    # Load audio
                    if isinstance(audio_to_analyze, str):
                        y, sr = load_audio_from_base64(audio_to_analyze)
                    else:
                        audio_to_analyze.seek(0)
                        data, sr_orig = sf.read(audio_to_analyze)
                        if data.ndim > 1:
                            data = np.mean(data, axis=1)
                        y = librosa.resample(data.astype(np.float32), 
                                            orig_sr=sr_orig, target_sr=config.TARGET_SR)
                        sr = config.TARGET_SR
                    
                    if y is None:
                        st.error("❌ Failed to load audio")
                    else:
                        # Clean audio
                        try:
                            y_clean = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.7)
                        except:
                            y_clean = y
                        
                        proc_time = time.time() - start_time
                        
                        # Display metrics
                        st.markdown("### 📊 Audio Quality Metrics")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        analyzer = AudioAnalyzer()
                        quality_metrics = analyzer.analyze_quality(y_clean, sr)
                        prosody_metrics = analyzer.analyze_prosody(y_clean, sr)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{len(y)/sr:.1f}s</div>
                                <div class="metric-label">Duration</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{quality_metrics['snr_db']:.1f}</div>
                                <div class="metric-label">SNR (dB)</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{prosody_metrics['speech_rate']:.1f}</div>
                                <div class="metric-label">Speech Rate</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{prosody_metrics['f0_mean']:.0f}</div>
                                <div class="metric-label">Avg Pitch (Hz)</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col5:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{proc_time:.2f}s</div>
                                <div class="metric-label">Processing Time</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Visualization
                        st.markdown("### 📈 Waveform & Spectrogram Analysis")
                        fig_wave = plot_waveform_and_spectrogram(y_clean, sr)
                        st.plotly_chart(fig_wave, use_container_width=True)
                        
                        # Quality Radar
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_radar = plot_audio_quality_radar(quality_metrics)
                            st.plotly_chart(fig_radar, use_container_width=True)
                        
                        with col2:
                            st.markdown("### 📋 Detailed Metrics")
                            
                            all_metrics = {**quality_metrics, **prosody_metrics}
                            metrics_df = pd.DataFrame(
                                list(all_metrics.items()),
                                columns=['Metric', 'Value']
                            )
                            metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.3f}")
                            
                            st.dataframe(metrics_df, use_container_width=True, height=400)
                        
                        # Model Predictions Section
                        st.markdown("---")
                        st.markdown("### 🎯 Dysfluency Predictions")
                        
                        le_path = os.path.join(config.OUTPUT_DIR, "label_encoder.pkl")
                        
                        if not os.path.exists(le_path):
                            st.warning("⚠️ No trained models found. Train models first in the 'Train Models' tab.")
                        else:
                            # Load models and make predictions
                            le = joblib.load(le_path)
                            class_names = list(le.classes_)
                            
                            # Extract features
                            classical_features = extract_enhanced_features(y_clean, sr)
                            
                            # Load scaler
                            scaler_path = os.path.join(config.OUTPUT_DIR, "scaler_classical.pkl")
                            if os.path.exists(scaler_path):
                                scaler = joblib.load(scaler_path)
                                X = scaler.transform(classical_features.reshape(1, -1))
                            else:
                                X = classical_features.reshape(1, -1)
                            
                            # Predict with models
                            model_predictions = {}
                            
                            model_files = {
                                'RandomForest': 'model_rf.pkl',
                                'XGBoost': 'model_xgb.pkl',
                                'GradientBoosting': 'model_gb.pkl',
                                'LightGBM': 'model_lgb.txt'
                            }
                            
                            for model_name, filename in model_files.items():
                                model_path = os.path.join(config.OUTPUT_DIR, filename)
                                
                                if os.path.exists(model_path):
                                    try:
                                        if filename.endswith('.pkl'):
                                            model = joblib.load(model_path)
                                            pred = model.predict(X)[0]
                                            proba = model.predict_proba(X)[0]
                                        elif filename.endswith('.txt'):
                                            model = lgb.Booster(model_file=model_path)
                                            proba = model.predict(X)[0]
                                            pred = np.argmax(proba)
                                        
                                        model_predictions[model_name] = {
                                            'pred': pred,
                                            'proba': proba,
                                            'model': model
                                        }
                                    except Exception as e:
                                        logging.error(f"Prediction error {model_name}: {e}")
                            
                            if model_predictions:
                                # Display predictions
                                cols = st.columns(len(model_predictions))
                                
                                for col, (model_name, result) in zip(cols, model_predictions.items()):
                                    with col:
                                        pred_class = class_names[result['pred']]
                                        confidence = result['proba'][result['pred']] * 100
                                        
                                        st.markdown(f"""
                                        <div class="prediction-card">
                                            <h4 style="color: white; margin: 0;">{model_name}</h4>
                                            <div class="prediction-class">{pred_class}</div>
                                            <div class="prediction-confidence">{confidence:.1f}% confidence</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Probability chart
                                        prob_df = pd.DataFrame({
                                            'Class': class_names,
                                            'Probability': result['proba'] * 100
                                        }).sort_values('Probability', ascending=False)
                                        
                                        fig_prob = px.bar(
                                            prob_df,
                                            x='Probability',
                                            y='Class',
                                            orientation='h',
                                            color='Probability',
                                            color_continuous_scale='Viridis'
                                        )
                                        fig_prob.update_layout(height=300, showlegend=False)
                                        st.plotly_chart(fig_prob, use_container_width=True)
                                
                                # SHAP Analysis
                                st.markdown("---")
                                st.markdown("### 🔍 Feature Importance (SHAP Analysis)")
                                
                                shap_options = [k for k in model_predictions.keys() if k != 'LightGBM']
                                
                                if shap_options:
                                    shap_model = st.selectbox(
                                        "Select model for SHAP explainability",
                                        options=shap_options
                                    )
                                    
                                    if st.button("🎯 Generate SHAP Analysis"):
                                        with st.spinner("Computing SHAP values..."):
                                            try:
                                                model = model_predictions[shap_model]['model']
                                                
                                                explainer = shap.TreeExplainer(model)
                                                shap_values = explainer.shap_values(X)
                                                
                                                st.markdown('<div class="shap-container">', unsafe_allow_html=True)
                                                st.markdown("#### SHAP Feature Importance")
                                                
                                                fig, ax = plt.subplots(figsize=(10, 6))
                                                
                                                if len(shap_values.shape) == 3:
                                                    shap.summary_plot(
                                                        shap_values[:, :, result['pred']],
                                                        X,
                                                        plot_type="bar",
                                                        show=False
                                                    )
                                                else:
                                                    shap.summary_plot(
                                                        shap_values,
                                                        X,
                                                        plot_type="bar",
                                                        show=False
                                                    )
                                                
                                                st.pyplot(fig)
                                                st.markdown('</div>', unsafe_allow_html=True)
                                                
                                                st.success("✅ SHAP analysis completed!")
                                            
                                            except Exception as e:
                                                st.error(f"SHAP error: {e}")
                                                logging.error(f"SHAP error: {e}")
    
    # =====================================================
    # TAB 2: Train Models
    # =====================================================
    with tab2:
        st.markdown("## 🎯 Model Training Pipeline")
        
        files = list_audio_files(config.DATA_DIR)
        
        if not files:
            st.error(f"❌ No audio files found in `{config.DATA_DIR}`")
            st.info("""
            **Setup Instructions:**
            1. Create folder structure: `segrigated_samples/class_name/audio_files.wav`
            2. Add audio files for each class
            3. Return here to train models
            """)
        else:
            # Dataset Overview
            st.markdown("### 📊 Dataset Overview")
            
            labels = [Path(f).parent.name for f in files]
            dist = Counter(labels)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(files)}</div>
                    <div class="metric-label">Total Files</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(dist)}</div>
                    <div class="metric-label">Classes</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(files) / len(dist):.1f}</div>
                    <div class="metric-label">Avg per Class</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Class distribution
            st.markdown("### 📈 Class Distribution")
            
            dist_df = pd.DataFrame(list(dist.items()), columns=['Class', 'Count'])
            dist_df = dist_df.sort_values('Count', ascending=False)
            
            fig_dist = px.bar(
                dist_df, 
                x='Class', 
                y='Count',
                color='Count',
                color_continuous_scale='Viridis',
                title="Audio Files per Class"
            )
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Display class details
            st.dataframe(dist_df, use_container_width=True)
            
            # Training Configuration
            st.markdown("---")
            st.markdown("### ⚙️ Training Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
                apply_smote = st.checkbox("Apply SMOTE (handle imbalance)", value=True)
            
            with col2:
                random_state = st.number_input("Random State", 0, 100, 42)
                cv_folds = st.number_input("Cross-validation Folds", 3, 10, 5)
            
            # Training Button
            st.markdown("---")
            
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                
                st.markdown('<div class="training-status">', unsafe_allow_html=True)
                st.markdown("### 🔄 Training In Progress...")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Load dataset
                status_text.text("Loading and processing dataset...")
                X, y_encoded, le, y_original = load_and_process_dataset(progress_bar, status_text)
                
                if X is None:
                    st.error("❌ Failed to load dataset")
                else:
                    st.success(f"✅ Loaded {len(X)} samples")
                    
                    # Save label encoder
                    joblib.dump(le, os.path.join(config.OUTPUT_DIR, "label_encoder.pkl"))
                    
                    # Split data
                    status_text.text("Splitting dataset...")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
                    )
                    
                    st.info(f"Training: {len(X_train)} | Testing: {len(X_test)}")
                    
                    # Apply SMOTE
                    if apply_smote:
                        status_text.text("Applying SMOTE...")
                        try:
                            smote = SMOTE(random_state=random_state)
                            X_train, y_train = smote.fit_resample(X_train, y_train)
                            st.success(f"✅ SMOTE applied: {len(X_train)} training samples")
                        except Exception as e:
                            st.warning(f"⚠️ SMOTE failed: {e}")
                    
                    # Train models
                    def update_progress(msg, progress):
                        status_text.text(msg)
                        progress_bar.progress(progress)
                    
                    results = train_classical_models(
                        X_train, X_test, y_train, y_test, le, 
                        progress_callback=update_progress
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 🎉 Training Complete!")
                    
                    results_data = []
                    
                    for model_name, result in results.items():
                        if result:
                            results_data.append({
                                'Model': model_name,
                                'Accuracy (%)': f"{result['accuracy'] * 100:.2f}",
                                'Precision (%)': f"{result['precision'] * 100:.2f}",
                                'Recall (%)': f"{result['recall'] * 100:.2f}",
                                'F1-Score (%)': f"{result['f1'] * 100:.2f}",
                                'CV Mean': f"{result.get('cv_mean', 0) * 100:.2f}" if 'cv_mean' in result else 'N/A',
                                'CV Std': f"{result.get('cv_std', 0) * 100:.2f}" if 'cv_std' in result else 'N/A'
                            })
                    
                    results_df = pd.DataFrame(results_data)
                    
                    # Save results
                    results_df.to_csv(os.path.join(config.OUTPUT_DIR, "training_results.csv"), index=False)
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Performance comparison
                    st.markdown("### 📊 Model Performance Comparison")
                    
                    fig_perf = px.bar(
                        results_df,
                        x='Model',
                        y=['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'],
                        barmode='group',
                        title="Model Metrics Comparison"
                    )
                    st.plotly_chart(fig_perf, use_container_width=True)
                    
                    # Confusion matrices
                    st.markdown("### 🎯 Confusion Matrices")
                    
                    class_names = list(le.classes_)
                    
                    cols = st.columns(2)
                    
                    for i, (model_name, result) in enumerate(results.items()):
                        if result and i < 4:
                            with cols[i % 2]:
                                fig_cm = plot_confusion_matrix(result['confusion_matrix'], class_names)
                                fig_cm.update_layout(title=f"{model_name} Confusion Matrix")
                                st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # ROC Curves
                    st.markdown("### 📈 ROC Curves")
                    
                    for model_name, result in results.items():
                        if result and 'y_pred_proba' in result:
                            fig_roc = plot_roc_curves(y_test, result['y_pred_proba'], class_names)
                            fig_roc.update_layout(title=f"{model_name} ROC Curves")
                            st.plotly_chart(fig_roc, use_container_width=True)
                            break
                    
                    st.success("🎉 All models trained and saved successfully!")
    
    # =====================================================
    # TAB 3: Results & Insights
    # =====================================================
    with tab3:
        st.markdown("## 📊 Results & Performance Insights")
        
        results_path = os.path.join(config.OUTPUT_DIR, "training_results.csv")
        
        if not os.path.exists(results_path):
            st.info("⚠️ No training results found. Train models first in the 'Train Models' tab.")
        else:
            # Load results
            results_df = pd.read_csv(results_path)
            
            st.markdown("### 🏆 Overall Performance")
            
            # Display results
            st.dataframe(results_df, use_container_width=True)
            
            # Performance metrics
            col1, col2 = st.columns(2)
            
            with col1:
                fig_acc = px.bar(
                    results_df,
                    x='Model',
                    y='Accuracy (%)',
                    color='Accuracy (%)',
                    color_continuous_scale='Viridis',
                    title="Model Accuracy Comparison"
                )
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col2:
                fig_f1 = px.bar(
                    results_df,
                    x='Model',
                    y='F1-Score (%)',
                    color='F1-Score (%)',
                    color_continuous_scale='Plasma',
                    title="Model F1-Score Comparison"
                )
                st.plotly_chart(fig_f1, use_container_width=True)
            
            # Best model
            best_model = results_df.loc[results_df['Accuracy (%)'].astype(float).idxmax()]
            
            st.markdown(f"""
            <div class="prediction-card">
                <h3 style="color: white; margin: 0;">🏆 Best Model</h3>
                <div class="prediction-class">{best_model['Model']}</div>
                <div class="prediction-confidence">Accuracy: {best_model['Accuracy (%)']}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed metrics
            st.markdown("### 📈 Detailed Metrics")
            
            fig_all = px.bar(
                results_df,
                x='Model',
                y=['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'],
                barmode='group',
                title="Comprehensive Model Comparison"
            )
            st.plotly_chart(fig_all, use_container_width=True)
            
            # Model files
            st.markdown("### 💾 Saved Model Files")
            
            model_files = {
                'Random Forest': 'model_rf.pkl',
                'XGBoost': 'model_xgb.pkl',
                'Gradient Boosting': 'model_gb.pkl',
                'LightGBM': 'model_lgb.txt',
                'Label Encoder': 'label_encoder.pkl',
                'Scaler': 'scaler_classical.pkl'
            }
            
            files_status = []
            for name, file in model_files.items():
                path = os.path.join(config.OUTPUT_DIR, file)
                exists = "✅" if os.path.exists(path) else "❌"
                size = f"{os.path.getsize(path) / 1024:.2f} KB" if os.path.exists(path) else "N/A"
                files_status.append({
                    'File': name,
                    'Status': exists,
                    'Size': size
                })
            
            files_df = pd.DataFrame(files_status)
            st.dataframe(files_df, use_container_width=True)

if __name__ == "__main__":
    main()
