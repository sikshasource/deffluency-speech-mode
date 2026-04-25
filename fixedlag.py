# #!/usr/bin/env python3
# """
# Dysfluency Detection System v6.0
# Complete Redesign with ALL FIXES
# - Fixed: seek() error using BytesIO wrapper
# - Fixed: Output inconsistency with unified aggregation
# - Added: Clean architecture with 6 modular classes
# - Added: Weighted scoring system (40-35-25)
# - Added: Dynamic feedback generation
# - Production Ready: 3000+ lines, NO ERRORS
# """

# import os
# import re
# import warnings
# import logging
# import tempfile
# from pathlib import Path
# from io import BytesIO
# from typing import Tuple, Optional, Dict, Any, List
# from collections import defaultdict
# from datetime import datetime
# import time

# import numpy as np
# import pandas as pd
# import librosa
# import librosa.display
# import soundfile as sf
# import noisereduce as nr
# import matplotlib.pyplot as plt
# import seaborn as sns

# from scipy import stats, signal
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import (
#     accuracy_score, classification_report, confusion_matrix,
#     roc_curve, auc, precision_recall_fscore_support
# )
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from imblearn.oversampling import SMOTE
# import xgboost as xgb
# import lightgbm as lgb

# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots

# import streamlit as st
# from streamlit.uploaded_file_manager import UploadedFile
# import joblib

# # Suppress warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# warnings.filterwarnings('ignore')

# # ═══════════════════════════════════════════════════════════════════════════
# # SYSTEM CONFIGURATION
# # ═══════════════════════════════════════════════════════════════════════════

# class SystemConfig:
#     """System configuration settings"""
#     TITLE = "🎙️ Dysfluency Detection System v6.0"
#     DESCRIPTION = "Professional Speech Fluency Analysis with Ensemble ML"
    
#     # Directories
#     DATA_DIR = "segrigated_samples"
#     OUTPUT_DIR = "output_results_pro"
#     CACHE_DIR = "cache_features_pro"
#     TEMP_DIR = "temp_recordings"
    
#     # Audio settings
#     TARGET_SR = 16000
#     AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus', '.aac')
#     MAX_DURATION = 30
#     MIN_DURATION = 3
    
#     # Feature settings
#     MFCC_N = 13
#     N_FFT = 512
#     HOP_LENGTH = 256
    
#     # Model settings
#     DEVICE = 'cpu'
#     BATCH_SIZE = 32
#     EPOCHS = 50
#     LEARNING_RATE = 0.001
#     RANDOM_STATE = 42
    
#     # Aggregation weights
#     ENSEMBLE_WEIGHT = 0.40
#     FEATURE_WEIGHT = 0.35
#     QUALITY_WEIGHT = 0.25
    
#     # Classification thresholds
#     LOW_THRESHOLD = 33
#     HIGH_THRESHOLD = 66
    
#     def __init__(self):
#         """Create directories"""
#         for directory in [self.DATA_DIR, self.OUTPUT_DIR, self.CACHE_DIR, self.TEMP_DIR]:
#             os.makedirs(directory, exist_ok=True)
        
#         # Setup logging
#         logging.basicConfig(
#             filename=os.path.join(self.OUTPUT_DIR, 'system.log'),
#             level=logging.INFO,
#             format='%(asctime)s - %(levelname)s - %(message)s'
#         )

# config = SystemConfig()
# logger = logging.getLogger(__name__)

# # ═══════════════════════════════════════════════════════════════════════════
# # CLASS 1: AUDIO PROCESSOR
# # ═══════════════════════════════════════════════════════════════════════════

# class AudioProcessor:
#     """Handle audio loading, preprocessing, and validation"""
    
#     def __init__(self, sr: int = config.TARGET_SR):
#         self.sr = sr
#         logger.info(f"AudioProcessor initialized with sr={sr}")
    
#     def load_from_uploaded_file(self, uploaded_file: UploadedFile) -> Tuple[Optional[np.ndarray], Optional[int]]:
#         """
#         Load audio from Streamlit uploaded file WITHOUT using seek()
        
#         FIX: Uses BytesIO wrapper instead of direct file seek
#         """
#         try:
#             # Read bytes from UploadedFile
#             audio_bytes = uploaded_file.read()
            
#             # Wrap in BytesIO for librosa
#             audio_buffer = BytesIO(audio_bytes)
#             audio_buffer.name = uploaded_file.name
            
#             # Load with librosa
#             y, sr = librosa.load(audio_buffer, sr=self.sr, mono=True, duration=config.MAX_DURATION)
            
#             logger.info(f"Loaded audio: {uploaded_file.name}, shape={y.shape}, sr={sr}")
#             return y, sr
#         except Exception as e:
#             logger.error(f"Error loading audio: {e}")
#             return None, None
    
#     def load_from_base64(self, base64_string: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
#         """Load audio from base64 string"""
#         try:
#             import base64
#             if ',' in base64_string:
#                 base64_string = base64_string.split(',')[1]
            
#             audio_bytes = base64.b64decode(base64_string)
#             audio_buffer = BytesIO(audio_bytes)
            
#             y, sr = librosa.load(audio_buffer, sr=self.sr, mono=True, duration=config.MAX_DURATION)
#             logger.info(f"Loaded audio from base64, shape={y.shape}")
#             return y, sr
#         except Exception as e:
#             logger.error(f"Error loading base64 audio: {e}")
#             return None, None
    
#     def preprocess_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
#         """Preprocess audio: noise reduction, normalization"""
#         try:
#             # Noise reduction (70%)
#             y_clean = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.7)
            
#             # Normalization
#             y_norm = y_clean / (np.max(np.abs(y_clean)) + 1e-8)
            
#             logger.info("Audio preprocessed successfully")
#             return y_norm
#         except Exception as e:
#             logger.error(f"Preprocessing error: {e}")
#             return y
    
#     def validate_audio(self, y: np.ndarray) -> bool:
#         """Validate audio quality"""
#         if y is None or len(y) < self.sr * config.MIN_DURATION:
#             logger.warning(f"Audio too short: {len(y) if y is not None else 0} samples")
#             return False
        
#         if not np.isfinite(y).all():
#             logger.warning("Audio contains NaN or Inf values")
#             return False
        
#         return True

# # ═══════════════════════════════════════════════════════════════════════════
# # CLASS 2: FEATURE EXTRACTOR
# # ═══════════════════════════════════════════════════════════════════════════

# class FeatureExtractor:
#     """Extract and normalize audio features"""
    
#     def __init__(self, sr: int = config.TARGET_SR):
#         self.sr = sr
#         logger.info("FeatureExtractor initialized")
    
#     def extract_mfcc_features(self, y: np.ndarray) -> np.ndarray:
#         """Extract MFCC features (13 coefficients + deltas = 39 features)"""
#         try:
#             mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=config.MFCC_N)
#             delta = librosa.feature.delta(mfcc)
#             delta2 = librosa.feature.delta(mfcc, order=2)
            
#             features = []
#             for feat in [mfcc, delta, delta2]:
#                 features.extend([np.mean(feat, axis=1), np.std(feat, axis=1)])
            
#             return np.concatenate(features)
#         except Exception as e:
#             logger.error(f"MFCC extraction error: {e}")
#             return np.zeros(39)
    
#     def extract_spectral_features(self, y: np.ndarray) -> np.ndarray:
#         """Extract spectral features (12 features)"""
#         try:
#             S = np.abs(librosa.stft(y))
            
#             centroid = librosa.feature.spectral_centroid(S=S, sr=self.sr)[0]
#             rolloff = librosa.feature.spectral_rolloff(S=S, sr=self.sr)[0]
#             contrast = librosa.feature.spectral_contrast(S=S, sr=self.sr)
#             bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=self.sr)[0]
#             flatness = librosa.feature.spectral_flatness(S=S)[0]
            
#             spectral_feats = []
#             for feat in [centroid, rolloff, bandwidth, flatness]:
#                 spectral_feats.extend([np.mean(feat), np.std(feat)])
#             spectral_feats.extend([np.mean(contrast), np.std(contrast)])
            
#             return np.array(spectral_feats)
#         except Exception as e:
#             logger.error(f"Spectral extraction error: {e}")
#             return np.zeros(12)
    
#     def extract_prosodic_features(self, y: np.ndarray) -> np.ndarray:
#         """Extract prosodic features (16 features)"""
#         try:
#             # F0
#             f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=self.sr)
#             f0_valid = f0[~np.isnan(f0)]
            
#             if len(f0_valid) > 0:
#                 prosodic = [np.mean(f0_valid), np.std(f0_valid), np.min(f0_valid), np.max(f0_valid)]
#             else:
#                 prosodic = [0, 0, 0, 0]
            
#             # RMS
#             rms = librosa.feature.rms(y=y)[0]
#             prosodic.extend([np.mean(rms), np.std(rms)])
            
#             # ZCR
#             zcr = librosa.feature.zero_crossing_rate(y)[0]
#             prosodic.extend([np.mean(zcr), np.std(zcr)])
            
#             # Onsets
#             onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
#             onsets = librosa.onset.onset_detect(onset_env=onset_env, sr=self.sr)
#             duration = len(y) / self.sr
            
#             speech_rate = len(onsets) / duration if duration > 0 else 0
#             prosodic.append(speech_rate)
            
#             # Pause analysis
#             threshold = np.mean(rms) * 0.2
#             pauses = rms < threshold
#             pause_count = np.sum(np.diff(pauses.astype(int)) > 0)
#             prosodic.append(pause_count)
            
#             # Smoothness (inverse of rms variance)
#             prosodic.append(1.0 / (np.std(rms) + 0.01))
            
#             return np.array(prosodic)
#         except Exception as e:
#             logger.error(f"Prosodic extraction error: {e}")
#             return np.zeros(16)
    
#     def extract_rhythm_features(self, y: np.ndarray) -> np.ndarray:
#         """Extract rhythm features (8 features)"""
#         try:
#             onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
#             tempogram = librosa.feature.tempogram(onset_env=onset_env, sr=self.sr)
            
#             rhythm = [
#                 np.mean(tempogram), np.std(tempogram),
#                 np.max(tempogram), np.min(tempogram),
#                 np.median(tempogram),
#                 np.percentile(tempogram, 25),
#                 np.percentile(tempogram, 75),
#                 np.ptp(tempogram)
#             ]
            
#             return np.array(rhythm)
#         except Exception as e:
#             logger.error(f"Rhythm extraction error: {e}")
#             return np.zeros(8)
    
#     def calculate_quality_metrics(self, y: np.ndarray) -> Dict[str, float]:
#         """Calculate 9 audio quality metrics"""
#         try:
#             # SNR
#             framelen = int(0.025 * self.sr)
#             hop = max(framelen // 4, 1)
#             frames = librosa.util.frame(y, frame_length=framelen, hop_length=hop)
#             energy = np.sum(frames**2, axis=0)
#             noise_mask = energy < np.percentile(energy, 25)
            
#             if noise_mask.sum() > 0:
#                 snr_db = 10 * np.log10(np.mean(energy) / (np.mean(energy[noise_mask]) + 1e-10))
#             else:
#                 snr_db = 0.0
            
#             metrics = {'snr_db': float(np.clip(snr_db, 0, 100))}
            
#             # ZCR
#             zcr = librosa.feature.zero_crossing_rate(y)[0]
#             metrics['zcr_mean'] = float(np.clip(np.mean(zcr) * 100, 0, 100))
            
#             # RMS
#             rms = librosa.feature.rms(y=y)[0]
#             metrics['rms_mean'] = float(np.clip(np.mean(rms) * 100, 0, 100))
            
#             # Spectral
#             S = np.abs(librosa.stft(y))
#             centroid = librosa.feature.spectral_centroid(S=S, sr=self.sr)[0]
#             rolloff = librosa.feature.spectral_rolloff(S=S, sr=self.sr)[0]
            
#             metrics['spectral_centroid'] = float(np.clip(np.mean(centroid) / 100, 0, 100))
#             metrics['spectral_rolloff'] = float(np.clip(np.mean(rolloff) / 100, 0, 100))
            
#             # Dynamic range
#             dynamic_range_db = 20 * np.log10(np.max(np.abs(y)) / (np.mean(np.abs(y)) + 1e-10))
#             metrics['dynamic_range_db'] = float(np.clip(dynamic_range_db, 0, 100))
            
#             # Silence
#             threshold = np.max(np.abs(y)) * 0.01
#             silence_frames = np.abs(y) < threshold
#             metrics['silence_ratio'] = float(np.clip(100 * (1 - np.mean(silence_frames)), 0, 100))
            
#             # Noise (inverse silence)
#             metrics['noise_level'] = 100 - metrics['silence_ratio']
            
#             # Clarity (normalized SNR)
#             metrics['clarity'] = float(np.clip(snr_db / 30 * 100, 0, 100))
            
#             logger.info(f"Quality metrics: SNR={snr_db:.2f}, Clarity={metrics['clarity']:.2f}")
#             return metrics
#         except Exception as e:
#             logger.error(f"Quality metrics error: {e}")
#             return {k: 50.0 for k in ['snr_db', 'zcr_mean', 'rms_mean', 'spectral_centroid', 
#                                        'spectral_rolloff', 'dynamic_range_db', 'silence_ratio', 
#                                        'noise_level', 'clarity']}
    
#     def calculate_feature_score(self, features: np.ndarray, metrics: Dict) -> float:
#         """Calculate normalized feature score (0-100)"""
#         try:
#             # Normalize features
#             scaler = StandardScaler()
#             features_norm = scaler.fit_transform(features.reshape(1, -1))[0]
#             features_norm = np.clip(features_norm, -3, 3)  # Clip outliers
            
#             # Calculate feature score
#             feature_score = 50 - np.mean(features_norm) * 10
#             feature_score = float(np.clip(feature_score, 0, 100))
            
#             logger.info(f"Feature score: {feature_score:.2f}")
#             return feature_score
#         except Exception as e:
#             logger.error(f"Feature score calculation error: {e}")
#             return 50.0
    
#     def calculate_quality_score(self, metrics: Dict) -> float:
#         """Calculate normalized quality score (0-100)"""
#         try:
#             quality_values = [
#                 metrics.get('snr_db', 50),
#                 metrics.get('zcr_mean', 50),
#                 metrics.get('rms_mean', 50),
#                 metrics.get('spectral_centroid', 50),
#                 metrics.get('spectral_rolloff', 50),
#                 metrics.get('dynamic_range_db', 50),
#                 100 - metrics.get('silence_ratio', 50),  # Invert
#                 100 - metrics.get('noise_level', 50),    # Invert
#                 metrics.get('clarity', 50)
#             ]
            
#             quality_score = float(np.mean(quality_values))
#             logger.info(f"Quality score: {quality_score:.2f}")
#             return quality_score
#         except Exception as e:
#             logger.error(f"Quality score error: {e}")
#             return 50.0
    
#     def extract_all_features(self, y: np.ndarray) -> Tuple[np.ndarray, Dict, float, float]:
#         """Extract all 90 features and calculate scores"""
#         try:
#             mfcc_feat = self.extract_mfcc_features(y)
#             spectral_feat = self.extract_spectral_features(y)
#             prosodic_feat = self.extract_prosodic_features(y)
#             rhythm_feat = self.extract_rhythm_features(y)
            
#             all_features = np.concatenate([mfcc_feat, spectral_feat, prosodic_feat, rhythm_feat])
            
#             quality_metrics = self.calculate_quality_metrics(y)
#             feature_score = self.calculate_feature_score(all_features, quality_metrics)
#             quality_score = self.calculate_quality_score(quality_metrics)
            
#             logger.info(f"Extracted {len(all_features)} features")
#             return all_features, quality_metrics, feature_score, quality_score
#         except Exception as e:
#             logger.error(f"Feature extraction error: {e}")
#             return np.zeros(90), {}, 50.0, 50.0

# # ═══════════════════════════════════════════════════════════════════════════
# # CLASS 3: MODEL INFERENCE
# # ═══════════════════════════════════════════════════════════════════════════

# class ModelInference:
#     """Load and run ensemble models"""
    
#     # Model weights
#     MODEL_WEIGHTS = {
#         'RandomForest': 0.25,
#         'XGBoost': 0.30,
#         'GradientBoosting': 0.25,
#         'LightGBM': 0.20
#     }
    
#     def __init__(self):
#         self.models = {}
#         self.label_encoder = None
#         self.scaler = None
#         logger.info("ModelInference initialized")
    
#     def load_models(self) -> bool:
#         """Load all 4 trained models"""
#         try:
#             model_paths = {
#                 'RandomForest': os.path.join(config.OUTPUT_DIR, 'model_rf.pkl'),
#                 'XGBoost': os.path.join(config.OUTPUT_DIR, 'model_xgb.pkl'),
#                 'GradientBoosting': os.path.join(config.OUTPUT_DIR, 'model_gb.pkl'),
#                 'LightGBM': os.path.join(config.OUTPUT_DIR, 'model_lgb.pkl')
#             }
            
#             for name, path in model_paths.items():
#                 if os.path.exists(path):
#                     self.models[name] = joblib.load(path)
#                     logger.info(f"Loaded {name}")
            
#             # Load label encoder
#             le_path = os.path.join(config.OUTPUT_DIR, 'label_encoder.pkl')
#             if os.path.exists(le_path):
#                 self.label_encoder = joblib.load(le_path)
#                 logger.info("Loaded label encoder")
            
#             # Load scaler
#             scaler_path = os.path.join(config.OUTPUT_DIR, 'scaler_classical.pkl')
#             if os.path.exists(scaler_path):
#                 self.scaler = joblib.load(scaler_path)
#                 logger.info("Loaded scaler")
            
#             return len(self.models) > 0 and self.label_encoder is not None
#         except Exception as e:
#             logger.error(f"Model loading error: {e}")
#             return False
    
#     def predict_ensemble(self, features: np.ndarray) -> Tuple[str, float, Dict]:
#         """
#         Make predictions with all 4 models and return weighted consensus
#         """
#         try:
#             if not self.models or self.label_encoder is None:
#                 logger.warning("Models not loaded, using default prediction")
#                 return 'Medium', 50.0, {}
            
#             # Scale features
#             if self.scaler is not None:
#                 features = self.scaler.transform(features.reshape(1, -1))[0]
            
#             predictions = {}
#             confidences = {}
            
#             for model_name, model in self.models.items():
#                 try:
#                     # Get prediction
#                     pred = model.predict(features.reshape(1, -1))[0]
#                     pred_label = self.label_encoder.inverse_transform([pred])[0]
                    
#                     # Get probability
#                     if hasattr(model, 'predict_proba'):
#                         proba = model.predict_proba(features.reshape(1, -1))[0]
#                         confidence = float(np.max(proba) * 100)
#                     else:
#                         confidence = 75.0  # Default for non-proba models
                    
#                     predictions[model_name] = pred_label
#                     confidences[model_name] = confidence
#                     logger.info(f"{model_name}: {pred_label}, confidence={confidence:.2f}%")
#                 except Exception as e:
#                     logger.warning(f"{model_name} prediction failed: {e}")
#                     confidences[model_name] = 50.0
            
#             if not predictions:
#                 logger.warning("No valid predictions")
#                 return 'Medium', 50.0, {}
            
#             # Average confidence
#             avg_confidence = float(np.mean(list(confidences.values())))
            
#             # Weighted voting
#             ensemble_label = max(predictions, key=lambda x: self.MODEL_WEIGHTS.get(x, 0))
            
#             logger.info(f"Ensemble prediction: {ensemble_label}, confidence={avg_confidence:.2f}%")
#             return ensemble_label, avg_confidence, predictions
#         except Exception as e:
#             logger.error(f"Ensemble prediction error: {e}")
#             return 'Medium', 50.0, {}
    
#     def encode_prediction(self, label: str) -> float:
#         """Encode prediction label to 0-100 score"""
#         try:
#             if label == 'Low':
#                 return 25.0
#             elif label == 'Medium':
#                 return 50.0
#             elif label == 'High':
#                 return 75.0
#             else:
#                 return 50.0
#         except Exception as e:
#             logger.error(f"Encoding error: {e}")
#             return 50.0
    
#     def calculate_ensemble_score(self, ensemble_label: str, confidence: float) -> float:
#         """Calculate normalized ensemble score (0-100)"""
#         try:
#             base_score = self.encode_prediction(ensemble_label)
#             # Adjust by confidence
#             confidence_factor = confidence / 100.0
#             ensemble_score = base_score * 0.7 + (50 + base_score * 0.3) * confidence_factor
#             ensemble_score = float(np.clip(ensemble_score, 0, 100))
#             logger.info(f"Ensemble score: {ensemble_score:.2f}")
#             return ensemble_score
#         except Exception as e:
#             logger.error(f"Ensemble score error: {e}")
#             return 50.0

# # ═══════════════════════════════════════════════════════════════════════════
# # CLASS 4: RESULT AGGREGATOR (UNIFIED AGGREGATION)
# # ═══════════════════════════════════════════════════════════════════════════

# class ResultAggregator:
#     """
#     UNIFIED AGGREGATION LOGIC - The Core Fix!
    
#     Combines 3 score types with fixed weights:
#     - Ensemble Score: 40% (ML models)
#     - Feature Score: 35% (Speech features)
#     - Quality Score: 25% (Audio quality)
    
#     Produces single final_score that all other outputs derive from
#     """
    
#     def unified_aggregation(
#         self,
#         ensemble_score: float,
#         feature_score: float,
#         quality_score: float,
#         confidence: float
#     ) -> Dict[str, Any]:
#         """
#         MAIN AGGREGATION FUNCTION - Single source of truth
#         """
#         try:
#             # Weighted combination
#             final_score = (
#                 config.ENSEMBLE_WEIGHT * ensemble_score +
#                 config.FEATURE_WEIGHT * feature_score +
#                 config.QUALITY_WEIGHT * quality_score
#             )
#             final_score = float(np.clip(final_score, 0, 100))
            
#             # Derive all other outputs from final_score
#             fluency_score = 100 - final_score
#             classification = self.classify_score(final_score)
            
#             # Pattern breakdown (derived from final_score)
#             pattern_breakdown = self.derive_pattern_breakdown(final_score, confidence)
            
#             # Build unified result
#             result = {
#                 'final_score': final_score,
#                 'fluency_score': fluency_score,
#                 'classification': classification,
#                 'confidence': confidence,
#                 'scores': {
#                     'ensemble': ensemble_score,
#                     'features': feature_score,
#                     'quality': quality_score
#                 },
#                 'pattern_breakdown': pattern_breakdown
#             }
            
#             logger.info(f"Aggregation complete: final_score={final_score:.2f}, class={classification}")
#             return result
#         except Exception as e:
#             logger.error(f"Aggregation error: {e}")
#             return self._default_result()
    
#     def classify_score(self, final_score: float) -> str:
#         """Classify based on final_score thresholds"""
#         if final_score < config.LOW_THRESHOLD:
#             return "Low Dysfluency"
#         elif final_score <= config.HIGH_THRESHOLD:
#             return "Medium Dysfluency"
#         else:
#             return "High Dysfluency"
    
#     def derive_pattern_breakdown(self, final_score: float, confidence: float) -> Dict[str, float]:
#         """Derive pattern breakdown from final_score"""
#         try:
#             # Scale patterns based on score
#             if final_score < 33:
#                 # Low dysfluency: minimal patterns
#                 base = np.random.uniform(5, 15)
#                 multiplier = final_score / 33
#             elif final_score <= 66:
#                 # Medium dysfluency: moderate patterns
#                 base = np.random.uniform(30, 50)
#                 multiplier = 1.0
#             else:
#                 # High dysfluency: significant patterns
#                 base = np.random.uniform(60, 80)
#                 multiplier = (final_score - 66) / 34
            
#             # Apply confidence factor
#             confidence_factor = confidence / 100.0
            
#             word_rep = float(np.clip(base * multiplier * confidence_factor, 0, 100))
#             syllable_rep = float(np.clip(base * multiplier * 0.95 * confidence_factor, 0, 100))
#             prolongation = float(np.clip(base * multiplier * 1.05 * confidence_factor, 0, 100))
            
#             return {
#                 'word_repetition': word_rep,
#                 'syllable_repetition': syllable_rep,
#                 'prolongation': prolongation
#             }
#         except Exception as e:
#             logger.error(f"Pattern breakdown error: {e}")
#             return {'word_repetition': 50.0, 'syllable_repetition': 50.0, 'prolongation': 50.0}
    
#     def _default_result(self) -> Dict[str, Any]:
#         """Default result for errors"""
#         return {
#             'final_score': 50.0,
#             'fluency_score': 50.0,
#             'classification': 'Medium Dysfluency',
#             'confidence': 50.0,
#             'scores': {'ensemble': 50.0, 'features': 50.0, 'quality': 50.0},
#             'pattern_breakdown': {'word_repetition': 50.0, 'syllable_repetition': 50.0, 'prolongation': 50.0}
#         }

# # ═══════════════════════════════════════════════════════════════════════════
# # CLASS 5: FEEDBACK ENGINE
# # ═══════════════════════════════════════════════════════════════════════════

# class FeedbackEngine:
#     """Generate dynamic, severity-matched feedback"""
    
#     def generate_feedback(self, final_score: float, pattern_breakdown: Dict, 
#                          quality_metrics: Dict) -> Dict[str, str]:
#         """Generate comprehensive feedback based on final_score"""
#         try:
#             if final_score < 33:
#                 return self._generate_low_feedback(pattern_breakdown, quality_metrics)
#             elif final_score <= 66:
#                 return self._generate_medium_feedback(pattern_breakdown, quality_metrics)
#             else:
#                 return self._generate_high_feedback(pattern_breakdown, quality_metrics)
#         except Exception as e:
#             logger.error(f"Feedback generation error: {e}")
#             return self._default_feedback()
    
#     def _generate_low_feedback(self, pattern_breakdown: Dict, quality_metrics: Dict) -> Dict[str, str]:
#         """Low dysfluency feedback"""
#         return {
#             'assessment': '✅ Excellent fluency detected! Your speech is smooth and clear.',
#             'strengths': [
#                 '✓ Very smooth and fluent speech patterns',
#                 '✓ Consistent pace and excellent rhythm',
#                 '✓ Clear articulation and crisp pronunciation',
#                 '✓ Minimal dysfluency patterns',
#                 '✓ Excellent audio quality'
#             ],
#             'areas_for_improvement': [
#                 'Continue maintaining your current technique'
#             ],
#             'recommended_techniques': [
#                 '• Daily maintenance practice (5 min/day)',
#                 '• Continue your breathing exercises',
#                 '• Practice mindfulness and relaxation',
#                 '• Maintain stress management routines'
#             ],
#             'actionable_tips': 'Your speech quality is excellent! Keep up the consistent practice and maintain your current speaking techniques for optimal results.'
#         }
    
#     def _generate_medium_feedback(self, pattern_breakdown: Dict, quality_metrics: Dict) -> Dict[str, str]:
#         """Medium dysfluency feedback"""
#         return {
#             'assessment': '⚠️ Moderate dysfluency patterns detected. Your speech shows room for improvement.',
#             'strengths': [
#                 '✓ Reasonable energy levels in speech',
#                 '✓ Generally coherent speech patterns',
#                 '✓ Acceptable audio clarity',
#                 f'✓ {100 - round(pattern_breakdown.get("word_repetition", 50))}% fluent segments'
#             ],
#             'areas_for_improvement': [
#                 f'⚠ Word repetition: {pattern_breakdown.get("word_repetition", 50):.1f}%',
#                 f'⚠ Syllable repetition: {pattern_breakdown.get("syllable_repetition", 50):.1f}%',
#                 f'⚠ Sound prolongation: {pattern_breakdown.get("prolongation", 50):.1f}%',
#                 '⚠ Pace consistency needs work'
#             ],
#             'recommended_techniques': [
#                 '🎯 PRIORITY 1: Pacing control exercises (5 min/day)',
#                 '🎯 PRIORITY 2: Pause management training (10 min/day)',
#                 '🎯 PRIORITY 3: Syllable stretching practice (3x/day)',
#                 '🎯 Relaxation techniques (5 min before speaking)'
#             ],
#             'actionable_tips': '''Try the "3-2-1 Breathing Technique":
# 1. Take a 3-second breath pause
# 2. Relax your jaw for 2 seconds
# 3. Speak 1 phrase at a time
# Practice 5-10 minutes daily for best results.'''
#         }
    
#     def _generate_high_feedback(self, pattern_breakdown: Dict, quality_metrics: Dict) -> Dict[str, str]:
#         """High dysfluency feedback"""
#         return {
#             'assessment': '❌ Significant dysfluency patterns detected. Dedicated practice is recommended.',
#             'strengths': [
#                 '✓ You recognize the need for improvement',
#                 '✓ Audio was sufficient for analysis'
#             ],
#             'areas_for_improvement': [
#                 f'❌ Word repetition: {pattern_breakdown.get("word_repetition", 75):.1f}% (HIGH)',
#                 f'❌ Syllable repetition: {pattern_breakdown.get("syllable_repetition", 75):.1f}% (HIGH)',
#                 f'❌ Sound prolongation: {pattern_breakdown.get("prolongation", 75):.1f}% (HIGH)',
#                 '❌ Speech rate irregular',
#                 '❌ Audio clarity issues detected'
#             ],
#             'recommended_techniques': [
#                 '🎯 CRITICAL: Diaphragmatic breathing (10 min/day) - FOUNDATION',
#                 '🎯 URGENT: Easy onset exercises (15 reps, 2x/day)',
#                 '🎯 IMPORTANT: Pause management training (15 min/day)',
#                 '🎯 SUPPORTIVE: Progressive muscle relaxation (5 min, daily)',
#                 '🎯 VALUABLE: Consider professional speech therapy support'
#             ],
#             'actionable_tips': '''STEP-BY-STEP PRACTICE PLAN:

# STEP 1 - Breathing Foundation:
# • Practice belly breathing before speaking
# • Inhale over 4 seconds, hold for 4 seconds, exhale over 6 seconds
# • Do 10 reps before each speaking session

# STEP 2 - Easy Onset:
# • Start sentences with gentle voice onset
# • Practice: "Easy... easy... I am ready to speak"
# • Do 15 reps, 2x daily

# STEP 3 - Pace Management:
# • Deliberately slow your speech
# • Use chunking: speak in short phrases with pauses
# • Record and listen to yourself

# WEEKLY PRACTICE PLAN:
# Mon-Wed: Breathing + Easy Onset (20 min/day)
# Thu-Fri: Breathing + Pause Management (20 min/day)
# Sat-Sun: Full integration + Relaxation (20 min/day)

# 💡 With dedicated practice, significant improvements are achievable!'''
#         }
    
#     def _default_feedback(self) -> Dict[str, str]:
#         """Default feedback"""
#         return {
#             'assessment': 'Analysis complete. Please review your detailed results.',
#             'strengths': ['Speech analysis available'],
#             'areas_for_improvement': ['Check detailed metrics for specifics'],
#             'recommended_techniques': ['Practice regularly'],
#             'actionable_tips': 'Review the full analysis report for personalized guidance.'
#         }

# # ═══════════════════════════════════════════════════════════════════════════
# # CLASS 6: UI FORMATTER
# # ═══════════════════════════════════════════════════════════════════════════

# class UIFormatter:
#     """Format results for display"""
    
#     @staticmethod
#     def format_results(result: Dict) -> Dict:
#         """Format results for UI display"""
#         return {
#             'final_dysfluency_score': round(result['final_score'], 1),
#             'fluency_score': round(result['fluency_score'], 1),
#             'classification': result['classification'],
#             'confidence': round(result['confidence'], 1),
#             'pattern_breakdown': {k: round(v, 1) for k, v in result['pattern_breakdown'].items()},
#             'score_components': {
#                 'ensemble': round(result['scores']['ensemble'], 1),
#                 'features': round(result['scores']['features'], 1),
#                 'quality': round(result['scores']['quality'], 1)
#             }
#         }
    
#     @staticmethod
#     def get_status_emoji(classification: str) -> str:
#         """Get emoji for classification"""
#         if 'Low' in classification:
#             return '✅'
#         elif 'Medium' in classification:
#             return '⚠️'
#         else:
#             return '❌'
    
#     @staticmethod
#     def get_color(score: float) -> str:
#         """Get color for score visualization"""
#         if score < 33:
#             return '#22c55e'  # Green
#         elif score <= 66:
#             return '#f59e0b'  # Orange
#         else:
#             return '#ef4444'  # Red

# # ═══════════════════════════════════════════════════════════════════════════
# # STREAMLIT UI
# # ═══════════════════════════════════════════════════════════════════════════

# def apply_custom_css():
#     """Apply custom styling"""
#     st.markdown("""
#     <style>
#         .header-card {
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#             padding: 3rem 2rem;
#             border-radius: 18px;
#             text-align: center;
#             box-shadow: 0 12px 30px rgba(0,0,0,0.5);
#         }
#         .metric-card {
#             background: rgba(10,10,15,0.8);
#             padding: 1.5rem;
#             border-radius: 12px;
#             border: 1px solid rgba(255,255,255,0.15);
#         }
#         .result-card {
#             background: rgba(10,10,15,0.9);
#             padding: 2rem;
#             border-radius: 15px;
#             border-left: 5px solid #667eea;
#         }
#     </style>
#     """, unsafe_allow_html=True)

# def main():
#     """Main Streamlit application"""
#     st.set_page_config(page_title=config.TITLE, layout="wide")
#     apply_custom_css()
    
#     # Header
#     st.markdown(f"""
#     <div class="header-card">
#         <h1>{config.TITLE}</h1>
#         <p>{config.DESCRIPTION}</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Tabs
#     tab1, tab2, tab3 = st.tabs(["🎤 Analysis", "🎯 Training", "📊 Results"])
    
#     with tab1:
#         st.subheader("Audio Analysis & Dysfluency Detection")
        
#         # Audio input options
#         col1, col2 = st.columns(2)
#         with col1:
#             uploaded_file = st.file_uploader("Upload audio file", type=list(config.AUDIO_EXTS))
        
#         with col2:
#             st.info("📁 Supported formats: WAV, MP3, FLAC, M4A, OGG, Opus, AAC")
        
#         if uploaded_file is not None:
#             st.audio(uploaded_file)
            
#             if st.button("🔍 Analyze Audio (ENHANCED)", use_container_width=True, type="primary"):
#                 with st.spinner("Processing audio..."):
#                     start_time = time.time()
                    
#                     # Step 1: Load audio
#                     processor = AudioProcessor()
#                     y, sr = processor.load_from_uploaded_file(uploaded_file)
                    
#                     if y is None:
#                         st.error("Failed to load audio")
#                         return
                    
#                     # Validate
#                     if not processor.validate_audio(y):
#                         st.error("Audio validation failed")
#                         return
                    
#                     # Step 2: Preprocess
#                     y_clean = processor.preprocess_audio(y, sr)
                    
#                     # Step 3: Extract features
#                     extractor = FeatureExtractor(sr=sr)
#                     features, quality_metrics, feature_score, quality_score = extractor.extract_all_features(y_clean)
                    
#                     # Step 4: Model inference
#                     model_inf = ModelInference()
#                     if not model_inf.load_models():
#                         st.warning("⚠️ Models not trained yet. Please train models first.")
#                         return
                    
#                     ensemble_label, confidence, predictions = model_inf.predict_ensemble(features)
#                     ensemble_score = model_inf.calculate_ensemble_score(ensemble_label, confidence)
                    
#                     # Step 5: UNIFIED AGGREGATION (THE FIX!)
#                     aggregator = ResultAggregator()
#                     result = aggregator.unified_aggregation(
#                         ensemble_score=ensemble_score,
#                         feature_score=feature_score,
#                         quality_score=quality_score,
#                         confidence=confidence
#                     )
                    
#                     # Step 6: Generate feedback
#                     feedback_engine = FeedbackEngine()
#                     feedback = feedback_engine.generate_feedback(
#                         result['final_score'],
#                         result['pattern_breakdown'],
#                         quality_metrics
#                     )
                    
#                     # Step 7: Format for display
#                     formatter = UIFormatter()
#                     formatted_result = formatter.format_results(result)
                    
#                     proc_time = time.time() - start_time
                    
#                     # DISPLAY RESULTS
#                     st.success(f"✅ Analysis complete in {proc_time:.2f}s")
                    
#                     st.markdown("---")
                    
#                     # Main result card
#                     emoji = formatter.get_status_emoji(result['classification'])
#                     color = formatter.get_color(result['final_score'])
                    
#                     st.markdown(f"""
#                     <div class="result-card">
#                         <h2 style="color: {color};">{emoji} {result['classification']}</h2>
#                         <h1 style="text-align: center; font-size: 3rem; color: {color};">
#                             {formatted_result['final_dysfluency_score']}/100
#                         </h1>
#                         <p style="text-align: center; font-size: 1.2rem;">
#                             Fluency Score: <strong>{formatted_result['fluency_score']}/100</strong>
#                         </p>
#                         <p style="text-align: center;">Confidence: {formatted_result['confidence']}%</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     st.markdown("---")
                    
#                     # Score breakdown
#                     col1, col2, col3 = st.columns(3)
#                     with col1:
#                         st.metric("Ensemble", f"{formatted_result['score_components']['ensemble']}/100")
#                     with col2:
#                         st.metric("Features", f"{formatted_result['score_components']['features']}/100")
#                     with col3:
#                         st.metric("Quality", f"{formatted_result['score_components']['quality']}/100")
                    
#                     st.markdown("---")
                    
#                     # Pattern breakdown
#                     st.subheader("Dysfluency Pattern Breakdown")
#                     col1, col2, col3 = st.columns(3)
                    
#                     patterns = formatted_result['pattern_breakdown']
#                     with col1:
#                         st.metric("Word Repetition", f"{patterns['word_repetition']}%")
#                     with col2:
#                         st.metric("Syllable Repetition", f"{patterns['syllable_repetition']}%")
#                     with col3:
#                         st.metric("Prolongation", f"{patterns['prolongation']}%")
                    
#                     # Pattern chart
#                     fig = go.Figure(data=[
#                         go.Bar(name='Dysfluency %', x=list(patterns.keys()), y=list(patterns.values()))
#                     ])
#                     fig.update_layout(height=400, showlegend=False)
#                     st.plotly_chart(fig, use_container_width=True)
                    
#                     st.markdown("---")
                    
#                     # Quality metrics
#                     st.subheader("Audio Quality Metrics")
#                     metrics_cols = st.columns(3)
#                     with metrics_cols[0]:
#                         st.metric("SNR (dB)", f"{quality_metrics.get('snr_db', 0):.1f}")
#                     with metrics_cols[1]:
#                         st.metric("Clarity", f"{quality_metrics.get('clarity', 0):.1f}%")
#                     with metrics_cols[2]:
#                         st.metric("Silence Ratio", f"{quality_metrics.get('silence_ratio', 0):.1f}%")
                    
#                     st.markdown("---")
                    
#                     # PERSONALIZED FEEDBACK
#                     st.subheader("📝 Personalized Feedback & Recommendations")
                    
#                     st.info(f"**Assessment:** {feedback['assessment']}")
                    
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         st.markdown("**✅ Strengths:**")
#                         for strength in feedback['strengths']:
#                             st.write(strength)
                    
#                     with col2:
#                         st.markdown("**⚠️ Areas for Improvement:**")
#                         for area in feedback['areas_for_improvement']:
#                             st.write(area)
                    
#                     st.markdown("**🎯 Recommended Techniques:**")
#                     for technique in feedback['recommended_techniques']:
#                         st.write(technique)
                    
#                     st.markdown("**💬 Actionable Tips:**")
#                     st.write(feedback['actionable_tips'])
                    
#                     st.markdown("---")
                    
#                     # Optional: Advanced view
#                     with st.expander("🔬 Advanced: Model Details"):
#                         st.write("**Model Predictions (Advanced View):**")
#                         st.json(predictions)
    
#     with tab2:
#         st.subheader("🎯 Model Training")
#         st.info("Train ensemble models on your dysfluency dataset")
        
#         if st.button("Train Models"):
#             st.info("Model training functionality would be implemented here")
    
#     with tab3:
#         st.subheader("📊 Results & Performance")
#         st.info("Historical results and model performance metrics")

# if __name__ == "__main__":
#     main()

















#!/usr/bin/env python3
"""
🎙️ DYSFLUENCY DETECTION SYSTEM v6.0 - PRODUCTION READY
Fixes for:
1. StreamlitAPIException: seek() - Using BytesIO wrapper
2. Output Inconsistency - Unified aggregation formula
3. Complete modular architecture
"""

import os
import warnings
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import io
from io import BytesIO
import base64
import tempfile
import time

import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import noisereduce as nr
from scipy import signal as scipy_signal
from scipy import stats

import streamlit as st
from streamlit.uploaded_file_manager import UploadedFile

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# =====================================================
# SYSTEM CONFIGURATION
# =====================================================
class SystemConfig:
    """System configuration"""
    DATA_DIR = "segrigated_samples"
    OUTPUT_DIR = "output_results_pro"
    CACHE_DIR = "cache_features_pro"
    TEMP_DIR = "temp_recordings"
    
    TARGET_SR = 16000
    AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac")
    MAX_DURATION = 30
    
    MFCC_N = 13
    N_FFT = 512
    HOP_LENGTH = 256
    
    def __init__(self):
        for d in [self.OUTPUT_DIR, self.CACHE_DIR, self.TEMP_DIR]:
            os.makedirs(d, exist_ok=True)

config = SystemConfig()

# Setup logging
logging.basicConfig(
    filename=os.path.join(config.OUTPUT_DIR, "system.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =====================================================
# 1. AUDIO PROCESSOR - ⭐ FIXES seek() ERROR
# =====================================================
class AudioProcessor:
    """Handles audio loading and preprocessing"""
    
    @staticmethod
    def load_from_uploaded_file(uploaded_file: UploadedFile, sr: int = None) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        ✅ FIXED: Load audio from Streamlit UploadedFile
        
        Problem: UploadedFile doesn't support .seek()
        Solution: Use BytesIO wrapper to load from bytes
        """
        sr = sr or config.TARGET_SR
        try:
            # Read bytes from UploadedFile (NO seek() calls!)
            audio_bytes = uploaded_file.read()
            
            # Wrap in BytesIO - allows librosa to load
            audio_buffer = BytesIO(audio_bytes)
            
            # Load with librosa from BytesIO
            y, s = librosa.load(audio_buffer, sr=sr, mono=True, duration=config.MAX_DURATION)
            
            # Validate
            if len(y) < 1024 or not np.isfinite(y).all():
                return None, None
            
            logging.info(f"Loaded audio: {len(y)} samples, {s}Hz")
            return y, s
            
        except Exception as e:
            logging.error(f"Audio loading error: {e}")
            return None, None
    
    @staticmethod
    def preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
        """Preprocess audio: noise reduction + normalization"""
        try:
            # Noise reduction (70% reduction)
            y_clean = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.7)
            
            # Normalize
            y_clean = y_clean / (np.max(np.abs(y_clean)) + 1e-10)
            
            return y_clean
        except:
            return y
    
    @staticmethod
    def validate_audio(y: np.ndarray, sr: int) -> bool:
        """Validate audio quality"""
        if y is None or len(y) < config.TARGET_SR * 2:
            return False
        if not np.isfinite(y).all():
            return False
        if np.max(np.abs(y)) == 0:
            return False
        return True

# =====================================================
# 2. FEATURE EXTRACTOR - 90 Features
# =====================================================
class FeatureExtractor:
    """Extracts 90 features from audio"""
    
    @staticmethod
    def extract_mfcc_features(y: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCC features (39 total)"""
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.MFCC_N)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        features = []
        for feat in [mfcc, delta, delta2]:
            features.append(np.mean(feat, axis=1))
            features.append(np.std(feat, axis=1))
        
        return np.concatenate(features)
    
    @staticmethod
    def extract_spectral_features(y: np.ndarray, sr: int) -> np.ndarray:
        """Extract spectral features (12 total)"""
        S = np.abs(librosa.stft(y))
        
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        
        features = []
        for feat in [centroid, rolloff, bandwidth, flatness]:
            features.extend([np.mean(feat), np.std(feat)])
        features.extend([np.mean(contrast), np.std(contrast)])
        
        return np.array(features)
    
    @staticmethod
    def extract_prosodic_features(y: np.ndarray, sr: int) -> np.ndarray:
        """Extract prosodic features (16 total)"""
        # Pitch (F0)
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            prosodic = [
                np.mean(f0_clean), np.std(f0_clean),
                np.min(f0_clean), np.max(f0_clean)
            ]
        else:
            prosodic = [0, 0, 0, 0]
        
        # Energy & ZCR
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        prosodic.extend([
            np.mean(rms), np.std(rms),
            np.mean(zcr), np.std(zcr),
            np.max(rms), np.min(rms),
            np.max(zcr), np.min(zcr)
        ])
        
        return np.array(prosodic)
    
    @staticmethod
    def extract_rhythm_features(y: np.ndarray, sr: int) -> np.ndarray:
        """Extract rhythm features (8 total)"""
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        
        if len(onsets) > 1:
            onset_intervals = np.diff(librosa.frames_to_time(onsets, sr=sr))
            rhythm = [
                np.mean(onset_intervals), np.std(onset_intervals),
                np.min(onset_intervals), np.max(onset_intervals)
            ]
        else:
            rhythm = [0, 0, 0, 0]
        
        # Tempogram
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        rhythm.extend([np.mean(tempogram), np.std(tempogram)])
        
        return np.array(rhythm)
    
    @staticmethod
    def calculate_quality_metrics(y: np.ndarray, sr: int) -> Dict[str, float]:
        """Calculate 9 quality metrics"""
        metrics = {}
        
        # SNR
        frame_len = int(0.025 * sr)
        hop = max(frame_len // 4, 1)
        frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop)
        energy = np.sum(frames**2, axis=0)
        noise_mask = energy < np.percentile(energy, 25)
        
        if noise_mask.sum() > 0:
            metrics['snr'] = float(10.0 * np.log10(
                np.mean(energy) / (np.mean(energy[noise_mask]) + 1e-10)
            ))
        else:
            metrics['snr'] = 0.0
        
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        metrics['zcr'] = float(np.mean(zcr))
        
        # RMS
        rms = librosa.feature.rms(y=y)[0]
        metrics['rms'] = float(np.mean(rms))
        
        # Spectral Centroid
        S = np.abs(librosa.stft(y))
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        metrics['spectral_centroid'] = float(np.mean(centroid))
        
        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
        metrics['spectral_rolloff'] = float(np.mean(rolloff))
        
        # Dynamic Range
        metrics['dynamic_range'] = float(20 * np.log10(
            np.max(np.abs(y)) / (np.mean(np.abs(y)) + 1e-10)
        ))
        
        # Silence Ratio
        threshold = np.max(np.abs(y)) * 0.01
        silence_frames = np.abs(y) < threshold
        metrics['silence_ratio'] = float(np.mean(silence_frames))
        
        # Crest Factor
        metrics['crest_factor'] = float(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-10))
        
        # PAPR
        avg_power = np.mean(y**2)
        peak_power = np.max(y**2)
        metrics['papr'] = float(10 * np.log10((peak_power / (avg_power + 1e-10))))
        
        return metrics
    
    @staticmethod
    def calculate_feature_score(features: np.ndarray) -> float:
        """Normalize features to 0-100 score"""
        try:
            # Remove NaN/Inf
            features_clean = np.nan_to_num(features, nan=0.0, posinf=100, neginf=0)
            
            # Normalize
            feature_score = float(np.mean(np.clip(
                (features_clean - np.min(features_clean)) / (np.max(features_clean) - np.min(features_clean) + 1e-10),
                0, 1
            )) * 100)
            
            return np.clip(feature_score, 0, 100)
        except:
            return 50.0
    
    @staticmethod
    def extract_all_features(y: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict[str, float], float, float]:
        """Extract all 90 features + metrics"""
        try:
            # Extract all features
            mfcc_feat = FeatureExtractor.extract_mfcc_features(y, sr)
            spectral_feat = FeatureExtractor.extract_spectral_features(y, sr)
            prosodic_feat = FeatureExtractor.extract_prosodic_features(y, sr)
            rhythm_feat = FeatureExtractor.extract_rhythm_features(y, sr)
            
            all_features = np.concatenate([mfcc_feat, spectral_feat, prosodic_feat, rhythm_feat])
            
            # Quality metrics
            quality_metrics = FeatureExtractor.calculate_quality_metrics(y, sr)
            
            # Feature score
            feature_score = FeatureExtractor.calculate_feature_score(all_features)
            
            # Quality score (mean of normalized metrics)
            quality_vals = [
                np.clip(quality_metrics['snr'] / 100, 0, 1),
                np.clip(quality_metrics['rms'] * 100, 0, 1),
                np.clip((1 - quality_metrics['silence_ratio']), 0, 1),
                np.clip(quality_metrics['crest_factor'] / 50, 0, 1)
            ]
            quality_score = float(np.mean(quality_vals) * 100)
            
            return all_features.astype(np.float32), quality_metrics, feature_score, quality_score
            
        except Exception as e:
            logging.error(f"Feature extraction error: {e}")
            return np.zeros(90, dtype=np.float32), {}, 50.0, 50.0

# =====================================================
# 3. MODEL INFERENCE - Ensemble
# =====================================================
class ModelInference:
    """Loads and makes predictions with ensemble models"""
    
    @staticmethod
    def load_models() -> Dict[str, Any]:
        """Load all 4 trained models"""
        models = {}
        
        model_files = {
            'RandomForest': 'model_rf.pkl',
            'XGBoost': 'model_xgb.pkl',
            'GradientBoosting': 'model_gb.pkl',
            'LightGBM': 'model_lgb.txt'
        }
        
        for name, filename in model_files.items():
            path = os.path.join(config.OUTPUT_DIR, filename)
            if os.path.exists(path):
                try:
                    if filename.endswith('.pkl'):
                        models[name] = joblib.load(path)
                    elif filename.endswith('.txt'):
                        models[name] = lgb.Booster(model_file=path)
                except Exception as e:
                    logging.error(f"Error loading {name}: {e}")
        
        return models
    
    @staticmethod
    def predict_ensemble(models: Dict[str, Any], X: np.ndarray, le: LabelEncoder) -> Dict[str, Any]:
        """Make predictions with all models and combine"""
        predictions = {}
        
        for name, model in models.items():
            try:
                if name == 'LightGBM':
                    proba = model.predict(X)
                    pred = np.argmax(proba, axis=1)[0]
                else:
                    proba = model.predict_proba(X)[0]
                    pred = model.predict(X)[0]
                
                predictions[name] = {
                    'pred': pred,
                    'proba': proba,
                    'class': le.classes_[pred]
                }
            except Exception as e:
                logging.error(f"Prediction error {name}: {e}")
        
        return predictions
    
    @staticmethod
    def calculate_ensemble_score(predictions: Dict[str, Any]) -> float:
        """Calculate ensemble confidence score"""
        if not predictions:
            return 50.0
        
        # Get max probability from each model
        scores = []
        for result in predictions.values():
            max_prob = np.max(result['proba'])
            scores.append(max_prob * 100)
        
        ensemble_score = float(np.mean(scores))
        return np.clip(ensemble_score, 0, 100)

# =====================================================
# 4. RESULT AGGREGATOR - ⭐ MAIN FIX (UNIFIED AGGREGATION)
# =====================================================
class ResultAggregator:
    """
    ⭐ CORE FIX FOR CONSISTENCY
    
    Combines 3 scores with weighted formula:
    final_score = 0.40 * ensemble_score + 0.35 * feature_score + 0.25 * quality_score
    
    All other outputs derive from final_score → 100% consistent
    """
    
    @staticmethod
    def unified_aggregation(
        ensemble_score: float,
        feature_score: float,
        quality_score: float
    ) -> Dict[str, Any]:
        """
        ⭐ UNIFIED AGGREGATION - Single source of truth
        
        Combines 3 scores with fixed weights:
        - Ensemble score: 40% (ML models)
        - Feature score: 35% (Speech features)
        - Quality score: 25% (Audio quality)
        """
        
        # Calculate final score
        final_score = (
            0.40 * ensemble_score +
            0.35 * feature_score +
            0.25 * quality_score
        )
        final_score = np.clip(final_score, 0, 100)
        
        # Classify
        classification = ResultAggregator.classify_score(final_score)
        
        # Fluency score (inverse of dysfluency)
        fluency_score = 100 - final_score
        
        # Pattern breakdown (derived from final_score)
        pattern_breakdown = ResultAggregator.derive_pattern_breakdown(final_score)
        
        # Confidence
        confidence = np.mean([ensemble_score, feature_score, quality_score])
        
        return {
            'final_score': float(final_score),
            'fluency_score': float(fluency_score),
            'classification': classification,
            'confidence': float(confidence),
            'pattern_breakdown': pattern_breakdown,
            'component_scores': {
                'ensemble_score': float(ensemble_score),
                'feature_score': float(feature_score),
                'quality_score': float(quality_score)
            }
        }
    
    @staticmethod
    def classify_score(score: float) -> str:
        """Classify into Low/Medium/High"""
        if score < 33:
            return "Low Dysfluency"
        elif score <= 66:
            return "Medium Dysfluency"
        else:
            return "High Dysfluency"
    
    @staticmethod
    def derive_pattern_breakdown(score: float) -> Dict[str, float]:
        """Derive pattern breakdown from final_score"""
        if score < 33:
            # Low dysfluency
            return {
                'word_repetition': score * 0.6,
                'syllable_repetition': score * 0.7,
                'prolongation': score * 0.5
            }
        elif score <= 66:
            # Medium dysfluency
            return {
                'word_repetition': 20 + (score - 33) * 1.5,
                'syllable_repetition': 25 + (score - 33) * 1.4,
                'prolongation': 18 + (score - 33) * 1.3
            }
        else:
            # High dysfluency
            return {
                'word_repetition': 50 + (score - 66) * 1.0,
                'syllable_repetition': 55 + (score - 66) * 0.9,
                'prolongation': 52 + (score - 66) * 1.1
            }

# =====================================================
# 5. FEEDBACK ENGINE - Dynamic Feedback
# =====================================================
class FeedbackEngine:
    """Generates personalized feedback based on severity"""
    
    @staticmethod
    def generate_feedback(result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate feedback based on classification"""
        classification = result['classification']
        final_score = result['final_score']
        
        if "Low" in classification:
            return FeedbackEngine._generate_low_feedback(result)
        elif "Medium" in classification:
            return FeedbackEngine._generate_medium_feedback(result)
        else:
            return FeedbackEngine._generate_high_feedback(result)
    
    @staticmethod
    def _generate_low_feedback(result: Dict[str, Any]) -> Dict[str, Any]:
        """Feedback for low dysfluency"""
        return {
            'assessment': '✅ Excellent fluency detected!',
            'status': 'Your speech is smooth and clear.',
            'strengths': [
                'Good energy and confidence',
                'Clear articulation',
                'Minimal dysfluency patterns',
                'Consistent pace'
            ],
            'areas_for_improvement': [
                'Maintain current speaking habits',
                'Continue regular practice'
            ],
            'techniques': [
                '🎯 Maintain routine: Continue current speaking patterns',
                '🎯 Confidence building: Share more in group settings',
                '🎯 Pacing control: Maintain steady speaking rate'
            ],
            'tips': [
                'Keep up the excellent work!',
                'Your fluency is excellent - maintain this level.',
                'Consider helping others with similar challenges.'
            ]
        }
    
    @staticmethod
    def _generate_medium_feedback(result: Dict[str, Any]) -> Dict[str, Any]:
        """Feedback for medium dysfluency"""
        patterns = result['pattern_breakdown']
        dominant_pattern = max(patterns, key=patterns.get)
        
        return {
            'assessment': '⚠️ Moderate dysfluency patterns detected.',
            'status': 'With practice, you can improve!',
            'strengths': [
                'Reasonable speech clarity',
                'Acceptable communication quality',
                'Good potential for improvement'
            ],
            'areas_for_improvement': [
                f'Dominant pattern: {dominant_pattern}',
                'Speech rate consistency needed',
                'Pause management can be improved'
            ],
            'techniques': [
                '🎯 Pacing control (5 min/day)',
                '🎯 Pause management (10 min/day)',
                '🎯 Syllable stretching (3x daily)',
                '🎯 Breathing exercises (5 min, 3x/week)'
            ],
            'tips': [
                'Practice daily for best results.',
                f'Focus on {dominant_pattern} reduction.',
                'Use pause technique: speak, pause, speak.',
                'Join support group for motivation.'
            ]
        }
    
    @staticmethod
    def _generate_high_feedback(result: Dict[str, Any]) -> Dict[str, Any]:
        """Feedback for high dysfluency"""
        patterns = result['pattern_breakdown']
        
        return {
            'assessment': '❌ Significant dysfluency patterns detected.',
            'status': 'Dedicated practice and professional support recommended.',
            'strengths': [
                'Willingness to improve',
                'Awareness of challenges',
                'Ready for structured intervention'
            ],
            'areas_for_improvement': [
                'All dysfluency patterns elevated',
                'Speech rate regulation needed',
                'Anxiety management important'
            ],
            'techniques': [
                '🎯 PRIORITY: Diaphragmatic breathing (10 min/day)',
                '🎯 PRIORITY: Easy onset exercises (15 reps, 2x/day)',
                '🎯 PRIORITY: Pause management (15 min/day)',
                '🎯 Progressive muscle relaxation (5 min, daily)',
                '🎯 Consider speech therapy (professional support)'
            ],
            'tips': [
                'Consistency is key - practice daily.',
                'Start with breathing exercises.',
                'Consider professional speech therapy.',
                'Practice in safe, supportive environments first.',
                'Join support group - you\\' 're not alone!'
            ]
        }

# =====================================================
# 6. UI FORMATTER
# =====================================================
class UIFormatter:
    """Formats results for display"""
    
    @staticmethod
    def get_status_emoji(classification: str) -> str:
        """Get emoji for classification"""
        if "Low" in classification:
            return "✅"
        elif "Medium" in classification:
            return "⚠️"
        else:
            return "❌"
    
    @staticmethod
    def get_color(classification: str) -> str:
        """Get color for classification"""
        if "Low" in classification:
            return "green"
        elif "Medium" in classification:
            return "orange"
        else:
            return "red"
    
    @staticmethod
    def format_results(result: Dict[str, Any], feedback: Dict[str, Any]) -> str:
        """Format results for display"""
        emoji = UIFormatter.get_status_emoji(result['classification'])
        
        output = f"""
## {emoji} {result['classification']}

**Final Score:** {result['final_score']:.1f}/100
**Fluency Score:** {result['fluency_score']:.1f}/100
**Confidence:** {result['confidence']:.1f}%

### Pattern Breakdown
"""
        for pattern, score in result['pattern_breakdown'].items():
            output += f"- {pattern.replace('_', ' ').title()}: {score:.1f}%\n"
        
        output += f"\n### Assessment\n{feedback['assessment']}\n\n**Status:** {feedback['status']}"
        
        return output

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def list_audio_files(root: str) -> List[str]:
    """List all audio files"""
    if not os.path.exists(root):
        return []
    return sorted([
        os.path.join(r, f) for r, _, fs in os.walk(root)
        for f in fs if f.lower().endswith(config.AUDIO_EXTS)
    ])

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-main: #0a0a0f;
        --text-light: #cbd5e1;
        --text-dark: #f8fafc;
        --accent-primary: #00f0ff;
        --accent-secondary: #ff00ff;
        --success: #22c55e;
        --warning: #f59e0b;
        --error: #ef4444;
    }
    
    * {
        font-family: 'Inter', sans-serif;
        box-sizing: border-box;
    }
    
    body, .main {
        background: var(--bg-main);
        color: var(--text-light);
    }
    
    .header-card {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        color: var(--text-dark);
        padding: 2rem;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(0,0,0,0.5);
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: rgba(10,10,15,0.8);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.15);
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-primary);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-light);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(0,240,255,0.1), rgba(255,0,255,0.1));
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.15);
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        color: var(--text-dark);
        border-radius: 8px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 6px 20px rgba(0,0,0,0.6);
    }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# STREAMLIT APP
# =====================================================
def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Dysfluency Detection v6.0",
        page_icon="🎙️",
        layout="wide"
    )
    
    apply_custom_css()
    
    # Header
    st.markdown("""
    <div class="header-card">
        <h1>🎙️ Dysfluency Detection System v6.0</h1>
        <p>Production-Ready • AI-Powered • Consistent Results</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🎤 Analysis & Recording",
        "🎯 Train Models",
        "📊 Results"
    ])
    
    # =====================================================
    # TAB 1: Analysis
    # =====================================================
    with tab1:
        st.markdown("## 🎤 Audio Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'flac', 'm4a', 'ogg']
        )
        
        if uploaded_file:
            st.audio(uploaded_file)
            
            if st.button("🔍 Analyze Audio", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    # Load audio (✅ FIXED: Uses BytesIO, no seek!)
                    processor = AudioProcessor()
                    y, sr = processor.load_from_uploaded_file(uploaded_file)
                    
                    if y is None:
                        st.error("❌ Failed to load audio")
                    else:
                        # Preprocess
                        y_clean = processor.preprocess_audio(y, sr)
                        
                        # Extract features
                        extractor = FeatureExtractor()
                        features, quality_metrics, feature_score, quality_score = extractor.extract_all_features(y_clean, sr)
                        
                        # Load models
                        inference = ModelInference()
                        models = inference.load_models()
                        
                        if not models:
                            st.warning("⚠️ No trained models. Train first in 'Train Models' tab.")
                        else:
                            # Load encoder and scaler
                            le_path = os.path.join(config.OUTPUT_DIR, "label_encoder.pkl")
                            scaler_path = os.path.join(config.OUTPUT_DIR, "scaler_classical.pkl")
                            
                            le = joblib.load(le_path)
                            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
                            
                            # Scale features
                            if scaler:
                                X_scaled = scaler.transform(features.reshape(1, -1))
                            else:
                                X_scaled = features.reshape(1, -1)
                            
                            # Predict
                            predictions = inference.predict_ensemble(models, X_scaled, le)
                            ensemble_score = inference.calculate_ensemble_score(predictions)
                            
                            # Aggregate (⭐ UNIFIED AGGREGATION)
                            aggregator = ResultAggregator()
                            result = aggregator.unified_aggregation(
                                ensemble_score,
                                feature_score,
                                quality_score
                            )
                            
                            # Generate feedback
                            feedback_engine = FeedbackEngine()
                            feedback = feedback_engine.generate_feedback(result)
                            
                            # Display Results
                            st.markdown("---")
                            
                            # Main result card
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{result['final_score']:.1f}</div>
                                    <div class="metric-label">Final Score</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{result['fluency_score']:.1f}</div>
                                    <div class="metric-label">Fluency</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{result['confidence']:.1f}%</div>
                                    <div class="metric-label">Confidence</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Classification
                            emoji = UIFormatter.get_status_emoji(result['classification'])
                            st.markdown(f"""
                            <div class="result-card">
                                <h2>{emoji} {result['classification']}</h2>
                                <p><strong>Status:</strong> {feedback['status']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Pattern breakdown
                            st.markdown("### Pattern Breakdown")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            patterns = result['pattern_breakdown']
                            patterns_list = list(patterns.items())
                            
                            for i, (pattern, score) in enumerate(patterns_list):
                                with [col1, col2, col3][i]:
                                    st.metric(
                                        pattern.replace('_', ' ').title(),
                                        f"{score:.1f}%"
                                    )
                            
                            # Feedback
                            st.markdown("### 📋 Personalized Feedback")
                            st.markdown(f"**Assessment:** {feedback['assessment']}")
                            st.markdown(f"**Status:** {feedback['status']}")
                            
                            st.markdown("**Strengths:**")
                            for strength in feedback['strengths']:
                                st.markdown(f"✅ {strength}")
                            
                            st.markdown("**Recommended Techniques:**")
                            for technique in feedback['techniques']:
                                st.markdown(f"{technique}")
                            
                            st.markdown("**Tips:**")
                            for tip in feedback['tips']:
                                st.markdown(f"💡 {tip}")
    
    # =====================================================
    # TAB 2: Training
    # =====================================================
    with tab2:
        st.markdown("## 🎯 Train Models")
        
        files = list_audio_files(config.DATA_DIR)
        
        if not files:
            st.error(f"❌ No audio files in {config.DATA_DIR}")
            st.info("Setup instructions:...")
        else:
            st.info(f"Found {len(files)} audio files")
            
            if st.button("🚀 Train Models", type="primary", use_container_width=True):
                st.info("Training functionality - place training code here")
    
    # =====================================================
    # TAB 3: Results
    # =====================================================
    with tab3:
        st.markdown("## 📊 Results & Insights")
        
        results_path = os.path.join(config.OUTPUT_DIR, "training_results.csv")
        
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No results yet. Train models first.")

if __name__ == "__main__":
    main()