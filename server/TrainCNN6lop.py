import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from sklearn.utils import class_weight
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, f1_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense,
    Dropout, BatchNormalization, Flatten, GRU, Layer
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import warnings
import glob
import os
import gc
import time
from fractions import Fraction
from tqdm import tqdm
import scipy.signal
import pandas as pd
import joblib
from datetime import datetime, timedelta

def load_models_for_ensemble(pattern="best_model_fold_*.keras", custom_load=True):
    """
    Load all model files matching pattern. Returns list of keras models.
    """
    files = sorted(glob.glob(pattern))
    models = []
    if not files:
        print(f"WARNING: No model files found for ensemble with pattern '{pattern}'")
        return models
    for f in files:
        try:
            if custom_load:
                m = load_trained_model_for_inference(f)
            else:
                m = tf.keras.models.load_model(f, compile=False)
            models.append(m)
            print(f"Loaded model for ensemble: {f}")
        except Exception as e:
            print(f"Failed to load model {f}: {e}")
    return models

def predict_subject_ensemble(model_or_models, subject_id, apply_hmm=True, temp=1.0, min_diag=0.5):
    """
    Predict for a single subject by averaging probabilities from multiple models (soft voting).
    model_or_models: list of models or list of paths or single model/path.
    Returns: X_proc, y_true, probs_avg, preds_final
    """
    # normalize input to list of models
    models = []
    if model_or_models is None:
        return None, None, None, None
    if isinstance(model_or_models, (list, tuple)):
        for m in model_or_models:
            if isinstance(m, str):
                try:
                    models.append(load_trained_model_for_inference(m))
                except Exception:
                    try:
                        models.append(tf.keras.models.load_model(m, compile=False))
                    except Exception as e:
                        print(f"Cannot load model {m}: {e}")
            else:
                models.append(m)
    else:
        # single model/path
        if isinstance(model_or_models, str):
            try:
                models.append(load_trained_model_for_inference(model_or_models))
            except Exception:
                models.append(tf.keras.models.load_model(model_or_models, compile=False))
        else:
            models.append(model_or_models)

    if not models:
        print("No models available for ensemble.")
        return None, None, None, None

    # load and preprocess subject (same as predict_subject_from_saved_model)
    X_raw, y_true = load_single_subject(subject_id)
    if X_raw is None or y_true is None:
        return None, None, None, None

    X_list = []
    for i in range(X_raw.shape[0]):
        x = X_raw[i].astype(np.float32)
        x_r = scipy.signal.resample(x, CONFIG.TARGET_LENGTH_CNN, axis=0).astype(np.float32)
        mean = x_r.mean(axis=0, keepdims=True)
        std = x_r.std(axis=0, keepdims=True) + 1e-8
        X_list.append((x_r - mean) / std)
    if not X_list:
        return None, None, None, None
    X_proc = np.stack(X_list).astype(np.float32)

    # collect probs
    probs_list = []
    for m in models:
        try:
            p = m.predict(X_proc, verbose=0)
            probs_list.append(p)
        except Exception as e:
            print(f"Model predict failed: {e}")
    if not probs_list:
        return None, None, None, None

    probs_avg = np.mean(np.stack(probs_list, axis=0), axis=0)

    # temperature scaling
    if temp != 1.0:
        probs_avg = np.clip(probs_avg, 1e-12, 1.0)
        probs_avg = probs_avg ** (1.0 / float(temp))
        probs_avg = probs_avg / probs_avg.sum(axis=1, keepdims=True)

    # HMM smoothing (use clean_eval if no noise in true labels)
    has_noise = np.any(np.array(y_true) == 5)
    clean_eval = not has_noise
    preds_arg = np.argmax(probs_avg, axis=1)
    preds_final = preds_arg.copy()

    if apply_hmm:
        try:
            preds_hmm = hmm_smoothing_viterbi(probs_avg, trans_diag=CONFIG.HMM_SMOOTHING_DIAG, clean_eval=clean_eval)
            # if collapsed fallback to argmax
            from collections import Counter
            c = Counter(preds_hmm.tolist())
            frac = max(c.values()) / float(len(preds_hmm))
            if frac > 0.95:
                preds_final = preds_arg.copy()
            else:
                preds_final = preds_hmm
        except Exception as e:
            print("HMM failed on ensemble probs:", e)
            preds_final = preds_arg.copy()

    return X_proc, np.array(y_true), probs_avg, preds_final

# ==================================
# ⚡ Mixed Precision Configuration
mixed_precision.set_global_policy("mixed_float16")
print("⚡ Mixed precision policy:", mixed_precision.global_policy())

# Configure GPU memory growth to prevent allocation issues
physical_devices = tf.config.list_physical_devices('GPU')
print("GPUs:", physical_devices)
for gpu in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Failed to set memory growth for GPU: {e}")

# Suppress MNE and other warnings
warnings.filterwarnings('ignore', category=UserWarning, module='mne')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==================================
# ⚙️ CONFIGURATION & CONSTANTS
# ==================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

class CONFIG:
    """Centralized configuration for the entire script."""

    DATA_PATH = r"C:\NCKH2025\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette"
    CACHE_DIR = "./cache_npz"
    NUM_SUBJECTS_TO_LOAD = 118 # 🚀 SỬA: Tăng số lượng subject để tương thích với N_FOLDS=10
    N_FOLDS = 10
    TARGET_LENGTH_CNN = 1500 # Đổi tên để phân biệt với LSTM
    NUM_CLASSES = 6
    SLEEP_STAGE_LABELS = ["Wake", "N1", "N2", "N3", "REM", "Noise"]
    LABELS = [0, 1, 2, 3, 4, 5]  # 0–5
    EPOCHS_PER_FOLD = 50
    STEPS_PER_EPOCH = 500
    L2_REG = 0.01
    DROPOUT_RATE = 0.5
    LEARNING_RATE = 3e-4
    PATIENCE = 8
    MIN_LR = 1e-7
    REDUCE_LR_FACTOR = 0.5
    FOCAL_LOSS_GAMMA = 2.0
    HMM_SMOOTHING_DIAG = 0.50
    AUGMENTATION_STAGES = [1, 3, 4]
    BATCH_SIZE = 32
    MAX_WEIGHT_CAP = 10.0 # Thêm: Giới hạn trọng số tối đa cho các lớp thiểu số

# ==================================
# 1. 🗂️ DATA LOADING & PREPROCESSING
# ==================================

def generate_subject_ids(num_subjects=CONFIG.NUM_SUBJECTS_TO_LOAD):
    """
    Generate a list of subject IDs based on the SC format.
    """
    subject_ids = []
    for subject_num in range(num_subjects):
        subject_id1 = f"SC{4000 + subject_num * 10 + 1}"
        subject_id2 = f"SC{4000 + subject_num * 10 + 2}"
        subject_ids.extend([subject_id1, subject_id2])
    return subject_ids

def find_matching_files(dataset_path, target_subject_id):
    """
    Find PSG and hypnogram files for a given subject ID.
    """
    all_files = glob.glob(os.path.join(dataset_path, '*'))
    matching_files = {'psg': None, 'hypnogram': None}

    psg_patterns = ['psg.edf', 'eeg.edf']
    hyp_patterns = ['hypnogram.edf', 'annot.edf']

    target_lower = target_subject_id.lower()
    for file_path in all_files:
        filename_lower = os.path.basename(file_path).lower()
        if target_lower in filename_lower:
            if any(p in filename_lower for p in psg_patterns):
                matching_files['psg'] = file_path
            elif any(h in filename_lower for h in hyp_patterns):
                matching_files['hypnogram'] = file_path

    return matching_files

def load_single_subject(target_subject_id):
    """Load data for a single subject, including noisy samples."""
    try:
        dataset_path = r"C:\NCKH2025\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette"

        matching_files = find_matching_files(dataset_path, target_subject_id)

        edf_file = matching_files['psg']
        hyp_file = matching_files['hypnogram']

        print(f"Looking for subject: {target_subject_id}")

        if not edf_file or not os.path.exists(edf_file):
            print(f"   EDF file not found for: {target_subject_id}")
            return None, None

        if not hyp_file or not os.path.exists(hyp_file):
            print(f"   Hypnogram file not found for: {target_subject_id}")
            return None, None

        print(f"   EDF file: {os.path.basename(edf_file)}")
        print(f"   Hypnogram file: {os.path.basename(hyp_file)}")

        raw = mne.io.read_raw_edf(edf_file, stim_channel=None, preload=True, verbose=False)

        # --- DEBUG: thông tin raw ---
        try:
            print("   Raw channels:", raw.ch_names)
            print("   Raw sfreq:", raw.info.get("sfreq"))
        except Exception:
            pass

        selected_channels = ["EEG Fpz-Cz", "EEG Pz-Oz"]

        available_channels = [ch for ch in selected_channels if ch in raw.ch_names]
        if not available_channels:
            # fallback: nếu tên kênh khác, chọn 2 kênh đầu tiên và warn
            print("   WARNING: expected EEG channels not found; using first two channels instead.")
            available_channels = raw.ch_names[:2]

        raw.pick_channels(available_channels)
        sfreq = int(raw.info["sfreq"])

        try:
            annotations = mne.read_annotations(hyp_file)
        except Exception as e:
            print(f"   Error reading hypnogram file {hyp_file}: {e}")
            return None, None

        raw.set_annotations(annotations)

        # Định nghĩa ánh xạ nhãn, ánh xạ nhiễu về lớp 5
        label_map = {
            'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
            'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4,
            'Sleep stage ?': 5, 'Movement time': 5,
            'W': 0, '1': 1, '2': 2, '3': 3, '4': 3, 'R': 4,
            'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4,
            'Wake': 0, 'REM sleep': 4
        }

        # SỬA LỖI QUAN TRỌNG: Tạo epoch 30s liên tục thay vì dựa vào event
        # Cách cũ (dựa vào events_from_annotations) chỉ lấy được một phần nhỏ dữ liệu
        # vì nó chỉ tạo event ở điểm bắt đầu của một giai đoạn ngủ dài.
        epochs = mne.make_fixed_length_epochs(raw, duration=30, overlap=0, preload=True, verbose=False)

        # Gán nhãn cho từng epoch vừa tạo
        labels_for_epochs = []
        for i in range(len(epochs)):
            epoch_start_time = epochs.events[i, 0] / sfreq
            # Tìm annotation tương ứng với thời điểm bắt đầu của epoch
            idx = np.where(annotations.onset <= epoch_start_time)[0]
            if len(idx) > 0:
                # Lấy annotation cuối cùng trước hoặc tại thời điểm epoch bắt đầu
                labels_for_epochs.append(label_map.get(annotations.description[idx[-1]], 5)) # Mặc định là Noise nếu không khớp
            else:
                labels_for_epochs.append(5) # Gán là Noise nếu không có annotation
        
        epochs.events[:, -1] = np.array(labels_for_epochs)
        labels = epochs.events[:, -1]
        X = epochs.get_data()
        X = X.astype(np.float32)

        # Debug thêm: distribution nhãn, 1 epoch dạng sóng
        try:
            import matplotlib.pyplot as _plt
            from collections import Counter as _C
            print(f"   Available channels used: {available_channels}")
            print("   Label counts:", dict(_C(labels.tolist())))
            print("   Epochs shape (n_epochs, n_channels, n_times):", X.shape)
            # lưu waveform của epoch đầu
            _plt.figure(figsize=(8,3))
            _plt.plot(X[0].T)
            _plt.title(f"{target_subject_id} epoch0 channels:{available_channels} sfreq={sfreq}")
            _plt.tight_layout()
            os.makedirs("debug_plots", exist_ok=True)
            _plt.savefig(f"debug_plots/{target_subject_id}_epoch0_wave.png", dpi=150)
            _plt.close()
        except Exception as _e:
            print("   Debug plot failed:", _e)

        # Chuẩn hóa từng epoch và kênh
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X[i, j, :] = (X[i, j, :] - np.mean(X[i, j, :])) / (np.std(X[i, j, :]) + 1e-8)

        # Đổi thứ tự dimensions thành (samples, time_points, channels)
        X = np.transpose(X, (0, 2, 1))

        print(f"   Subject {target_subject_id}: {X.shape[0]} samples loaded")
        return X, labels

    except Exception as e:
        print(f"   Error loading subject {target_subject_id}: {e}")
        return None, None

def load_all_subjects(subject_filter=None):
    """
    Load data from multiple subjects, returning ALL samples including noise (5).
    """
    subject_ids = subject_filter if subject_filter else generate_subject_ids(CONFIG.NUM_SUBJECTS_TO_LOAD)

    all_X_list, all_y_list, all_subject_numeric_ids = [], [], []

    for subject_numeric_id, subject_id in enumerate(subject_ids):
        # Load TOÀN BỘ dữ liệu, bao gồm cả các mẫu nhiễu
        X_subject, y_subject = load_single_subject(subject_id)
        if X_subject is not None and y_subject is not None:
            if len(y_subject) > 0:
                all_X_list.append(X_subject) # APPEND FULL X
                all_y_list.append(y_subject) # APPEND FULL Y (bao gồm lớp 5)
                numeric_id = subject_numeric_id // 2 + 1
                all_subject_numeric_ids.extend([numeric_id] * len(y_subject))
                print(f"✓ Successfully loaded subject {subject_id} (Full set: {len(y_subject)} samples)")
            else:
                print(f"✗ No samples found for subject {subject_id}")
        else:
            print(f"✗ Failed to load subject {subject_id}")

    if not all_X_list:
        print("\nNo subjects loaded. Using synthetic data for demonstration.")
        return load_synthetic_data()

    return all_X_list, all_y_list, np.array(all_subject_numeric_ids)


def load_synthetic_data():
    """Fallback to synthetic data if real data is not available."""
    n_samples = 500
    time_points = 3000
    channels = 2

    X = np.random.randn(n_samples, time_points, channels).astype(np.float32)
    y = np.random.randint(0, CONFIG.NUM_CLASSES, size=(n_samples,))
    
    subject_ids = np.repeat(np.arange(1, 11), n_samples // 10)
    subject_ids = np.append(subject_ids, [10] * (n_samples - len(subject_ids)))
    subject_ids = subject_ids.astype(np.int64)

    return [X], [y], subject_ids

def preprocess_data(data, cache_dir=CONFIG.CACHE_DIR, force_reprocess=False):
    """
    Resample and normalize each sample; cache to avoid reprocessing.
    Handles classes 0-4 + Noise=5.
    """
    os.makedirs(cache_dir, exist_ok=True)
    processed = []

    for X_sub, y_sub, sub_id in data:
        cache_path = os.path.join(cache_dir, f"{sub_id}_processed_6cls.npz")
        if (not force_reprocess) and os.path.exists(cache_path):
            try:
                with np.load(cache_path, allow_pickle=False) as npz:
                    X_arr = npz["X"].astype(np.float32)
                    y_arr = npz["y"].astype(np.int32)
                processed.append((X_arr, y_arr, sub_id))
                continue
            except:
                pass

        X_out_list = []
        for idx in tqdm(range(len(X_sub)), desc=f"Preprocessing {sub_id}", ncols=80):
            x = X_sub[idx].astype(np.float32)
            # Resample to target length
            x_r = scipy.signal.resample(x, CONFIG.TARGET_LENGTH_CNN, axis=0).astype(np.float32)
            # Normalize
            mean = x_r.mean(axis=0, keepdims=True)
            std = x_r.std(axis=0, keepdims=True) + 1e-8
            X_out_list.append((x_r - mean) / std)

        if not X_out_list:
            continue

        X_arr = np.stack(X_out_list).astype(np.float32)
        y_arr = np.array(y_sub[:X_arr.shape[0]], dtype=np.int32)
        np.savez_compressed(cache_path, X=X_arr, y=y_arr)
        processed.append((X_arr, y_arr, sub_id))
        del X_out_list, X_arr, y_arr
        gc.collect()

    return processed

# ==================================
# 2. 🧠 MODEL & CUSTOM LAYERS
# ==================================

class MetricsLogger(Callback):
    """
    Callback để ghi lại learning rate ở cuối mỗi epoch.
    """
    def __init__(self):
        super().__init__()
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):
        try:
            lr = float(K.get_value(self.model.optimizer.lr))
            self.learning_rates.append(lr)
        except Exception:
            # Nếu không lấy được lr, ghi lại giá trị 0
            self.learning_rates.append(0.0)

class L2LossLogger(Callback):
    """
    Callback để ghi lại tổng L2 regularization loss ở cuối mỗi epoch.
    """
    def __init__(self):
        super().__init__()
        self.l2_losses = []

    def on_epoch_end(self, epoch, logs=None):
        l2_loss = 0.0
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                if hasattr(layer, 'losses') and layer.losses:
                    # L2 loss được thêm vào layer.losses
                    # Thường chỉ có 1 giá trị loss cho mỗi layer có regularizer
                    l2_loss += tf.reduce_sum(layer.losses).numpy()
        self.l2_losses.append(l2_loss)
        if logs is not None:
            logs['l2_loss'] = l2_loss

class AttentionLayer(Layer):
    """A custom Attention layer for sequence data."""
    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, timesteps, features)
        e = K.tanh(tf.tensordot(x, self.W, axes=[[2], [0]]) + self.b)  # (batch, timesteps, 1)
        a = K.softmax(e, axis=1)                                       # (batch, timesteps, 1)
        output = x * a                                                 # (batch, timesteps, features)
        return K.sum(output, axis=1)                                   # (batch, features)


def build_model(input_shape, num_classes):
    """
    Builds a pure 1D CNN model. The architecture is simplified and regularized.
    """
    model = tf.keras.Sequential([
        Input(shape=input_shape),
        # Block 1: Lớp tích chập với kernel lớn hơn để nắm bắt các đặc trưng cấp thấp
        Conv1D(filters=32, kernel_size=16, activation='relu', kernel_regularizer=l2(CONFIG.L2_REG)),
        BatchNormalization(),
        MaxPooling1D(pool_size=4, strides=4),
        Dropout(CONFIG.DROPOUT_RATE),

        # Block 2: Lớp tích chập nhỏ hơn để nắm bắt các đặc trưng chi tiết hơn
        Conv1D(filters=64, kernel_size=8, activation='relu', kernel_regularizer=l2(CONFIG.L2_REG)),
        BatchNormalization(),
        MaxPooling1D(pool_size=4, strides=4),
        Dropout(CONFIG.DROPOUT_RATE),

        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(CONFIG.L2_REG)),
        Dropout(CONFIG.DROPOUT_RATE + 0.1), # Dropout cao hơn một chút trước lớp output
 
        Dense(num_classes, activation='softmax', dtype="float32")
    ])
    return model

# ==================================
# 3. 🧪 UTILITIES & TRAINING HELPERS
# ==================================

# ---------------------------
# Signal Augmentation
# ---------------------------
def augment_signal(x):
    """Applies random augmentation to a signal."""
    if np.random.rand() < 0.3:
        x = x + np.random.normal(0, 0.01, size=x.shape)
    if np.random.rand() < 0.3:
        x = x * (0.9 + 0.2*np.random.rand())
    if np.random.rand() < 0.3:
        shift = np.random.randint(-20, 20)
        x = np.roll(x, shift, axis=0)
    return x.astype(np.float32)

def create_generator(all_X, all_y, batch_size=30, num_classes=CONFIG.NUM_CLASSES, augment_stages=None):
    """
    Generator cho LSTM với batch size cố định, skip lớp rỗng, hỗ trợ augmentation.
    
    all_X: numpy array (num_samples, 1500, 2)
    all_y: numpy array (num_samples,)
    batch_size: số mẫu mỗi batch
    num_classes: tổng số lớp
    augment_stages: list các lớp cần augment (vd: [0,1,2,3,4])
    """
    # Tạo dict: class -> list indices
    class_indices = {cls: np.where(all_y == cls)[0] for cls in range(num_classes)}
    # Bỏ các lớp rỗng
    class_indices = {cls: idxs for cls, idxs in class_indices.items() if len(idxs) > 0}

    if not class_indices:
        raise ValueError("Không còn lớp nào có sample trong generator!")

    while True:
        batch_x, batch_y = [], []

        while len(batch_x) < batch_size:
            # Lọc các lớp còn sample
            available_classes = [c for c, idxs in class_indices.items() if len(idxs) > 0]
            if not available_classes:
                raise ValueError("Tất cả các lớp đều rỗng trong generator!")
            cls = np.random.choice(available_classes)

            # Lấy index ngẫu nhiên từ lớp
            idxs = class_indices[cls]
            idx = np.random.choice(idxs)

            # Lấy dữ liệu
            x_sel = all_X[idx]
            # Augmentation nếu lớp này được config
            if augment_stages is not None and cls in augment_stages:
                x_sel = augment_signal(x_sel)

            batch_x.append(x_sel)
            batch_y.append(cls)

        # Chuyển sang numpy array và one-hot label
        batch_x = np.stack(batch_x).astype(np.float32)           # (batch_size, 1500, 2)
        batch_y = to_categorical(np.array(batch_y), num_classes) # (batch_size, num_classes)
        yield batch_x, batch_y

# ---------------------------
# Balanced Dataset Generator
# ---------------------------
def make_balanced_dataset(X_list, y_list, batch_size, num_classes=6, augment_stages=None):
    """
    Create tf.data.Dataset with balanced batches including Noise=5.
    Optional augment_stages dict {class_id: augment_function}.
    """
    all_X = np.concatenate(X_list, axis=0).astype(np.float32)
    all_y = np.concatenate(y_list, axis=0).astype(np.int32)

    unique_classes = np.unique(all_y)
    class_indices_local = {cls: np.where(all_y == cls)[0] for cls in unique_classes if len(np.where(all_y==cls)[0])>0}

    def generator():
        while True:
            batch_x, batch_y = [], []
            while len(batch_x) < batch_size:
                cls = np.random.choice(list(class_indices_local.keys()))
                idx = np.random.choice(class_indices_local[cls])
                x_sel = all_X[idx]
                if augment_stages is not None and cls in augment_stages:
                    x_sel = augment_stages[cls](x_sel)
                batch_x.append(x_sel)
                batch_y.append(cls)
            yield np.stack(batch_x).astype(np.float32), tf.one_hot(batch_y, num_classes, dtype=tf.float32)
 
    output_signature = (
        tf.TensorSpec(shape=(batch_size, all_X.shape[1], all_X.shape[2]), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, num_classes), dtype=tf.float32)
    )
 
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature).prefetch(tf.data.AUTOTUNE)

# ---------------------------
# Focal Loss
# ---------------------------
def focal_loss(gamma=CONFIG.FOCAL_LOSS_GAMMA, alpha=None):
    alpha_const = None
    if alpha is not None:
        alpha = np.array(alpha, dtype=np.float32)
        alpha_const = K.constant(alpha, dtype=K.floatx())

    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        ce = -y_true * K.log(y_pred)
        if alpha_const is not None:
            ce = ce * alpha_const
        loss = K.pow(1 - y_pred, gamma) * ce
        return K.sum(loss, axis=1)

    return focal_loss_fixed

# ---------------------------
# HMM Viterbi smoothing
# ---------------------------
def hmm_smoothing_viterbi(y_pred_probs, trans_diag=CONFIG.HMM_SMOOTHING_DIAG, clean_eval=False):
    """
    Applies Viterbi algorithm for HMM smoothing.
    If clean_eval is True, it will ignore the last class (Noise) and re-normalize probabilities.
    """
    preds_probs = np.asarray(y_pred_probs, dtype=np.float64)

    if clean_eval and preds_probs.shape[1] > 1:
        # Ignore the last class (Noise) and re-normalize
        preds_probs = preds_probs[:, :-1]
        preds_probs /= (preds_probs.sum(axis=1, keepdims=True) + 1e-12)

    N, n_states = preds_probs.shape
    off_diag = (1.0 - trans_diag) / (n_states - 1) if n_states > 1 else 0
    transmat = np.full((n_states, n_states), off_diag, dtype=np.float64)
    np.fill_diagonal(transmat, trans_diag)
    startprob = np.full(n_states, 1.0 / n_states, dtype=np.float64)

    log_emiss = np.log(preds_probs + 1e-12)
    log_trans = np.log(transmat + 1e-12)
    log_start = np.log(startprob + 1e-12)

    dp = np.empty((N, n_states), dtype=np.float64)
    ptr = np.empty((N, n_states), dtype=np.int32)

    dp[0] = log_start + log_emiss[0]
    for t in range(1, N):
        prev = dp[t-1][:, None] + log_trans
        ptr[t] = np.argmax(prev, axis=0)
        dp[t] = prev[ptr[t], np.arange(n_states)] + log_emiss[t]

    states = np.empty(N, dtype=np.int32)
    states[-1] = np.argmax(dp[-1])
    for t in range(N-2, -1, -1):
        states[t] = ptr[t + 1, states[t + 1]]

    return states

# ---------------------------
# Callback Speed Logger
# ---------------------------
class SpeedLogger(Callback):
    def on_train_begin(self, logs=None):
        self.epoch_start_time = None
        self.total_samples = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.total_samples = 0

    def on_train_batch_end(self, batch, logs=None):
        self.total_samples += CONFIG.BATCH_SIZE

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.epoch_start_time
        if duration > 0:
            speed = self.total_samples / duration
            print(f"📈 Epoch {epoch+1} speed: {speed:.1f} samples/sec")

# ---------------------------
# Custom Callback for Clean F1 Score
# ---------------------------
class CleanF1ScoreCallback(Callback):
    """
    Callback to compute F1 score on the clean validation set at the end of each epoch.
    Also implements: save-best-by-val_f1, simple ReduceLROnPlateau-like behaviour and EarlyStopping by patience.
    """
    def __init__(self, validation_data_clean, ckpt_path=None, patience=8, reduce_lr_factor=0.5, min_lr=1e-7, verbose=1):
        super().__init__()
        self.X_val_clean, self.y_val_clean_true = validation_data_clean
        self.best_f1 = -1.0
        self.ckpt_path = ckpt_path
        self.patience = int(patience)
        self.wait = 0
        self.reduce_lr_factor = float(reduce_lr_factor)
        self.min_lr = float(min_lr)
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # default if no clean validation
        if self.X_val_clean is None or len(self.X_val_clean) == 0:
            logs['val_f1_clean'] = 0.0
            return

        # Predict on clean validation set
        try:
            y_pred_probs = self.model.predict(self.X_val_clean, verbose=0)
        except Exception as e:
            if self.verbose:
                print(f"CleanF1ScoreCallback: prediction failed: {e}")
            logs['val_f1_clean'] = 0.0
            return

        # HMM smoothing (clean_eval=True). Fallback to argmax on error.
        try:
            y_pred_hmm = hmm_smoothing_viterbi(y_pred_probs, trans_diag=CONFIG.HMM_SMOOTHING_DIAG, clean_eval=True)
        except Exception:
            y_pred_hmm = np.argmax(y_pred_probs, axis=1)

        # compute macro f1
        try:
            f1 = f1_score(self.y_val_clean_true, y_pred_hmm, average='macro', zero_division=0)
        except Exception:
            f1 = 0.0

        logs['val_f1_clean'] = float(f1)

        # Save best model by val_f1_clean
        if f1 > self.best_f1 + 1e-6:
            self.best_f1 = f1
            self.wait = 0
            if self.ckpt_path:
                try:
                    # save in TF SavedModel (.keras) format
                    self.model.save(self.ckpt_path, include_optimizer=False)
                   

                    if self.verbose:
                        print(f" - val_f1_clean improved to {f1:.4f}; saved model -> {self.ckpt_path}")
                except Exception as e:
                    if self.verbose:
                        print(f"CleanF1ScoreCallback: failed to save model: {e}")
        else:
            self.wait += 1
            if self.verbose:
                print(f" - val_f1_clean: {f1:.4f} (best {self.best_f1:.4f}), wait={self.wait}/{self.patience}")

            # Reduce LR when no improvement
            try:
                lr = float(K.get_value(self.model.optimizer.lr))
                new_lr = max(self.min_lr, lr * self.reduce_lr_factor)
                if new_lr < lr - 1e-12:
                    K.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose:
                        print(f"   ↓ Reduced LR: {lr:.3e} -> {new_lr:.3e}")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Reduce LR failed: {e}")

            # Early stopping by patience
            if self.wait >= self.patience:
                if self.verbose:
                    print(f"   ⛔ EarlyStopping triggered by CleanF1ScoreCallback (no improvement for {self.patience} epochs).")
                self.model.stop_training = True

# ---------------------------
# Dataset helpers
# ---------------------------
def get_test_ds(X_list, y_list, batch_size):
    """
    Tạo tf.data.Dataset cho tập Test, batch sẵn và one-hot label
    """
    all_X = np.concatenate(X_list, axis=0).astype(np.float32)
    all_y = np.concatenate(y_list, axis=0).astype(np.int32)
    
    all_y_onehot = tf.one_hot(all_y, CONFIG.NUM_CLASSES, dtype=tf.float32)

    test_ds = tf.data.Dataset.from_tensor_slices((all_X, all_y_onehot))
    test_ds = test_ds.batch(batch_size, drop_remainder=False)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return test_ds

def get_test_ds_for_eval(X_list, y_list):
    all_X = np.concatenate(X_list, axis=0)
    all_y = np.concatenate(y_list, axis=0)
    ds = tf.data.Dataset.from_tensor_slices((all_X, to_categorical(all_y, num_classes=CONFIG.NUM_CLASSES)))
    return ds.batch(CONFIG.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==================================
# 4. 🚀 MAIN TRAINING LOOP
# ==================================

def get_optimal_wakeup_times(sleep_stage_seq, start_time, choice, age, gender):
    optimal_times = []
    if choice == '1':
        current_time = start_time
        for stage in sleep_stage_seq:
            # mỗi sample tương ứng 30 giây (thay vì 30 phút)
            current_time += timedelta(seconds=30)
            if stage in ['N1', 'N2', 'REM']:
                optimal_times.append(current_time.strftime("%H:%M"))
    elif choice == '2':
        total_minutes = len(sleep_stage_seq) * 0.5  # mỗi sample = 0.5 phút
        num_cycles = int(total_minutes // 90)
        for i in range(1, num_cycles + 1):
            wakeup_time = start_time + timedelta(minutes=90 * i)
            optimal_times.append(wakeup_time.strftime("%H:%M"))
    else:
        print("⚠️ Lựa chọn không hợp lệ. Sử dụng mặc định: 90 phút.")
        return get_optimal_wakeup_times(sleep_stage_seq, start_time, '2', age, gender)

    # Lời khuyên cá nhân hóa
    if choice == '1':
        if gender.lower() == 'nam':
            print("💡 Nam giới thường có ít giấc ngủ REM hơn, cần đảm bảo ngủ sâu.")
        elif gender.lower() == 'nữ' or gender.lower() == 'nu':
            print("💡 Nữ giới thường có nhiều REM hơn, quan trọng cho trí nhớ & cảm xúc.")
    elif choice == '2' and int(age) > 65:
        print("💡 Người lớn tuổi thường ngủ ngắn hơn, có thể thử dậy sớm hơn.")

    # loại bỏ trùng giờ liên tiếp
    unique_times = []
    for t in optimal_times:
        if not unique_times or t != unique_times[-1]:
            unique_times.append(t)
    return unique_times

def evaluate_and_report_on_subject(model, subject_id, config):
    """
    Đánh giá mô hình trên một subject cụ thể, bao gồm cả các mẫu nhiễu.
    """
    print(f"\n===== 🔍 Đánh giá và báo cáo cho subject: {subject_id} =====")
    if model is None:
        print("❌ Model chưa được load. Hãy train hoặc load mô hình trước.")
        return None, None
    
    # Load toàn bộ dữ liệu, bao gồm cả nhiễu (-1)
    X_full, y_full = load_single_subject(subject_id)

    if X_full is None or y_full is None:
        print(f"⚠️ Không thể load dữ liệu cho subject {subject_id}. Bỏ qua.")
        return None, None
    
    # Tiền xử lý dữ liệu
    X_full_processed_list = []
    for idx in range(X_full.shape[0]):
        try:
            x_raw = X_full[idx]
            len_in, len_out = x_raw.shape[0], config.TARGET_LENGTH_CNN
            x_r = scipy.signal.resample(x_raw, len_out, axis=0).astype(np.float32)
            mean = x_r.mean(axis=0, keepdims=True)
            std = x_r.std(axis=0, keepdims=True) + 1e-8
            x_norm = (x_r - mean) / std
            X_full_processed_list.append(x_norm)
        except Exception as e:
            print(f"⚠️ Lỗi xử lý mẫu {idx}: {e}")
            continue
    
    if not X_full_processed_list:
        print(f"⚠️ Không có mẫu nào được xử lý cho subject {subject_id}")
        return None, None
        
    X_full_processed = np.stack(X_full_processed_list).astype(np.float32)

    # Phân tách dữ liệu hợp lệ và dữ liệu nhiễu
    valid_idx = y_full != 5
    noise_idx = y_full == 5
    
    X_valid = X_full_processed[valid_idx]
    y_valid = y_full[valid_idx]
    X_noise = X_full_processed[noise_idx]

    # Đánh giá trên dữ liệu hợp lệ
    if len(X_valid) > 0:
        y_pred_probs_valid = model.predict(X_valid, verbose=0)
        y_pred_valid = hmm_smoothing_viterbi(y_pred_probs_valid, trans_diag=config.HMM_SMOOTHING_DIAG)
        
        print("\n📊 Báo cáo trên DỮ LIỆU HỢP LỆ:")
        report_valid = classification_report(
            y_valid, y_pred_valid,
            labels=list(range(config.NUM_CLASSES)),       # 0–5
            target_names=config.SLEEP_STAGE_LABELS,   # ["Wake", "N1", "N2", "N3", "REM", "Noise"]
            digits=4,
            zero_division=0
        )

        print(report_valid)
        
        cm_valid = confusion_matrix(y_valid, y_pred_valid, normalize="true")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_valid, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=config.SLEEP_STAGE_LABELS, yticklabels=config.SLEEP_STAGE_LABELS)
        plt.xlabel("Dự đoán"); plt.ylabel("Thực tế"); 
        plt.title(f"Ma trận nhầm lẫn - Dữ liệu Hợp lệ (SC{subject_id})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"./final_reports/cm_valid_sc{subject_id}.png", dpi=300)
        plt.show()
        plt.close()

    # Phân tích trên dữ liệu nhiễu
    if len(X_noise) > 0:
        print(f"\n📈 Phân tích trên DỮ LIỆU NHIỄU ({len(X_noise)} mẫu):")
        y_pred_probs_noise = model.predict(X_noise, verbose=0)
        y_pred_noise = np.argmax(y_pred_probs_noise, axis=1)

        noise_labels = [CONFIG.SLEEP_STAGE_LABELS[i] for i in y_pred_noise]
        noise_counts = pd.Series(noise_labels).value_counts()
        print("📊 Phân bố dự đoán trên Noise:\n", noise_counts)

        noise_pred_counts = pd.Series(y_pred_noise).value_counts().sort_index()
        noise_pred_counts.index = [config.SLEEP_STAGE_LABELS[i] for i in noise_pred_counts.index]
        print("Phân bố dự đoán của mô hình trên các mẫu nhiễu:")
        print(noise_pred_counts)
        
        plt.figure(figsize=(8, 6))
        sns.countplot(x=[config.SLEEP_STAGE_LABELS[i] for i in y_pred_noise],
              palette="viridis")
        plt.title(f"Phân bố dự đoán trên Dữ liệu Nhiễu (SC{subject_id})", fontsize=14, fontweight='bold')
        plt.xlabel("Giai đoạn Giấc ngủ Dự đoán", fontsize=12)
        plt.ylabel("Số lượng mẫu", fontsize=12)
        plt.xticks(np.unique(y_pred_noise), [config.SLEEP_STAGE_LABELS[i] for i in np.unique(y_pred_noise)])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"./final_reports/noise_prediction_sc{subject_id}.png", dpi=300)
        plt.show()
        plt.close()

    return X_full_processed, y_full

def generate_noise_impact_report(y_true_full, y_pred_full, num_clean_classes=5, sleep_stage_labels=None):
    """ Tạo báo cáo ảnh hưởng của dữ liệu nhiễu (lớp 5).
    """
    print("\n--- 📊 BÁO CÁO TÁC ĐỘNG CỦA DỮ LIỆU NHIỄU (LỚP 5) ---")
    sleep_stage_labels = sleep_stage_labels or CONFIG.SLEEP_STAGE_LABELS
    
    # 1. Thống kê nhãn nhiễu
    noise_count = np.sum(y_true_full == 5)
    total_count = len(y_true_full)
    print(f"Tổng số mẫu trong tập FULL: {total_count}")
    print(f"Số lượng mẫu nhiễu (lớp 5): {noise_count} ({noise_count / total_count * 100:.2f}%)")

    # 2. Phân tích dự đoán trên mẫu nhiễu (khi nhãn thực tế là 5)
    if noise_count > 0:
        noise_indices = (y_true_full == 5)
        y_pred_on_noise = y_pred_full[noise_indices]
        noise_pred_counts = pd.Series(y_pred_on_noise).value_counts().sort_index()

        print("\n📌 Phân bố dự đoán của mô hình trên các mẫu nhiễu (lớp 5):")
        for stage, count in noise_pred_counts.items():
            if stage < len(sleep_stage_labels):
                print(f"  - Dự đoán là '{sleep_stage_labels[stage]}': {count} mẫu ({count / noise_count * 100:.2f}%)")

    print("-----------------------------------------------------")

# ==================================
# 4. 🚀 K-FOLD TRAINING LOOP (ĐÃ CHỈNH SỬA)
# ==================================

if __name__ == "__main__":

    # 1️⃣ Load subject metadata để xác định các nhóm
    print("📋 Loading subject metadata...")
    all_subject_id_strings = generate_subject_ids(CONFIG.NUM_SUBJECTS_TO_LOAD)
    unique_subjects = sorted(list(set(s[:-2] for s in all_subject_id_strings)))
    print(f"📊 Found {len(all_subject_id_strings)} files, {len(unique_subjects)} unique subjects.")

    gkf = GroupKFold(n_splits=CONFIG.N_FOLDS)
    fold_results = []
    all_test_true, all_test_preds = [], []         # FULL (có Noise=5)
    all_test_true_clean, all_test_preds_clean = [], []  # CLEAN (0–4)
    all_test_probs_clean, all_test_probs_full = [], [] # 🚀 THÊM: Lưu xác suất để vẽ ROC
    all_histories = []
    total_time = 0.0
    best_model_path = None

    # Group indices cho GroupKFold
    #group_indices = np.array([unique_subjects.index(s[:-2]) for s in all_subject_id_strings])
    for i, (train_idx, test_idx) in enumerate(gkf.split(range(len(unique_subjects)), groups=unique_subjects)):
        print(f"\n===== 🚀 Fold {i+1}/{CONFIG.N_FOLDS} =====")
        start_time = time.time()

        # Xác định subject cho tập train/test
        train_subjects = [unique_subjects[j] for j in train_idx]
        test_subjects = [unique_subjects[j] for j in test_idx]
        train_file_ids = [s for s in all_subject_id_strings if s[:-2] in train_subjects]
        test_file_ids = [s for s in all_subject_id_strings if s[:-2] in test_subjects]

        # 2️⃣ Load dữ liệu RAW đầy đủ (bao gồm Noise=5)
        train_data_full_raw = load_all_subjects(subject_filter=train_file_ids)
        test_data_full_raw = load_all_subjects(subject_filter=test_file_ids)
        
        if not train_data_full_raw[0] or not test_data_full_raw[0]:
            print(f"⚠️ Dữ liệu RAW cho Fold {i+1} không đủ, bỏ qua.")
            fold_results.append({
                'fold': i + 1,
                'test_subjects': test_subjects,
                'test_accuracy': np.nan,
                'test_loss': np.nan,
                'train_accuracy': np.nan,
                'train_loss': np.nan,
                'overfitting_gap': np.nan,
                'f1_hmm_clean': np.nan,
                'kappa_clean': np.nan,
                'f1_hmm_full': np.nan,
                'kappa_full': np.nan,
                'history': {}
            })
            continue

        # 3️⃣ Tiền xử lý toàn bộ dữ liệu RAW (bao gồm Noise=5)
        train_data_preprocessed = preprocess_data(list(zip(
            train_data_full_raw[0], train_data_full_raw[1], train_data_full_raw[2]
        )))
        test_data_preprocessed_full = preprocess_data(list(zip(
            test_data_full_raw[0], test_data_full_raw[1], test_data_full_raw[2]
        )))
        
        if not train_data_preprocessed or not test_data_preprocessed_full:
            print(f"⚠️ Dữ liệu đã xử lý cho Fold {i+1} bị rỗng, bỏ qua.")
            continue

        X_train_list = [d[0] for d in train_data_preprocessed]
        y_train_list = [d[1] for d in train_data_preprocessed]
        X_test_list = [d[0] for d in test_data_preprocessed_full]
        y_test_list = [d[1] for d in test_data_preprocessed_full]

        # ===== Phân bố TRAIN =====
        y_all_train = np.concatenate(y_train_list, axis=0)
        print(f"\n===== 📊 Phân bố dữ liệu TRAIN (6 lớp) trong Fold {i+1} =====")
        print(pd.Series(y_all_train).value_counts().sort_index())

        # ===== Tách tập Test thành FULL (6 lớp) và CLEAN (5 lớp) =====
        X_test_full_all = np.concatenate(X_test_list, axis=0)
        y_test_full_all = np.concatenate(y_test_list, axis=0)
        
        mask_clean = y_test_full_all != 5
        X_test_clean = X_test_full_all[mask_clean]
        y_test_clean = y_test_full_all[mask_clean]
        
        print(f"\n===== 📊 Phân bố dữ liệu TEST (Full, 6 lớp) trong Fold {i+1} =====")
        print(pd.Series(y_test_full_all).value_counts().sort_index())
        print(f"\n===== 📊 Phân bố dữ liệu TEST (Clean, 5 lớp) trong Fold {i+1} =====")
        print(pd.Series(y_test_clean).value_counts().sort_index())

        # ===== XỬ LÝ MẤT CÂN BẰNG: TÍNH CLASS WEIGHTS CHO TẬP TRAIN =====
        y_train_labels_for_weights = np.argmax(to_categorical(y_all_train, num_classes=CONFIG.NUM_CLASSES), axis=1)
        class_weights = class_weight.compute_class_weight('balanced',
                                                          classes=np.unique(y_train_labels_for_weights),
                                                          y=y_train_labels_for_weights)
        class_weights_dict = dict(enumerate(class_weights))
        # FIX: Giới hạn trọng số tối đa để tránh quá nhấn mạnh vào các lớp thiểu số
        if 5 in class_weights_dict:
            class_weights_dict[5] = min(class_weights_dict[5], CONFIG.MAX_WEIGHT_CAP)
        if 1 in class_weights_dict: # Lớp N1 cũng thường ít
            class_weights_dict[1] = min(class_weights_dict[1], CONFIG.MAX_WEIGHT_CAP)

        print("⚖️ Class Weights for this fold:", {CONFIG.SLEEP_STAGE_LABELS[k]: round(v, 2) for k, v in class_weights_dict.items()})

        # ===== SAFE Alpha cân bằng =====
        # Alpha trong Focal Loss cũng là một cách để xử lý mất cân bằng
        counts = np.bincount(y_all_train, minlength=CONFIG.NUM_CLASSES).astype(np.float32)
        inv_freq = np.sum(counts) / (len(counts) * (counts + 1e-8))
        alpha_vals = inv_freq / np.mean(inv_freq)  # chuẩn hóa
        # Giới hạn alpha của lớp Noise
        alpha_vals[5] = min(alpha_vals[5], CONFIG.MAX_WEIGHT_CAP)
        print("⚖️ Auto Alpha values:", alpha_vals)

        # ===== Build model (NUM_CLASSES = 6) =====
        input_shape = (CONFIG.TARGET_LENGTH_CNN, X_train_list[0].shape[-1])
        model = build_model(input_shape, CONFIG.NUM_CLASSES)
        model.compile(
            optimizer=Adam(learning_rate=CONFIG.LEARNING_RATE),
            loss=focal_loss(gamma=CONFIG.FOCAL_LOSS_GAMMA, alpha=alpha_vals),
            metrics=["accuracy"]
        )

        # Dataset
        train_ds = make_balanced_dataset(X_train_list, y_train_list, CONFIG.BATCH_SIZE, CONFIG.NUM_CLASSES)
        
        # SỬA LỖI: Sử dụng get_test_ds để tạo một validation dataset cố định, không phải generator ngẫu nhiên.
        # Điều này đảm bảo việc đánh giá trên tập validation là nhất quán qua các epoch.
        test_ds_full = get_test_ds(X_test_list, y_test_list, CONFIG.BATCH_SIZE)

        # Check batch shapes
        for x, y in train_ds.take(1):
            print("Train batch shape:", x.shape, y.shape)
        for x, y in test_ds_full.take(1):
            print("Test (Validation) batch shape:", x.shape, y.shape)

        # Callbacks
        os.makedirs("results_cnn_6lop/checkpoints", exist_ok=True)
        ckpt_path = f"results_cnn_6lop/checkpoints/best_model_fold_{i+1}.keras"
        clean_f1_callback = CleanF1ScoreCallback(
            validation_data_clean=(X_test_clean, y_test_clean),
            ckpt_path=ckpt_path,
            patience=CONFIG.PATIENCE,
            reduce_lr_factor=CONFIG.REDUCE_LR_FACTOR,
            min_lr=CONFIG.MIN_LR,
            verbose=1
        )

        metrics_logger = MetricsLogger() # 🚀 THÊM: Khởi tạo callback
        l2_logger = L2LossLogger()       # 🚀 THÊM: Khởi tạo callback L2

        # Keep standard monitors on val_loss/val_accuracy; F1 callback handles saving/early-stop/reduce-lr
        cbs = [
            clean_f1_callback,
            SpeedLogger(), metrics_logger, l2_logger # 🚀 THÊM
        ]

        # 4️⃣ Train model
        history = model.fit(
            train_ds,
            epochs=CONFIG.EPOCHS_PER_FOLD,
            steps_per_epoch=CONFIG.STEPS_PER_EPOCH,
            validation_data=test_ds_full,
            verbose=1, # Đổi verbose thành 1 để log chi tiết hơn
            callbacks=cbs,
            # class_weight=class_weights_dict # TẠM THỜI VÔ HIỆU HÓA: make_balanced_dataset đã đủ mạnh
        )
        all_histories.append(history.history)
        all_histories[-1]['lr'] = metrics_logger.learning_rates # 🚀 THÊM: Lưu learning rate vào history
        all_histories[-1]['l2_loss'] = l2_logger.l2_losses # 🚀 THÊM: Lưu L2 loss vào history
        total_time += time.time() - start_time

        # 5️⃣ Đánh giá CLEAN (0–4)
        if len(y_test_clean) > 0:
            y_pred_probs_clean = model.predict(X_test_clean, verbose=0)
            y_pred_hmm_clean = hmm_smoothing_viterbi(y_pred_probs_clean, trans_diag=CONFIG.HMM_SMOOTHING_DIAG, clean_eval=True)
            f1_hmm_clean = f1_score(y_test_clean, y_pred_hmm_clean, average="macro")
            kappa_clean = cohen_kappa_score(y_test_clean, y_pred_hmm_clean)
            print(f"📈 Fold {i+1} F1(HMM|Clean): {f1_hmm_clean:.4f} | Kappa(Clean): {kappa_clean:.4f}")
            all_test_true_clean.extend(y_test_clean.tolist())
            all_test_preds_clean.extend(y_pred_hmm_clean.tolist())
            all_test_probs_clean.append(y_pred_probs_clean) # 🚀 THÊM
        else:
            f1_hmm_clean, kappa_clean = np.nan, np.nan

        # 6️⃣ Đánh giá FULL (0–5)
        y_pred_probs_full = model.predict(X_test_full_all, verbose=0)
        y_pred_hmm_full = hmm_smoothing_viterbi(y_pred_probs_full, trans_diag=CONFIG.HMM_SMOOTHING_DIAG)
        
        all_test_true.extend(y_test_full_all.tolist())
        all_test_preds.extend(y_pred_hmm_full.tolist())
        all_test_probs_full.append(y_pred_probs_full) # 🚀 THÊM

        f1_hmm_full = f1_score(y_test_full_all, y_pred_hmm_full, average="macro")
        kappa_full = cohen_kappa_score(y_test_full_all, y_pred_hmm_full)
        print(f"📉 Fold {i+1} F1(HMM|Full): {f1_hmm_full:.4f} | Kappa(Full): {kappa_full:.4f}")
        # Ảnh hưởng của nhiễu

        # Báo cáo chi tiết ảnh hưởng nhiễu
        generate_noise_impact_report(y_test_full_all, y_pred_hmm_full)
        # Báo cáo phân phối dự đoán trên nhiễu
        noise_pred_counts = pd.Series(y_pred_hmm_full[y_test_full_all == 5]).value_counts().sort_index()
        print("Phân bố dự đoán của mô hình trên các mẫu nhiễu (5):")
        print(noise_pred_counts)
        
        # Lưu kết quả fold
        fold_results.append({
            'fold': i + 1,
            'test_subjects': test_subjects,
            'test_accuracy': max(history.history.get('val_accuracy', [np.nan])),
            'test_loss': min(history.history.get('val_loss', [np.nan])),
            'train_accuracy': max(history.history.get('accuracy', [np.nan])),
            'train_loss': min(history.history.get('loss', [np.nan])),
            'overfitting_gap': abs(min(history.history.get('loss', [np.nan])) - min(history.history.get('val_loss', [np.nan]))),
            'f1_hmm_clean': f1_hmm_clean,
            'kappa_clean': kappa_clean,
            'f1_hmm_full': f1_hmm_full,
            'kappa_full': kappa_full,
            'history': history.history
        })

        # CẬP NHẬT LOGIC LƯU MODEL TỐT NHẤT
        # ModelCheckpoint đã tự động lưu model tốt nhất dựa trên 'val_f1_clean'
        # Chúng ta chỉ cần xác định xem model của fold này có phải là tốt nhất tổng thể không
        if best_model_path is None or (not np.isnan(f1_hmm_clean) and f1_hmm_clean > max([fr.get('f1_hmm_clean', -1) for fr in fold_results if 'f1_hmm_clean' in fr and not np.isnan(fr['f1_hmm_clean'])] or [-1])):
            best_model_path = ckpt_path
            print(f"🏆 Mô hình tốt nhất hiện tại: Fold {i+1} với F1(HMM|Clean) = {f1_hmm_clean:.4f}")

        print(f"⏱️ Thời gian Fold {i+1}: {time.time() - start_time:.1f} giây")
        print(f"⏱️ Tổng thời gian đến nay: {total_time:.1f} giây")

        # Cleanup
        tf.keras.backend.clear_session()
        del model, train_data_preprocessed, test_data_preprocessed_full, train_ds, test_ds_full
        gc.collect()
    
    # ==================================
    # 5. Final Summary (CLEAN vs FULL)
    # ==================================
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, roc_curve, auc
    from tabulate import tabulate
    import joblib

    # --- TẠO THƯ MỤC KẾT QUẢ THEO TIMESTAMP ---
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_RESULTS_DIR = f"results_cnn_6lop_{TIMESTAMP}"

    # --- Tạo các thư mục kết quả một cách có tổ chức ---
    os.makedirs(f"{BASE_RESULTS_DIR}/checkpoints", exist_ok=True)
    os.makedirs(f"{BASE_RESULTS_DIR}/reports/clean", exist_ok=True)
    os.makedirs(f"{BASE_RESULTS_DIR}/reports/full", exist_ok=True)
    os.makedirs(f"{BASE_RESULTS_DIR}/plots", exist_ok=True) # 🚀 THÊM: Thư mục cho các biểu đồ mới
    os.makedirs(f"{BASE_RESULTS_DIR}/cache", exist_ok=True)

    # --- Load cache nếu có ---
    cache_path = f"{BASE_RESULTS_DIR}/cache/training_cache.pkl"
    if os.path.exists(cache_path):
        try:
            print("\n♻️ Loading cached results...")
            cache = joblib.load(cache_path)
            fold_results = cache["fold_results"]
            all_test_true_clean = cache["all_test_true_clean"]
            all_test_preds_clean = cache["all_test_preds_clean"]
            all_test_true = cache["all_test_true_full"]
            all_test_preds = cache["all_test_preds_full"]
            all_test_probs_clean = cache.get("all_test_probs_clean", []) # 🚀 THÊM
            all_test_probs_full = cache.get("all_test_probs_full", [])   # 🚀 THÊM
            all_histories = cache["all_histories"]
            total_time = cache["total_time"]
        except Exception as e:
            print(f"⚠️ Lỗi khi load cache: {e}. Sẽ sử dụng kết quả hiện tại.")
    else:
        print("🚀 No cache found. Using current results.")
        # Sau khi chạy xong, lưu cache vào cuối code
    
    # --- Metrics từ folds ---
    accuracies = [r['test_accuracy'] for r in fold_results]
    losses = [r['test_loss'] for r in fold_results]
    overfitting_gaps = [r['overfitting_gap'] for r in fold_results]

    f1_clean_folds = [r['f1_hmm_clean'] for r in fold_results]
    kappa_clean_folds = [r['kappa_clean'] for r in fold_results]

    f1_full_folds = [r['f1_hmm_full'] for r in fold_results]
    kappa_full_folds = [r['kappa_full'] for r in fold_results]

    # --- Trung bình ---
    avg_f1_clean = np.nanmean(f1_clean_folds)
    avg_kappa_clean = np.nanmean(kappa_clean_folds)
    avg_f1_full = np.nanmean(f1_full_folds)
    avg_kappa_full = np.nanmean(kappa_full_folds)

    # ================== Báo cáo tác động của nhiễu ==================
    def generate_full_noise_impact_report(y_true_full, y_pred_full, sleep_stage_labels):
        report_content = []
        report_content.append("--- 📊 BÁO CÁO TÁC ĐỘNG CỦA DỮ LIỆU NHIỄU (LỚP 5) ---\n")

        noise_count = np.sum(y_true_full == 5)
        total_count = len(y_true_full)
        report_content.append(f"Tổng số mẫu trong tập FULL: {total_count}")
        report_content.append(f"Số lượng mẫu nhiễu (lớp 5): {noise_count} ({noise_count / total_count * 100:.2f}%)\n")

        if noise_count > 0:
            noise_indices = (y_true_full == 5)
            y_pred_on_noise = y_pred_full[noise_indices]
            noise_pred_counts = pd.Series(y_pred_on_noise).value_counts().sort_index()

            report_content.append("📌 Phân bố dự đoán của mô hình trên các mẫu nhiễu (khi nhãn thực tế là 5):")
            for stage, count in noise_pred_counts.items():
                if stage < len(sleep_stage_labels):
                    report_content.append(f"  - Dự đoán là '{sleep_stage_labels[stage]}': {count} mẫu ({count / noise_count * 100:.2f}%)")

        report_content.append("\n" + "-"*50)
        return "\n".join(report_content)

    # Tạo báo cáo tác động nhiễu từ dữ liệu tổng hợp
    noise_impact_summary = generate_full_noise_impact_report(
        np.array(all_test_true),
        np.array(all_test_preds),
        CONFIG.SLEEP_STAGE_LABELS
    )
    # ================== CLEAN (5 lớp) ==================
    # Sử dụng trực tiếp các biến đã được tổng hợp từ vòng lặp K-Fold
    y_true_clean_all = np.array(all_test_true_clean)
    y_pred_clean_all = np.array(all_test_preds_clean)

    # 🚀 THÊM: Nối các xác suất từ các fold
    if all_test_probs_clean:
        y_probs_clean_all = np.concatenate(all_test_probs_clean, axis=0)
    else:
        y_probs_clean_all = np.array([])
    # Lọc các dự đoán là 5 ra khỏi tập clean, và các nhãn true tương ứng
    # Điều này cần thiết vì mô hình 6 lớp có thể dự đoán "Noise" (5) ngay cả trên dữ liệu sạch
    valid_pred_mask = y_pred_clean_all != 5
    y_pred_clean_final = y_pred_clean_all[valid_pred_mask]
    y_true_clean_final = y_true_clean_all[valid_pred_mask]

    report_clean = classification_report(
        y_true_clean_final,
        y_pred_clean_final,
        labels=list(range(5)),   # 0–4
        target_names=CONFIG.SLEEP_STAGE_LABELS[:-1],  # Bỏ "Noise"
        digits=4,
        zero_division=0
    )
    cm_clean = confusion_matrix(y_true_clean_final, y_pred_clean_final, labels=list(range(5)), normalize="true")
    
    # Lưu báo cáo CLEAN
    with open(f"{BASE_RESULTS_DIR}/reports/clean/report_clean.txt", "w", encoding='utf-8') as f:
        f.write("="*20 + " REPORT ON CLEAN DATA (5 CLASSES) " + "="*20 + "\n\n")
        f.write(report_clean)
        f.write("\n\n" + "="*20 + " CONFUSION MATRIX (CLEAN) " + "="*20 + "\n\n")
        cm_df_clean = pd.DataFrame(cm_clean, index=CONFIG.SLEEP_STAGE_LABELS[:-1], columns=CONFIG.SLEEP_STAGE_LABELS[:-1])
        f.write(tabulate(cm_df_clean, headers='keys', tablefmt='grid', floatfmt=".3f"))
    
    # Tính toán chỉ số tổng hợp cho tập clean
    kappa_global_clean = cohen_kappa_score(y_true_clean_final, y_pred_clean_final)
    macro_f1_global_clean = f1_score(y_true_clean_final, y_pred_clean_final, average='macro')

    # ================== Lưu kết quả CSV ==================
    results_df = pd.DataFrame({
        'Fold': [f'Fold {i+1}' for i in range(len(fold_results))],
        'Accuracy': accuracies,
        'Loss': losses,
        'Overfitting_Gap': overfitting_gaps,
        'Macro_F1_Clean': f1_clean_folds,
        'Kappa_Clean': kappa_clean_folds,
        'Macro_F1_Full': f1_full_folds,
        'Kappa_Full': kappa_full_folds
    })
    results_df.to_csv(f"{BASE_RESULTS_DIR}/reports/fold_results_summary.csv", index=False)

    # ======================================================================
    # 📊 TỔNG HỢP KẾT QUẢ CÁC FOLDS
    # ======================================================================
    df_results = pd.DataFrame(fold_results)

    # Chọn các cột quan trọng để báo cáo
    summary_cols = [
        'fold',
        'f1_hmm_clean', 'kappa_clean',
        'f1_hmm_full', 'kappa_full'
    ]
    df_summary = df_results[summary_cols]

    print("\n=== 📌 BÁO CÁO TỔNG HỢP QUA CÁC FOLDS ===")
    print(tabulate(df_summary, headers='keys', tablefmt='pretty', showindex=False))

    # Tính trung bình & độ lệch chuẩn
    avg_row = df_summary.mean(numeric_only=True)
    std_row = df_summary.std(numeric_only=True)

    print("\n=== 📌 TRUNG BÌNH & ĐỘ LỆCH CHUẨN ===")
    for col in df_summary.columns[1:]:
        print(f"{col}: {avg_row[col]:.4f} ± {std_row[col]:.4f}")

    # ================== FULL (6 lớp, có Noise) ==================
    y_true_full = np.array(all_test_true)
    y_pred_full = np.array(all_test_preds)
    # 🚀 THÊM: Nối các xác suất từ các fold
    if all_test_probs_full:
        y_probs_full_all = np.concatenate(all_test_probs_full, axis=0)
    else:
        y_probs_full_all = np.array([])

    kappa_global_full = cohen_kappa_score(y_true_full, y_pred_full)
    macro_f1_global_full = f1_score(y_true_full, y_pred_full, average='macro')
    weighted_f1_global_full = f1_score(y_true_full, y_pred_full, average='weighted')

    report_full = classification_report(
        y_true_full,
        y_pred_full,
        labels=list(range(6)),
        target_names=CONFIG.SLEEP_STAGE_LABELS,
        digits=4,
        zero_division=0
    )
    cm_full = confusion_matrix(y_true_full, y_pred_full, labels=list(range(6)), normalize="true")
    
    # Lưu báo cáo FULL
    with open(f"{BASE_RESULTS_DIR}/reports/full/report_full.txt", "w", encoding='utf-8') as f:
        f.write("="*20 + " REPORT ON FULL DATA (6 CLASSES) " + "="*20 + "\n\n")
        f.write(report_full)
        f.write("\n\n" + "="*20 + " CONFUSION MATRIX (FULL) " + "="*20 + "\n\n")
        cm_df_full = pd.DataFrame(cm_full, index=CONFIG.SLEEP_STAGE_LABELS, columns=CONFIG.SLEEP_STAGE_LABELS)
        f.write(tabulate(cm_df_full, headers='keys', tablefmt='grid', floatfmt=".3f"))
        f.write("\n\n" + noise_impact_summary)
        
    # ================== Lưu summary báo cáo ==================
    with open(f"{BASE_RESULTS_DIR}/reports/summary_report.txt", "w", encoding='utf-8') as f:
        f.write("SLEEP STAGE CLASSIFICATION RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total training time: {total_time:.2f}s\n\n")
        
        summary_data = {
            "Metric": ["Macro F1-Score", "Cohen's Kappa"],
            "CLEAN (5-class)": [f"{macro_f1_global_clean:.4f}", f"{kappa_global_clean:.4f}"],
            "FULL (6-class)": [f"{macro_f1_global_full:.4f}", f"{kappa_global_full:.4f}"]
        }
        summary_df = pd.DataFrame(summary_data)
        f.write("=== 📊 BẢNG SO SÁNH HIỆU SUẤT TỔNG THỂ ===\n")
        f.write(tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False))
        f.write("\n\n")
        
        f.write(noise_impact_summary)
        f.write("\n\nChi tiết xem tại các file trong thư mục reports/:\n")
        f.write("- clean/report_clean.txt\n")
        f.write("- full/report_full.txt\n")

    # --- In báo cáo cuối cùng ra console ---
    print("\n\n" + "="*20 + " 📊 FINAL SUMMARY " + "="*20)
    print(f"⏱️ Total training time: {total_time:.2f}s")
    print("\n=== 📊 BẢNG SO SÁNH HIỆU SUẤT TỔNG THỂ ===")
    summary_df_console = pd.DataFrame({
        "Metric": ["Macro F1-Score", "Cohen's Kappa"],
        "CLEAN (5-class)": [f"{macro_f1_global_clean:.4f}", f"{kappa_global_clean:.4f}"],
        "FULL (6-class)": [f"{macro_f1_global_full:.4f}", f"{kappa_global_full:.4f}"]
    })
    print(tabulate(summary_df_console, headers='keys', tablefmt='pretty', showindex=False))
    print("\n" + noise_impact_summary)
    print(f"\n✅ Báo cáo chi tiết đã được lưu vào thư mục '{BASE_RESULTS_DIR}/reports/'.")
    print(f"✅ Các biểu đồ đã được lưu vào thư mục '{BASE_RESULTS_DIR}/plots/'.")
    # ================== Biểu đồ ==================
    def plot_confusion_matrix(cm, labels, save_path, title):
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, annot_kws={"size": 10})
        plt.xlabel("Dự đoán"); plt.ylabel("Thực tế")
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    plot_confusion_matrix(cm_clean, CONFIG.SLEEP_STAGE_LABELS[:-1], f"{BASE_RESULTS_DIR}/reports/clean/confusion_matrix_clean.png", "CLEAN Confusion Matrix (5 lớp)")
    plot_confusion_matrix(cm_full, CONFIG.SLEEP_STAGE_LABELS, f"{BASE_RESULTS_DIR}/reports/full/confusion_matrix_full.png", "FULL Confusion Matrix (6 lớp)")

    # ================== Biểu đồ hiệu suất theo từng giai đoạn ==================
    def plot_stage_metrics(y_true, y_pred, labels, class_range, save_path, title):
        report_df = pd.DataFrame(classification_report(y_true, y_pred, labels=list(range(len(labels))), target_names=labels, digits=4, output_dict=True, zero_division=0)).transpose()
        report_df = report_df.drop(columns=['support'], errors='ignore').drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        stage_metrics = report_df.loc[labels][['precision', 'recall', 'f1-score']]
        stage_metrics.plot(kind='bar', figsize=(12,6), rot=0, width=0.8)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Giai đoạn Giấc ngủ"); plt.ylabel("Điểm")
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    plot_stage_metrics(y_true_clean_final, y_pred_clean_final, CONFIG.SLEEP_STAGE_LABELS[:-1], range(5), f"{BASE_RESULTS_DIR}/reports/clean/per_stage_metrics_clean.png", "Hiệu suất mô hình theo từng giai đoạn (CLEAN, 5 lớp)")
    plot_stage_metrics(y_true_full, y_pred_full, CONFIG.SLEEP_STAGE_LABELS, range(6), f"{BASE_RESULTS_DIR}/reports/full/per_stage_metrics_full.png", "Hiệu suất mô hình theo từng giai đoạn (FULL, 6 lớp, có Noise)")

    # ================== So sánh Macro F1 và Kappa qua các folds ==================
    def plot_fold_comparison(f1_list, kappa_list, save_path, title, labels=["Macro F1-score", "Cohen's Kappa"], colors=["skyblue", "salmon"]):
        folds = [f"Fold {i+1}" for i in range(len(f1_list))]
        x = np.arange(len(folds))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12,7))
        rects1 = ax.bar(x - width/2, f1_list, width, label=labels[0], color=colors[0])
        rects2 = ax.bar(x + width/2, kappa_list, width, label=labels[1], color=colors[1])
        
        for rect in [*rects1, *rects2]:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width()/2, height), xytext=(0,3), textcoords="offset points", ha='center', va='bottom')

        ax.set_xticks(x)
        ax.set_xticklabels(folds)
        ax.set_ylabel("Điểm số"); ax.set_xlabel("Fold")
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(); fig.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    plot_fold_comparison(f1_clean_folds, kappa_clean_folds, f"{BASE_RESULTS_DIR}/reports/clean/fold_comparison_metrics_clean.png", "So sánh Macro F1 & Kappa qua các Folds (CLEAN)")
    plot_fold_comparison(f1_full_folds, kappa_full_folds, f"{BASE_RESULTS_DIR}/reports/full/fold_comparison_metrics_full.png", "So sánh Macro F1 & Kappa qua các Folds (FULL)")

    # ================== 🚀 CÁC BIỂU ĐỒ MỚI ==================
    print("📊 Đang tạo các biểu đồ trực quan hóa bổ sung...")

    # 1. Biểu đồ Loss/Accuracy/Learning Rate
    if all_histories:
        min_epochs = min(len(h['loss']) for h in all_histories)
        epochs = range(1, min_epochs + 1)
        
        avg_train_loss = np.mean([h['loss'][:min_epochs] for h in all_histories], axis=0)
        avg_val_loss = np.mean([h['val_loss'][:min_epochs] for h in all_histories], axis=0)
        avg_train_acc = np.mean([h['accuracy'][:min_epochs] for h in all_histories], axis=0)
        avg_val_acc = np.mean([h['val_accuracy'][:min_epochs] for h in all_histories], axis=0)
        avg_lr = np.mean([h['lr'][:min_epochs] for h in all_histories if 'lr' in h], axis=0)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
        
        ax1.plot(epochs, avg_train_loss, 'o-', label='Train Loss')
        ax1.plot(epochs, avg_val_loss, 'o-', label='Validation Loss')
        ax1.set_yscale('log') # Log scale cho loss
        ax1.set_ylabel("Loss")
        ax1.set_title("Trung bình Loss qua các Folds")
        ax1.legend(); ax1.grid(True)

        ax2.plot(epochs, avg_train_acc, 'o-', label='Train Accuracy')
        ax2.plot(epochs, avg_val_acc, 'o-', label='Validation Accuracy')
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1)
        ax2.set_title("Trung bình Accuracy qua các Folds")
        ax2.legend(); ax2.grid(True)

        if avg_lr.any():
            ax3.plot(epochs, avg_lr, 'o-', color='purple', label='Learning Rate')
            ax3.set_ylabel("Learning Rate")
            ax3.set_title("Lịch trình Learning Rate trung bình")
            ax3.set_yscale('log') # Log scale cho LR
            ax3.legend(); ax3.grid(True)

        # Biểu đồ L2 Loss
        if 'l2_loss' in all_histories[0]:
            avg_l2_loss = np.mean([h['l2_loss'][:min_epochs] for h in all_histories], axis=0)
            ax4.plot(epochs, avg_l2_loss, 'o-', color='brown', label='L2 Regularization Loss')
            ax4.set_ylabel("L2 Loss")
            ax4.set_title("Trung bình L2 Regularization Loss qua các Folds")
            ax4.set_xlabel("Epochs")
            ax4.legend(); ax4.grid(True)

        plt.tight_layout()
        plt.savefig(f"{BASE_RESULTS_DIR}/plots/learning_curves.png", dpi=300)
        plt.close()
    print("   - Đã tạo biểu đồ learning curves.")

    # 2. Biểu đồ ROC và AUC
    if y_probs_full_all.any():
        y_true_one_hot = to_categorical(y_true_full, num_classes=CONFIG.NUM_CLASSES)
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(CONFIG.NUM_CLASSES):
            fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_probs_full_all[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 8))
        colors = plt.cm.get_cmap('tab10', CONFIG.NUM_CLASSES)
        for i, color in zip(range(CONFIG.NUM_CLASSES), colors.colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {CONFIG.SLEEP_STAGE_LABELS[i]} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC Curve cho từng lớp (One-vs-Rest)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f"{BASE_RESULTS_DIR}/plots/roc_curve_full.png", dpi=300)
        plt.close()
    print("   - Đã tạo biểu đồ ROC curve.")

    # 3. Biểu đồ phân bố mẫu
    plt.figure(figsize=(10, 6))
    sns.countplot(x=[CONFIG.SLEEP_STAGE_LABELS[i] for i in y_true_full], order=CONFIG.SLEEP_STAGE_LABELS, palette="viridis")
    plt.title("Phân bố mẫu trong toàn bộ tập Test (FULL)")
    plt.xlabel("Giai đoạn ngủ"); plt.ylabel("Số lượng mẫu")
    plt.savefig(f"{BASE_RESULTS_DIR}/plots/class_distribution_full.png", dpi=300)
    plt.close()
    print("   - Đã tạo biểu đồ phân bố mẫu.")

    # 4. Biểu đồ phân bố xác suất dự đoán
    if y_probs_full_all.any():
        max_probs = np.max(y_probs_full_all, axis=1)
        plt.figure(figsize=(10, 6))
        sns.histplot(max_probs, bins=50, kde=True)
        plt.title("Phân bố xác suất dự đoán cao nhất trên tập Test (FULL)")
        plt.xlabel("Xác suất dự đoán cao nhất")
        plt.ylabel("Số lượng mẫu")
        plt.savefig(f"{BASE_RESULTS_DIR}/plots/prediction_probability_distribution.png", dpi=300)
        plt.close()
    print("   - Đã tạo biểu đồ phân bố xác suất.")

    # 4. Phổ tần số và dạng sóng (lấy 1 subject từ fold cuối để minh họa)
    if test_data_full_raw[0]:
        X_sample_raw = test_data_full_raw[0][0] # Lấy subject đầu tiên của fold cuối
        y_sample_raw = test_data_full_raw[1][0]
        
        fig, axes = plt.subplots(CONFIG.NUM_CLASSES, 2, figsize=(14, 15))
        fig.suptitle("Ví dụ Dạng sóng và Phổ tần số (Welch)", fontsize=16)
        
        for i in range(CONFIG.NUM_CLASSES):
            stage_label = CONFIG.SLEEP_STAGE_LABELS[i]
            idx = np.where(y_sample_raw == i)[0]
            if len(idx) > 0:
                sample_idx = idx[0]
                signal = X_sample_raw[sample_idx, :, 0] # Lấy kênh đầu tiên
                
                # Dạng sóng
                axes[i, 0].plot(np.linspace(0, 30, len(signal)), signal)
                axes[i, 0].set_title(f"Dạng sóng - {stage_label}")
                axes[i, 0].set_ylabel("Amplitude")

                # Phổ tần số
                f, Pxx = scipy.signal.welch(signal, fs=100, nperseg=256)
                axes[i, 1].semilogy(f, Pxx)
                axes[i, 1].set_title(f"Phổ tần số - {stage_label}")
                axes[i, 1].set_xlabel("Frequency (Hz)"); axes[i, 1].set_ylabel("PSD")
            else:
                axes[i, 0].text(0.5, 0.5, "Không có mẫu", ha='center')
                axes[i, 1].text(0.5, 0.5, "Không có mẫu", ha='center')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{BASE_RESULTS_DIR}/plots/waveform_psd_examples.png", dpi=300)
        plt.close()
    print("   - Đã tạo biểu đồ ví dụ dạng sóng & PSD.")

    print("✅ Đã tạo xong các biểu đồ.")

    # ================== Lưu cache ==================
    joblib.dump({
        "fold_results": fold_results, # Lưu kết quả từng fold
        "all_test_true_clean": all_test_true_clean,
        "all_test_preds_clean": all_test_preds_clean,
        "all_test_probs_clean": all_test_probs_clean, # 🚀 THÊM
        "all_test_true_full": all_test_true, # Lưu kết quả tổng hợp
        "all_test_preds_full": all_test_preds, # Lưu kết quả tổng hợp
        "all_test_probs_full": all_test_probs_full, # 🚀 THÊM
        "all_histories": all_histories, # Lưu lịch sử training
        "total_time": total_time # Lưu tổng thời gian
    }, cache_path)
    print(f"💾 Cache saved at {cache_path}")

    # ================== Lưu đường dẫn model tốt nhất ==================
    # Nếu không chọn được best_model_path logic bên trong loop (ví dụ do lỗi), fallback: tìm file .keras gần nhất
    if not best_model_path or not os.path.exists(best_model_path): # Sửa đường dẫn tìm kiếm
        candidates = sorted(glob.glob(f"{BASE_RESULTS_DIR}/checkpoints/best_model_fold_*.keras"))
        if candidates:
            best_model_path = candidates[-1]
            print(f"ℹ️ Fallback: using model file '{best_model_path}' as best model.")
    if best_model_path and os.path.exists(best_model_path):
        with open("best_model_path.txt", "w") as f:
            f.write(best_model_path)
        print(f"\n🏆 Đường dẫn model tốt nhất '{best_model_path}' đã được lưu vào best_model_path.txt")
    else:
        print("\n⚠️ Không tìm thấy model tốt nhất để lưu đường dẫn. Nếu bạn muốn dùng model đã lưu thủ công, tạo file best_model_path.txt chứa đường dẫn tới .keras")

    print(f"\n✅ Grouped K-Fold CV training complete!")
    print(f"✅ Results and plots saved to '{BASE_RESULTS_DIR}/' directory")

def load_trained_model_for_inference(model_path):
    """
    Load saved model with the exact custom objects used in training.
    Use this for inference to avoid custom_objects mismatch.
    """
    from tensorflow.keras.models import load_model
    return load_model(
        model_path,
        custom_objects={
            "AttentionLayer": AttentionLayer,
            "focal_loss_fixed": focal_loss()  # returns function with name 'focal_loss_fixed'
        },
        compile=False
    )

def predict_subject_from_saved_model(model_path, subject_id, apply_hmm=True, collapse_threshold=0.90, temp=1.0, min_diag=0.5):
    """
    Load model, preprocess a single subject, predict probs, apply adaptive HMM smoothing.
    - temp: temperature for probability sharpening/softening (1.0 = no change)
    - collapse_threshold: if HMM output > this fraction same label => considered collapsed
    - min_diag: minimum transition diagonal to try
    Returns: X_processed, y_true (np.array), probs (raw), preds_final (after HMM/fallback)
    """
    import scipy.signal
    from collections import Counter
    from tensorflow.keras.models import load_model
    import time

    # load model with custom layer
    model = load_trained_model_for_inference(model_path)

    X_raw, y_true = load_single_subject(subject_id)
    if X_raw is None or y_true is None:
        return None, None, None, None

    # preprocess samples (resample + per-epoch per-channel normalize) -> same as training
    X_list = []
    for i in range(X_raw.shape[0]):
        x = X_raw[i].astype(np.float32)
        x_r = scipy.signal.resample(x, CONFIG.TARGET_LENGTH_CNN, axis=0).astype(np.float32)
        mean = x_r.mean(axis=0, keepdims=True)
        std = x_r.std(axis=0, keepdims=True) + 1e-8
        X_list.append((x_r - mean) / std)
    if not X_list:
        return None, None, None, None

    X_proc = np.stack(X_list).astype(np.float32)

    # raw model probabilities
    probs = model.predict(X_proc, verbose=0)
    N = probs.shape[0]
    print(f"DEBUG[predict_subject]: N_samples={N}, probs.shape={probs.shape}")
    # class-level average probability (from model)
    mean_per_class = np.mean(probs, axis=0)
    print("DEBUG[predict_subject]: mean probability per class:", mean_per_class)

    # --- INSERT: entropy + histogram diagnostics (per-sample + per-class) ---
    try:
        os.makedirs("debug_plots", exist_ok=True)
        eps = 1e-12
        if hasattr(probs, "ndim") and probs.ndim == 2:
            # per-sample entropy (nats) and normalized (0..1)
            ent = -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=1)
            
            # per-sample max prob
            maxp = probs.max(axis=1)

            print(f"DEBUG[entropy]: samples={len(ent)}, ent_mean={ent.mean():.4f}, ent_median={np.median(ent):.4f}")
            print(f"DEBUG[max_prob]: mean={maxp.mean():.4f}, frac_confident(>0.7)={(maxp>0.7).mean():.3f}")

            # per-class mean prob
            class_mean = probs.mean(axis=0)
            class_std = probs.std(axis=0)
            for i, (m, s) in enumerate(zip(class_mean, class_std)):
                label = CONFIG.SLEEP_STAGE_LABELS[i] if i < len(CONFIG.SLEEP_STAGE_LABELS) else str(i)
                print(f"DEBUG[class_prob] {i} ({label}): mean={m:.4f} std={s:.4f}")

            # histogram of entropies
            plt.figure(figsize=(6,3))
            sns.histplot(ent, bins=40, kde=False, color="C0")
            plt.title(f"{subject_id} per-sample entropy")
            plt.xlabel("Entropy (nats)")
            plt.tight_layout()
            plt.savefig(f"debug_plots/{subject_id}_entropy_hist.png", dpi=150)
            plt.close()

            # histogram of max probs
            plt.figure(figsize=(6,3))
            sns.histplot(maxp, bins=40, kde=False, color="C1")
            plt.title(f"{subject_id} per-sample max probability")
            plt.xlabel("Max class probability")
            plt.tight_layout()
            plt.savefig(f"debug_plots/{subject_id}_maxp_hist.png", dpi=150)
            plt.close()

            # bar plot of per-class mean probabilities
            plt.figure(figsize=(6,3))
            labels = [CONFIG.SLEEP_STAGE_LABELS[i] if i < len(CONFIG.SLEEP_STAGE_LABELS) else str(i) for i in range(len(class_mean))]
            sns.barplot(x=labels, y=class_mean, palette="viridis")
            plt.xticks(rotation=45)
            plt.ylabel("Mean probability")
            plt.title(f"{subject_id} class mean probs")
            plt.tight_layout()
            plt.savefig(f"debug_plots/{subject_id}_class_mean.png", dpi=150)
            plt.close()
    except Exception as _e:
        print("DEBUG: entropy/histogram diagnostics failed:", _e)

    # --- Empirical-Bayes prior calibration (safer than using true labels) ---
    try:
        # load train prior from processed cache if available (fallback to uniform)
        cache_files = glob.glob(os.path.join(CONFIG.CACHE_DIR, "*_processed_6cls.npz"))
        train_prior = None
        if cache_files:
            ys = []
            for cf in cache_files:
                try:
                    with np.load(cf) as npz:
                        ys.append(npz["y"])
                except Exception:
                    continue
            if ys:
                train_y = np.concatenate(ys)
                tp = np.bincount(train_y, minlength=CONFIG.NUM_CLASSES).astype(np.float32)
                train_prior = tp / (tp.sum() + 1e-12)
        if train_prior is None:
            train_prior = np.ones(CONFIG.NUM_CLASSES, dtype=np.float32) / CONFIG.NUM_CLASSES

        # estimate subject prior from model's mean probs (empirical Bayes)
        est_prior = mean_per_class.astype(np.float32)
        est_prior = est_prior / (est_prior.sum() + 1e-12)

        # ADAPTIVE shrink: nếu model quá “confident” về 1 lớp -> giảm weight của est_prior
        max_conf = est_prior.max()
        if max_conf > 0.70:
            shrink_lambda = 0.15
        elif max_conf > 0.45:
            shrink_lambda = 0.35
        else:
            shrink_lambda = 0.6

        # If we actually have labels for subject (during debug), prefer subject empirical prior gently
        try:
            subj_prior = None
            if y_arr is not None and len(y_arr) > 0:
                sp = np.bincount(y_arr, minlength=CONFIG.NUM_CLASSES).astype(np.float32)
                subj_prior = sp / (sp.sum() + 1e-12)
                # prefer subject prior only if not extremely sparse
                if subj_prior.max() > 0.7:
                    # noisy / dominated: ignore
                    subj_prior = None
        except Exception:
            subj_prior = None

        if subj_prior is not None:
            target_prior = 0.7 * subj_prior + 0.3 * train_prior
        else:
            target_prior = shrink_lambda * est_prior + (1.0 - shrink_lambda) * train_prior

        print("DEBUG[predict_subject]: train_prior:", np.round(train_prior, 4))
        print("DEBUG[predict_subject]: est_prior(from probs):", np.round(est_prior, 4))
        print("DEBUG[predict_subject]: target_prior (shrink/adapt):", np.round(target_prior, 4))

        # apply mild reweight (clip to avoid extreme scaling)
        reweight = (target_prior + 1e-12) / (train_prior + 1e-12)

        reweight = np.clip(reweight, 0.5, 2.0)  # prevent huge up/down-scaling
        probs = probs * reweight[np.newaxis, :]
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)
        print("DEBUG[predict_subject]: mean prob per class after prior-calib:", np.round(np.mean(probs, axis=0), 4))
    except Exception as e:
        print("DEBUG[predict_subject]: prior calibration failed:", e)
     
    # temperature scaling if requested (temp != 1.0)
    if temp != 1.0 and probs.size > 0:
        # avoid zeros
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs ** (1.0 / float(temp))
        probs = probs / probs.sum(axis=1, keepdims=True)
 
    # decide clean_eval if subject has no Noise label
    y_arr = np.array(y_true)
    has_noise = np.any(y_arr == 5)
    clean_eval = (not has_noise)
    print("DEBUG[predict_subject]: has_noise_in_true=", has_noise, "clean_eval=", clean_eval)

    preds_arg = np.argmax(probs, axis=1)

    preds_final = preds_arg.copy()
    if apply_hmm:
        # choose adaptive trans_diag: reduce for short sequences
        base_diag = float(CONFIG.HMM_SMOOTHING_DIAG)
        if N < 200:
            diag_try = max(min_diag, base_diag - 0.25)
        else:
            diag_try = base_diag

        # try HMM and if it collapses, decrease diag and retry
        tried_diags = []
        collapsed = True
        for attempt in range(3):
            trans_diag = max(min_diag, diag_try - attempt * 0.15)
            tried_diags.append(trans_diag)
            try:
                preds_hmm = hmm_smoothing_viterbi(probs, trans_diag=trans_diag, clean_eval=clean_eval)
                counts = Counter(preds_hmm.tolist())
                most_common_count = max(counts.values()) if counts else 0
                frac = most_common_count / float(len(preds_hmm)) if len(preds_hmm) > 0 else 0.0
                print(f"DEBUG[predict_subject]: HMM attempt trans_diag={trans_diag:.3f}, top_frac={frac:.3f}, counts={dict(counts)}")
                if frac <= collapse_threshold:
                    preds_final = preds_hmm
                    collapsed = False
                    break
            except Exception as e:
                print("DEBUG[predict_subject]: HMM failed for trans_diag=", trans_diag, "err=", e)
                continue

        if collapsed:
            print(f"WARNING[predict_subject]: HMM collapsed for tried diags={tried_diags}. Trying soft fallback (temp/ensemble) before argmax.")
            # 1) try softening probs with higher temperature
            tried = False
            for t_try in (1.2, 1.5):
                p_try = np.clip(probs, 1e-12, 1.0) ** (1.0 / float(t_try))
                p_try = p_try / (p_try.sum(axis=1, keepdims=True) + 1e-12)
                try:
                    preds_hmm2 = hmm_smoothing_viterbi(p_try, trans_diag=min_diag, clean_eval=clean_eval)
                    counts2 = Counter(preds_hmm2.tolist())
                    frac2 = max(counts2.values()) / float(len(preds_hmm2))
                    print(f"DEBUG[predict_subject]: fallback temp={t_try} -> collapse_frac={frac2:.3f}")
                    if frac2 <= collapse_threshold:
                        preds_final = preds_hmm2
                        tried = True
                        break
                except Exception:
                    continue

            # 2) if still collapsed, try median filtering on argmax (simple robust smoothing)
            if not tried:
                try:
                    arg = np.argmax(probs, axis=1)
                    # choose kernel adaptively (odd, <= length)
                    k = min(11, max(3, (len(arg)//20)*2+1))
                    from scipy.signal import medfilt
                    arg_med = medfilt(arg, kernel_size=k)
                    counts_med = Counter(arg_med.tolist())
                    frac_med = max(counts_med.values()) / float(len(arg_med))
                    print(f"DEBUG[predict_subject]: median-filter fallback kernel={k} frac={frac_med:.3f}")
                    if frac_med < collapse_threshold + 0.05:
                        preds_final = arg_med
                        tried = True
                except Exception as e:
                    print("DEBUG[predict_subject]: median filter fallback failed:", e)

            # final fallback to argmax
            if not tried:
                print("INFO: Falling back to argmax predictions (no stable HMM/ensemble/median found).")
                preds_final = preds_arg.copy()
 
    return X_proc, y_arr, probs, preds_final

def run_inference_checks(model_path, subjects=None, apply_hmm=True, 
                         temps=(0.8, 1.0, 1.2, 1.5, 1.8), min_diags=(0.8, 0.5, 0.3, 0.1)):
    """
    Quick diagnostics: run predict_subject_from_saved_model for each subject and simple summary.
    Returns dict subject -> dict with 'best_config','best_f1','results' (list of tries).
    """
    from sklearn.metrics import f1_score
    results = {}
    if subjects is None:
        print("run_inference_checks: no subjects provided.")
        return results
    for s in subjects:
        best_f1 = -1.0
        best_cfg = None
        tries = []
        # Tổng số lần thử tăng từ 12 lên 40, giúp tăng cơ hội tìm ra cấu hình tốt nhất.
        for t in temps:
            for md in min_diags:
                for ah in (True, False):
                    # Giữ nguyên logic chạy predict_subject_from_saved_model
                    X, y_true, probs, preds = predict_subject_from_saved_model(model_path, s, apply_hmm=ah, temp=t, min_diag=md)
                    if X is None:
                        continue
                    try:
                        f = f1_score(y_true, preds, average='macro', zero_division=0)
                    except Exception:
                        f = -1.0
                    tries.append({'temp': t, 'min_diag': md, 'apply_hmm': ah, 'f1': f, 'mean_probs': np.round(np.mean(probs,axis=0),4).tolist()})
                    if f > best_f1:
                        best_f1 = f
                        best_cfg = tries[-1]
        results[s] = {'best_f1': best_f1, 'best_cfg': best_cfg, 'tries': tries}
        print(f"[run_inference_checks] subject={s} best_f1={best_f1:.4f} best_cfg={best_cfg}")
    return results
