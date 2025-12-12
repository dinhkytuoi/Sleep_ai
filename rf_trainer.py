import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import os
import glob
from datetime import datetime

import mne
from scipy.signal import butter, lfilter, welch
from scipy.stats import entropy

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

# ==================================
# ⚙️ CONFIGURATION
# ==================================

class CONFIG:
    SEED = 42
    RAW_DATA_DIR = r"A:\lstm+cnn\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette"
    PROCESSED_DATA_DIR = "./processed_data_5_class"

    MODEL_DIR = "./saved_models_rf_class_weight"
    PLOTS_DIR = "./visualization_plots_rf_class_weight"

    SLEEP_STAGE_LABELS = ["Wake", "N1", "N2", "N3", "REM"]
    ANNOTATION_MAP = {
        "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
        "Sleep stage 3": 3, "Sleep stage 4": 3,
        "Sleep stage R": 4, "Sleep stage ?": -1, "Movement time": -1
    }

    EPOCH_DURATION_S = 30
    FREQ_BANDS = {"delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 12], "sigma": [12, 16], "beta": [16, 30]}

# ==================================
# 🔬 DATA PREPROCESSING
# ==================================

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def get_spectral_features(epoch_data, fs):
    freqs, psd = welch(epoch_data, fs=fs, nperseg=fs*2)
    band_powers = []
    for band in CONFIG.FREQ_BANDS.values():
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        band_powers.append(np.sum(psd[idx_band]))
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    spectral_entropy = entropy(psd_norm)
    return band_powers + [spectral_entropy]

def hjorth_parameters(epoch_data):
    activity = np.var(epoch_data)
    diff1 = np.diff(epoch_data)
    diff2 = np.diff(diff1)
    mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
    complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility if np.var(diff1) > 0 and mobility > 0 else 0
    return [mobility, complexity]

def extract_features(epoch_data, fs):
    stat_features = [np.std(epoch_data), np.ptp(epoch_data)]
    spectral_features = get_spectral_features(epoch_data, fs)
    hjorth_features = hjorth_parameters(epoch_data)
    return stat_features + spectral_features + hjorth_features

def preprocess_raw_edf_data(raw_data_path):
    print("🔬 Starting raw EDF data preprocessing (5 Lớp)...")
    psg_files = sorted(glob.glob(os.path.join(raw_data_path, "*PSG.edf")))
    hypno_files = sorted(glob.glob(os.path.join(raw_data_path, "*Hypnogram.edf")))
    if not psg_files or not hypno_files or len(psg_files) != len(hypno_files): return None, None, None

    all_features, all_labels, all_subject_ids = [], [], []
    for psg_filepath, hypno_filepath in zip(psg_files, hypno_files):
        subject_id = os.path.basename(psg_filepath).split('-')[0]
        print(f"   -> Processing subject: {subject_id}")
        raw = mne.io.read_raw_edf(psg_filepath, preload=True, verbose='WARNING')
        annot = mne.read_annotations(hypno_filepath)
        raw.set_annotations(annot, emit_warning=False)
        eeg_channel, fs = 'EEG Fpz-Cz', int(raw.info['sfreq'])
        eeg_data = raw.get_data(picks=[eeg_channel])[0]
        eeg_filtered = bandpass_filter(eeg_data, lowcut=0.5, highcut=45.0, fs=fs)
        events, _ = mne.events_from_annotations(raw, event_id=CONFIG.ANNOTATION_MAP, chunk_duration=CONFIG.EPOCH_DURATION_S)

        for event in events:
            start_sample, _, label = event
            if label == -1: continue
            end_sample = start_sample + CONFIG.EPOCH_DURATION_S * fs
            if end_sample > len(eeg_filtered): continue
            epoch_segment = eeg_filtered[start_sample:end_sample]
            features = extract_features(epoch_segment, fs)
            all_features.append(features)
            all_labels.append(label)
            all_subject_ids.append(subject_id)

    print("✅ Raw data preprocessing complete.")
    return np.array(all_features), np.array(all_labels), np.array(all_subject_ids)

def load_data():
    os.makedirs(CONFIG.PROCESSED_DATA_DIR, exist_ok=True)
    X_path = os.path.join(CONFIG.PROCESSED_DATA_DIR, "X_features.npy")
    y_path = os.path.join(CONFIG.PROCESSED_DATA_DIR, "y_labels.npy")
    subjects_path = os.path.join(CONFIG.PROCESSED_DATA_DIR, "subject_ids.npy")
    try:
        print(f"📥 Attempting to load pre-processed 5-class data from {CONFIG.PROCESSED_DATA_DIR}...")
        X, y, subject_ids = np.load(X_path), np.load(y_path), np.load(subjects_path)
        print(f"✅ Pre-processed 5-class data loaded successfully: X{X.shape}, y{y.shape}")
        return X, y, subject_ids
    except FileNotFoundError:
        print(f"❌ Pre-processed 5-class data not found in {CONFIG.PROCESSED_DATA_DIR}. Processing from raw EDF files...")
        X, y, subject_ids = preprocess_raw_edf_data(CONFIG.RAW_DATA_DIR)
        if X is not None and len(X) > 0:
            print("💾 Saving 5-class processed data for future use...")
            np.save(X_path, X); np.save(y_path, y); np.save(subjects_path, subject_ids)
            print("✅ Processed 5-class data saved.")
        return X, y, subject_ids

# ==================================
# 🤖 MODEL TRAINING (LƯU MODEL)
# ==================================

def train_random_forest_model(X, y, subject_ids):
    print("🤖 Training Optimized Random Forest (5 Lớp, 2/3 Train, 1/3 Test, Class Weight)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    unique_subjects = np.unique(subject_ids)
    if len(unique_subjects) < 2:
        print("❌ Cần ít nhất 2 subject để chia train/test.")
        return [None]*11

    np.random.seed(CONFIG.SEED)
    np.random.shuffle(unique_subjects)

    split_idx = int(len(unique_subjects) * 2 / 3)
    train_subjects = unique_subjects[:split_idx]
    test_subjects = unique_subjects[split_idx:]

    print(f"   -> Tổng số subjects: {len(unique_subjects)}")
    print(f"   -> Subjects huấn luyện (2/3): {len(train_subjects)}")
    print(f"   -> Subjects kiểm tra (1/3): {len(test_subjects)}")

    train_mask = np.isin(subject_ids, train_subjects)
    test_mask = np.isin(subject_ids, test_subjects)

    X_train, y_train = X_scaled[train_mask], y[train_mask]
    X_test, y_test = X_scaled[test_mask], y[test_mask] # X_test được tạo ở đây
    groups_train = subject_ids[train_mask]

    print(f"   -> Tổng số epoch: {len(X_scaled)}")
    print(f"   -> Epochs huấn luyện: {len(X_train)}")
    print(f"   -> Epochs kiểm tra: {len(X_test)}")
    print("   -> Sử dụng 'class_weight=balanced' để xử lý mất cân bằng.")

    print("🌳 Training model...")
    model = RandomForestClassifier(n_estimators=200,
                                   random_state=CONFIG.SEED,
                                   n_jobs=-1,
                                   max_depth=25,
                                   min_samples_leaf=5,
                                   class_weight='balanced'
                                  )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)

    print(f"\n🎯 Model Performance (trên 1/3 test subjects):")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score (Macro): {f1:.4f}")
    print(f"   Cohen's Kappa: {kappa:.4f}")

    print("\n💾 Saving model, scaler, and subject lists...")
    os.makedirs(CONFIG.MODEL_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(CONFIG.MODEL_DIR, "rf_model.joblib"))
    joblib.dump(scaler, os.path.join(CONFIG.MODEL_DIR, "rf_scaler.joblib"))
    np.save(os.path.join(CONFIG.MODEL_DIR, "rf_test_subjects.npy"), test_subjects)

    print(f"✅ Model và data đã được lưu vào {CONFIG.MODEL_DIR}")

    return model, scaler, y_test, y_pred, accuracy, f1, kappa, X_train, y_train, groups_train, X_test

# ==================================
# 📊 VISUALIZATION
# ==================================

def plot_stage_distribution(y_data, title, filename):
    plt.figure(figsize=(10, 6))
    unique_ticks = sorted(np.unique(y_data))
    unique_labels = [CONFIG.SLEEP_STAGE_LABELS[i] for i in unique_ticks if i < len(CONFIG.SLEEP_STAGE_LABELS)]
    sns.countplot(x=y_data, order=unique_ticks)
    plt.title(f'Phân bố giai đoạn - {title}', fontsize=16)
    plt.xlabel('Giai đoạn ngủ', fontsize=12)
    plt.ylabel('Số lượng Epochs', fontsize=12)
    plt.xticks(ticks=unique_ticks, labels=unique_labels)
    save_path = os.path.join(CONFIG.PLOTS_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"   -> Đã lưu biểu đồ phân bố: {filename}")

def plot_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=CONFIG.SLEEP_STAGE_LABELS,
                yticklabels=CONFIG.SLEEP_STAGE_LABELS)
    plt.title('Ma trận nhầm lẫn (Tập Test - Chuẩn hóa theo Recall)', fontsize=16)
    plt.xlabel('Nhãn dự đoán', fontsize=12)
    plt.ylabel('Nhãn thực tế', fontsize=12)
    save_path = os.path.join(CONFIG.PLOTS_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"   -> Đã lưu ma trận nhầm lẫn: {filename}")

def plot_per_stage_performance(y_true, y_pred, filename):
    report = classification_report(y_true, y_pred, target_names=CONFIG.SLEEP_STAGE_LABELS,
                                   output_dict=True, zero_division=0)
    df = pd.DataFrame(report).T
    df = df.drop(['accuracy', 'macro avg', 'weighted avg'])
    df = df.drop(columns=['support'])
    df.plot(kind='bar', figsize=(12, 7), rot=0)
    plt.title('Hiệu suất chi tiết từng giai đoạn (Tập Test)', fontsize=16)
    plt.xlabel('Giai đoạn ngủ', fontsize=12)
    plt.ylabel('Điểm số (0.0 - 1.0)', fontsize=12)
    plt.legend(loc='lower right')
    save_path = os.path.join(CONFIG.PLOTS_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"   -> Đã lưu biểu đồ hiệu suất: {filename}")

def plot_f1_kappa_comparison(f1, kappa, filename):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Macro F1-Score', "Cohen's Kappa"], y=[f1, kappa])
    plt.title('So sánh chỉ số tổng quát (Tập Test)', fontsize=16)
    plt.ylabel('Điểm số', fontsize=12)
    plt.ylim(0, 1.0)
    save_path = os.path.join(CONFIG.PLOTS_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"   -> Đã lưu biểu đồ F1-Kappa: {filename}")

def plot_learning_curve(estimator, X_train, y_train, groups_train, filename):
    """Vẽ đường cong học tập."""
    print("   -> Bắt đầu vẽ đường cong học tập...")

    n_splits_cv = min(3, len(np.unique(groups_train)))
    train_sizes_abs = np.linspace(0.2, 1.0, 3)

    if n_splits_cv < 2:
        print("   -> Không đủ nhóm (subject) để thực hiện CV. Bỏ qua vẽ đường cong học tập.")
        return

    print(f"   -> Chạy learning_curve với {n_splits_cv} folds và {len(train_sizes_abs)} kích thước.")

    try:
        cv = GroupKFold(n_splits=n_splits_cv)
        train_sizes, train_scores, valid_scores = learning_curve(
            estimator=estimator,
            X=X_train, y=y_train, groups=groups_train,
            cv=cv, n_jobs=-1,
            train_sizes=train_sizes_abs,
            scoring='f1_macro',
            error_score='raise'
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        valid_scores_mean = np.mean(valid_scores, axis=1)
        valid_scores_std = np.std(valid_scores, axis=1)

        plt.figure(figsize=(12, 8))
        plt.title('Đường cong học tập (Learning Curve - Dùng Class Weight)', fontsize=16)
        plt.xlabel('Số lượng mẫu huấn luyện', fontsize=12)
        plt.ylabel('F1-Score (Macro)', fontsize=12)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color='r')
        plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                         valid_scores_mean + valid_scores_std, alpha=0.1, color='g')
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Điểm huấn luyện (Training score)')
        plt.plot(train_sizes, valid_scores_mean, 'o-', color='g', label='Điểm kiểm định (Cross-validation score)')
        plt.legend(loc='best')
        save_path = os.path.join(CONFIG.PLOTS_DIR, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"   -> Đã lưu đường cong học tập: {filename}")

    except Exception as e:
        print(f"   -> ❌ LỖI khi vẽ đường cong học tập: {e}")
        print("   -> Bỏ qua việc vẽ biểu đồ này.")

# --- BẮT ĐẦU PHẦN MỚI ---
def plot_rf_training_dynamics(X_train, y_train, X_test, y_test, filename):
    """
    Vẽ biểu đồ Accuracy và F1-Score (trên tập Test)
    khi tăng dần số lượng cây (n_estimators)
    nhằm mô phỏng quá trình hội tụ của mô hình.
    """
    print("   -> Đang vẽ biểu đồ quá trình hội tụ (Accuracy & F1 theo số lượng cây)...")
    # Giảm số điểm để chạy nhanh hơn (từ 10 đến 200, 10 điểm)
    n_estimators_list = np.linspace(10, 200, 10, dtype=int)
    acc_scores, f1_scores = [], []

    for n in n_estimators_list:
        # Huấn luyện model tạm thời với số lượng cây 'n'
        model_temp = RandomForestClassifier(
            n_estimators=n, random_state=CONFIG.SEED, n_jobs=-1,
            max_depth=25, min_samples_leaf=5, class_weight='balanced'
        )
        model_temp.fit(X_train, y_train)
        y_pred = model_temp.predict(X_test) # Đánh giá trên tập Test
        acc_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, acc_scores, 'o-', label='Accuracy (trên tập Test)', color='steelblue')
    plt.plot(n_estimators_list, f1_scores, 's--', label='Macro F1-Score (trên tập Test)', color='darkorange')
    plt.title('Quá trình hội tụ theo số lượng cây (Random Forest)', fontsize=16)
    plt.xlabel('Số lượng cây (n_estimators)', fontsize=12)
    plt.ylabel('Điểm số (trên tập Test)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    save_path = os.path.join(CONFIG.PLOTS_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"   -> Đã lưu biểu đồ quá trình hội tụ: {filename}")


def plot_prediction_confidence_distribution(model, X_test, y_test, filename):
    """
    Vẽ phân bố xác suất dự đoán cao nhất trên tập test (độ tự tin của mô hình).
    """
    print("   -> Đang vẽ phân bố xác suất dự đoán (prediction confidence)...")
    # Kiểm tra xem model có hỗ trợ predict_proba không
    if not hasattr(model, "predict_proba"):
        print("   -> Model không hỗ trợ predict_proba, bỏ qua.")
        return

    # Lấy xác suất dự đoán cho từng lớp
    y_proba = model.predict_proba(X_test)
    # Lấy xác suất cao nhất (độ tự tin) cho mỗi dự đoán
    max_probs = np.max(y_proba, axis=1)

    # Vẽ biểu đồ histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(max_probs, bins=30, kde=True, color='royalblue')
    plt.title('Phân bố xác suất dự đoán cao nhất trên tập Test', fontsize=16)
    plt.xlabel('Xác suất dự đoán cao nhất (Độ tự tin)', fontsize=12)
    plt.ylabel('Số lượng mẫu', fontsize=12)
    plt.grid(alpha=0.3)
    save_path = os.path.join(CONFIG.PLOTS_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"   -> Đã lưu biểu đồ phân bố xác suất: {filename}")
# --- KẾT THÚC PHẦN MỚI ---

# ==================================
# 🚀 MAIN PIPELINE (ĐÃ CẬP NHẬT)
# ==================================
def main():
    print("🚀 SLEEP STAGE CLASSIFICATION PIPELINE (5 LỚP, Dùng Class Weight)")
    print("=" * 60)

    os.makedirs(CONFIG.PLOTS_DIR, exist_ok=True)

    X, y, subject_ids = load_data()
    if X is None or len(X) == 0:
        print("❌ Không có dữ liệu để xử lý. Dừng pipeline.")
        return

    plot_stage_distribution(y, "Toàn bộ Dữ liệu (5 Lớp)", "1_distribution_full_dataset.png")

    results = train_random_forest_model(X, y, subject_ids)
    if results[0] is None:
        print("❌ Huấn luyện thất bại.")
        return

    model, scaler, y_test, y_pred, acc, f1, kappa, X_train, y_train, groups_train, X_test = results

    print("\n📋 DETAILED CLASSIFICATION REPORT (5 Lớp, Class Weight, trên tập 1/3 test subjects):")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=CONFIG.SLEEP_STAGE_LABELS, digits=4, zero_division=0))
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Overall Macro F1-Score: {f1:.4f}")
    print(f"Overall Cohen's Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")

    print("\n📊 Generating and saving visualization plots...")

    plot_stage_distribution(y_test, "Tập Test (5 Lớp, 1/3 Subjects)", "2_distribution_test_set.png")
    plot_confusion_matrix(y_test, y_pred, "3_confusion_matrix.png")
    plot_per_stage_performance(y_test, y_pred, "4_per_stage_performance.png")
    plot_f1_kappa_comparison(f1, kappa, "5_f1_vs_kappa.png")

    base_model_for_lc = RandomForestClassifier(
        n_estimators=200, random_state=CONFIG.SEED, n_jobs=-1,
        max_depth=25, min_samples_leaf=5, class_weight='balanced'
    )

    plot_learning_curve(base_model_for_lc, X_train, y_train, groups_train, "6_learning_curve.png")

    # === GỌI CÁC HÀM VẼ BIỂU ĐỒ BỔ SUNG ===
    print("\n📊 Generating supplementary visualization plots...")
    # Chú ý: plot_rf_training_dynamics sẽ chạy hơi chậm vì nó huấn luyện lại model 10 lần
    plot_rf_training_dynamics(X_train, y_train, X_test, y_test, "7_rf_training_dynamics.png")
    plot_prediction_confidence_distribution(model, X_test, y_test, "8_prediction_confidence_distribution.png")

    print(f"\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📈 Tất cả 8 biểu đồ đã được lưu vào thư mục: {CONFIG.PLOTS_DIR}")
    print(f"📦 Model đã được lưu vào thư mục: {CONFIG.MODEL_DIR}")

if __name__ == "__main__":
    main()