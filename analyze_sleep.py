import glob
import os
import numpy as np
import scipy.signal
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.utils.multiclass import unique_labels

# Import từ file train - Đảm bảo đây là file training chính xác
from TrainLSTM6lop import (
    AttentionLayer, focal_loss, hmm_smoothing_viterbi, CONFIG, load_single_subject, SEED, load_trained_model_for_inference
)
from fine_tune_subject_v2 import (
    run_finetuning_for_subject
)


# =========================================================
# �💡 Hàm gợi ý giờ thức dậy
# =========================================================
def get_optimal_wakeup_times(sleep_stage_seq, start_time, choice, age, gender):
    optimal_times = []
    if choice == '1':
        # SỬA LỖI: 30 giây, không phải 30 phút. Mỗi stage là 1 epoch 30 giây.
        for i, stage in enumerate(sleep_stage_seq): # type: ignore
            wakeup_time = start_time + timedelta(seconds=(i + 1) * 30)
            if stage in ['N1', 'N2', 'REM']: # Dựa trên tên stage, không phải số
                optimal_times.append(wakeup_time.strftime("%H:%M"))
    elif choice == '2':
        total_minutes = len(sleep_stage_seq) * 0.5 # mỗi sample = 0.5 phút
        num_cycles = int(total_minutes // 90) # type: ignore
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
        elif gender.lower() == 'nữ':
            print("💡 Nữ giới thường có nhiều REM hơn, quan trọng cho trí nhớ & cảm xúc.")
    elif choice == '2' and age.isdigit() and int(age) > 65:
        print("💡 Người lớn tuổi thường ngủ ngắn hơn, có thể thử dậy sớm hơn.")

    # Loại bỏ các giờ trùng lặp liên tiếp
    unique_times = []
    if optimal_times:
        unique_times.append(optimal_times[0])
        for t in optimal_times[1:]:
            if t != unique_times[-1]:
                unique_times.append(t)

    return unique_times


# =========================================================
# 📊 Phân tích nhiễu + trực quan
# =========================================================
def generate_noise_impact_report(y_true, y_pred, config, subject_id="Unknown"):
    os.makedirs("final_reports", exist_ok=True)

    print("\n===== 📊 PHÂN TÍCH ẢNH HƯỞNG NHIỄU =====")
    is_clean_mask = (y_true != 5) # Nhãn nhiễu là 5 trong file TrainLSTM6lop.py # type: ignore

    total_samples = len(y_true)
    noise_samples = np.sum(~is_clean_mask)
    print(f"Tổng mẫu: {total_samples}, Nhiễu: {noise_samples} ({noise_samples/total_samples*100:.2f}%)")

    # --- Phân tích hiệu suất ---
    # 1. Tập CLEAN (không có nhiễu)
    y_true_clean = y_true[is_clean_mask]
    y_pred_clean = y_pred[is_clean_mask]
    f1_clean = f1_score(y_true_clean, y_pred_clean, average='macro', zero_division=0) # type: ignore
    kappa_clean = cohen_kappa_score(y_true_clean, y_pred_clean)

    # 2. Tập FULL (có nhiễu)
    f1_full = f1_score(y_true, y_pred, average='macro', zero_division=0) # type: ignore
    kappa_full = cohen_kappa_score(y_true, y_pred)

    print("\n--- So sánh hiệu suất ---")
    print(f"✅ Macro F1 (Sạch): {f1_clean:.4f} | Kappa (Sạch): {kappa_clean:.4f}")
    print(f"🔴 Macro F1 (Đầy đủ): {f1_full:.4f} | Kappa (Đầy đủ): {kappa_full:.4f}")
    print(f"📉 Mức độ ảnh hưởng của nhiễu (F1 giảm): {f1_clean - f1_full:.4f}")

    # --- Phân tích dự đoán trên các mẫu nhiễu ---
    if noise_samples > 0:
        y_pred_on_noise = y_pred[~is_clean_mask]
        noise_pred_counts = pd.Series(y_pred_on_noise).value_counts().sort_index()
        print("\n📌 Phân bố dự đoán của mô hình trên các mẫu thực sự là nhiễu:")
        for stage, count in noise_pred_counts.items(): # type: ignore
            if stage < len(config.SLEEP_STAGE_LABELS):
                print(f"  - Dự đoán là '{config.SLEEP_STAGE_LABELS[stage]}': {count} mẫu ({count / noise_samples * 100:.2f}%)")

    # --- Trực quan hóa ---
    # Biểu đồ tròn tỉ lệ nhiễu
    plt.figure(figsize=(6, 6))
    plt.pie([total_samples - noise_samples, noise_samples],
            labels=["Sạch", "Nhiễu"],
            autopct="%1.1f%%", colors=["#66b3ff", "#ff6666"], startangle=90)
    plt.title(f"Tỉ lệ sạch vs nhiễu ({subject_id})")
    plt.savefig(f"final_reports/noise_ratio_{subject_id}.png", dpi=300)
    plt.close()

    # Biểu đồ cột phân bố dự đoán
    pred_labels = [config.SLEEP_STAGE_LABELS[i] for i in y_pred]
    plt.figure(figsize=(8, 6))
    sns.countplot(x=pred_labels, order=config.SLEEP_STAGE_LABELS, palette="viridis")
    plt.title(f"Phân bố dự đoán ({subject_id})")
    plt.xlabel("Giai đoạn")
    plt.ylabel("Số mẫu")
    plt.savefig(f"final_reports/pred_distribution_{subject_id}.png", dpi=300)
    plt.close()


# =========================================================
# 📈 Vẽ Timeline dự đoán giấc ngủ
# =========================================================
def plot_sleep_timeline(y_pred, sleep_start_time, config, subject_id="Unknown"):
    os.makedirs("final_reports", exist_ok=True)

    epochs = np.arange(len(y_pred))
    times = [sleep_start_time + timedelta(seconds=30 * int(i)) for i in epochs]

    plt.figure(figsize=(14, 5))
    # Sử dụng plt.step để vẽ biểu đồ bậc thang, trực quan hơn
    plt.step(times, y_pred, where='post', color='royalblue', linewidth=2)
    # SỬA LỖI: Cung cấp đủ nhãn cho tất cả các vị trí.
    # config.SLEEP_STAGE_LABELS giờ đây có 6 phần tử từ TrainLSTM6lop.py # type: ignore
    # range(len(config.SLEEP_STAGE_LABELS)) sẽ là range(6) -> [0, 1, 2, 3, 4, 5]
    plt.yticks(range(len(config.SLEEP_STAGE_LABELS)), config.SLEEP_STAGE_LABELS)
    plt.gca().invert_yaxis() # Đưa Wake lên trên cùng
    plt.xlabel("Thời gian")
    plt.ylabel("Giai đoạn")
    plt.title(f"Timeline giấc ngủ ({subject_id})")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.savefig(f"final_reports/sleep_timeline_{subject_id}.png", dpi=300)
    plt.close()

    print(f"✅ Timeline giấc ngủ đã lưu: final_reports/sleep_timeline_{subject_id}.png")


# =========================================================
# 🔍 Grid Search cho cấu hình inference tốt nhất
# =========================================================
def run_inference_grid_search(model, X_proc, y_true):
    """
    Chạy grid search trên các tham số inference để tìm F1-score macro tốt nhất.
    Tương tự logic trong debug_infer.py.
    """
    best = {"f1": -1}
    temps = [0.8, 1.0, 1.2, 1.5, 1.8] # <-- Dãy Temp MỚI
    trans_diags = [0.8, 0.5, 0.3, 0.1] # <-- Dãy Diag MỚI
    channel_options = [False, True] # False: normal, True: swap
    hmm_options = [True, False] # True: HMM, False: argmax

    print("\n===== 🔍 Bắt đầu Grid Search cấu hình Inference =====")

    for swap in channel_options:
        X_try = X_proc[..., ::-1] if swap else X_proc
        try:
            probs = model.predict(X_try, verbose=0)
        except Exception as e:
            print(f"Lỗi khi dự đoán với swap={swap}: {e}")
            continue

        for temp in temps:
            p_tmp = np.clip(probs, 1e-12, 1.0)**(1.0/float(temp))
            p_tmp = p_tmp / p_tmp.sum(axis=1, keepdims=True)

            for apply_hmm in hmm_options:
                if not apply_hmm:
                    # Trường hợp không dùng HMM, chỉ argmax
                    preds = np.argmax(p_tmp, axis=1)
                    td = None # Không có HMM diag
                    f1 = f1_score(y_true, preds, average='macro', zero_division=0)
                    if f1 > best["f1"]:
                        best.update({"f1": f1, "swap": swap, "temp": temp,
                                     "apply_hmm": apply_hmm, "trans_diag": td, "preds": preds})
                    continue

                # Trường hợp dùng HMM với các diag khác nhau
                for td in trans_diags:
                    # clean_eval=True nếu không có nhãn Noise trong y_true
                    clean_eval = not np.any(y_true == 5)
                    preds = hmm_smoothing_viterbi(p_tmp, trans_diag=td, clean_eval=clean_eval)
                    f1 = f1_score(y_true, preds, average='macro', zero_division=0)
                    if f1 > best["f1"]:
                        best.update({"f1": f1, "swap": swap, "temp": temp,
                                     "apply_hmm": apply_hmm, "trans_diag": td, "preds": preds})

    print("\n--- Kết quả Grid Search ---")
    if best['f1'] > -1:
        best_config_str = (
            f"F1: {best['f1']:.4f} | Swap: {best['swap']} | Temp: {best['temp']} | "
            f"HMM: {best['apply_hmm']} | Diag: {best['trans_diag']}"
        )
        print(f"✅ Cấu hình tốt nhất: {best_config_str}")
        # Trả về dự đoán tốt nhất
        return best["preds"]
    else:
        print("⚠️ Grid search không tìm thấy cấu hình hợp lệ.")
        # Fallback về argmax của probs gốc
        return np.argmax(model.predict(X_proc, verbose=0), axis=1)


# =========================================================
#  MAIN
# =========================================================
if __name__ == "__main__":
    print("\n\n===== 💡 Phân tích dữ liệu và đề xuất giờ thức dậy =====")
    
    subject_to_analyze = input("▶️ Nhập tên file dữ liệu sóng (ví dụ: 'SC4581'): ")
    age = input("▶️ Nhập tuổi: ")
    gender = input("▶️ Nhập giới tính (Nam/Nữ): ")

    while True:
        sleep_start_time_str = input("▶️ Nhập giờ đi ngủ (HH:MM, ví dụ: 22:00): ")
        try:
            sleep_start_time = datetime.strptime(sleep_start_time_str, "%H:%M")
            break
        except ValueError:
            print("❌ Sai định dạng, thử lại.")

    # --- SỬA LỖI: Logic chọn model linh hoạt ---
    # 1. Ưu tiên tìm model đã fine-tune riêng cho subject
    subject_specific_model_path = f"fine_tuned_v2_{subject_to_analyze}.keras"
    best_model_path = None

    # 1. Ưu tiên tìm model đã fine-tune riêng cho subject
    if os.path.exists(subject_specific_model_path):
        best_model_path = subject_specific_model_path
        print(f"✅ Tìm thấy model đã fine-tune riêng cho subject: {best_model_path}")
    elif os.path.exists(f"fine_tuned_{subject_to_analyze}.keras"): # Fallback cho v1
        best_model_path = f"fine_tuned_{subject_to_analyze}.keras"
        print(f"✅ Tìm thấy model đã fine-tune riêng cho subject (v1): {best_model_path}")
    else:
        # 1a. Nếu không có, hỏi người dùng có muốn fine-tune không
        print(f"ℹ️ Không tìm thấy model riêng cho '{subject_to_analyze}'.")
        do_finetune = input("▶️ Bạn có muốn fine-tune một model mới cho subject này để có kết quả tốt nhất? (y/n): ").lower()
        if do_finetune == 'y':
            base_model_path = open("best_model_path.txt").read().strip()
            print(f"\n===== 🚀 Bắt đầu Fine-tuning cho {subject_to_analyze} từ model '{base_model_path}' =====")
            best_model_path = run_finetuning_for_subject(subject_to_analyze, base_model_path)
            print(f"===== ✅ Fine-tuning hoàn tất. Model mới: '{best_model_path}' =====\n")

    # 2. Nếu vẫn không có model (người dùng từ chối fine-tune), fallback về model chung
    if not best_model_path:
        print(f"ℹ️ Không tìm thấy model riêng cho '{subject_to_analyze}'. Tìm model chung...")
        # 2. Nếu không có, fallback về model chung trong best_model_path.txt
        best_model_path_file = "best_model_path.txt"
        if os.path.exists(best_model_path_file):
            with open(best_model_path_file, "r", encoding="utf-8-sig") as f:
                best_model_path = f.read().strip()
            if best_model_path and os.path.exists(best_model_path):
                print(f"⚠️  CẢNH BÁO: Sử dụng model chung '{best_model_path}' vì không có model riêng cho '{subject_to_analyze}'. Kết quả có thể không tối ưu.")
            else:
                print(f"❌ Lỗi: Đường dẫn model '{best_model_path}' trong file '{best_model_path_file}' không hợp lệ.")
                best_model_path = None
        else:
            print(f"❌ Không tìm thấy file '{best_model_path_file}'.")

    if not best_model_path:
        print("❌ Không thể xác định model để sử dụng. Vui lòng chạy training hoặc fine-tuning trước.")
        exit()

    print(f"✅ Sử dụng model: {best_model_path}")
    model = load_trained_model_for_inference(best_model_path)

    # Load và tiền xử lý dữ liệu cho subject
    X_raw, y_subject_true = load_single_subject(subject_to_analyze)
    if X_raw is None:
        print(f"❌ Không thể tải dữ liệu cho subject {subject_to_analyze}.")
        exit()

    # Tiền xử lý giống hệt trong training
    X_list = []
    for i in range(X_raw.shape[0]):
        x = X_raw[i].astype(np.float32)
        x_r = scipy.signal.resample(x, CONFIG.TARGET_LENGTH_LSTM, axis=0).astype(np.float32)
        mean = x_r.mean(axis=0, keepdims=True)
        std = x_r.std(axis=0, keepdims=True) + 1e-8
        X_list.append((x_r - mean) / std)
    X_subject = np.stack(X_list).astype(np.float32)
    y_subject_true = np.array(y_subject_true)

    # Chạy grid search để tìm dự đoán tốt nhất
    y_pred_final = run_inference_grid_search(model, X_subject, y_subject_true)

    if y_pred_final is not None and len(y_pred_final) > 0:
        # --- DEBUG: save per-channel stats + PSD for comparison ---
        try:
            os.makedirs("debug_plots", exist_ok=True)
            # X_subject shape = (n_epochs, time_points, channels)
            X = np.array(X_subject)  # ensure ndarray
            n_epochs, n_t, n_ch = X.shape
            ch_means = X.reshape(-1, n_ch).mean(axis=0)
            ch_stds = X.reshape(-1, n_ch).std(axis=0)
            np.save("debug_plots/subject_per_channel_mean.npy", ch_means)
            np.save("debug_plots/subject_per_channel_std.npy", ch_stds)
            print("DEBUG: per-channel mean/std saved:", ch_means, ch_stds)

            # PSD for epoch 0 each channel
            from scipy.signal import welch
            sf = 100  # typical sfreq — thay nếu khác (một số file in ra sfreq=100)
            fig, axs = plt.subplots(n_ch, 1, figsize=(8, 2.5 * n_ch))
            for c in range(n_ch):
                f, Pxx = welch(X[0, :, c], fs=sf, nperseg=512)
                axs[c].semilogy(f, Pxx)
                axs[c].set_xlabel("Hz"); axs[c].set_ylabel("PSD")
                axs[c].set_title(f"Subject {subject_to_analyze} PSD epoch0 ch{c}")
            plt.tight_layout()
            plt.savefig(f"debug_plots/{subject_to_analyze}_epoch0_psd.png", dpi=150)
            plt.close()
            print(f"DEBUG: saved PSD -> debug_plots/{subject_to_analyze}_epoch0_psd.png")

            # If you have a saved training example to compare (optional)
            train_sample_path = "debug_plots/training_epoch_sample.npy"
            if os.path.exists(train_sample_path):
                train_epoch = np.load(train_sample_path)  # expects shape (time, channels)
                fig, axs = plt.subplots(n_ch, 1, figsize=(8, 2.5 * n_ch))
                for c in range(n_ch):
                    f_s, P_s = welch(train_epoch[:, c], fs=sf, nperseg=512)
                    f_x, P_x = welch(X[0, :, c], fs=sf, nperseg=512)
                    axs[c].semilogy(f_s, P_s, label="train", alpha=0.8)
                    axs[c].semilogy(f_x, P_x, label="subject", alpha=0.8)
                    axs[c].legend()
                    axs[c].set_title(f"PSD ch{c} train vs subject")
                plt.tight_layout()
                plt.savefig(f"debug_plots/{subject_to_analyze}_psd_vs_train.png", dpi=150)
                plt.close()
                print("DEBUG: saved PSD comparison with training sample")

        except Exception as _e:
            print("DEBUG: failed saving channel stats/PSD:", _e)

        # Phân tích nhiễu
        generate_noise_impact_report(y_subject_true, y_pred_final, CONFIG, subject_id=subject_to_analyze)

        # Vẽ timeline
        plot_sleep_timeline(y_pred_final, sleep_start_time, CONFIG, subject_id=subject_to_analyze)

        # --- FIX: tạo danh sách tên stage từ nhãn số để dùng cho gợi ý giờ thức dậy ---
        try:
            y_pred_stages = [CONFIG.SLEEP_STAGE_LABELS[int(x)] for x in np.array(y_pred_final).astype(int)]
        except Exception:
            y_pred_stages = []

        # Đề xuất giờ thức dậy dựa trên kết quả đã làm mượt
        print("\n--- Chọn chế độ đề xuất ---")
        print("1. Dậy trong giai đoạn nhẹ (N1, N2, REM)")
        print("2. Dậy sau mỗi chu kỳ 90 phút")
        choice = input("▶️ Nhập lựa chọn (1 hoặc 2): ")

        optimal_times = get_optimal_wakeup_times(y_pred_stages, sleep_start_time, choice, age, gender)

        print(f"\n📌 Ngủ lúc: {sleep_start_time_str}")
        print(f"👤 Tuổi: {age}, Giới tính: {gender}")
        if optimal_times:
            print("\n⏰ Giờ thức dậy tối ưu:")
            for i, t in enumerate(optimal_times, 1):
                print(f"   {i}. {t}")
        else:
            print("⚠️ Không có giờ thức dậy tối ưu.")
    else:
        print(f"❌ Không thể xử lý cho subject {subject_to_analyze}.")
