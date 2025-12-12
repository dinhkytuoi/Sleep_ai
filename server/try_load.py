# try_load.py
import os, traceback
p = r"D:\AlarmApp\NCKH\server\models\best_model_fold_1.keras"
print("Path:", p)
print("os.path.exists:", os.path.exists(p))
if os.path.exists(p):
    print("size (bytes):", os.path.getsize(p))
    print("isfile:", os.path.isfile(p))
    print("models dir listing:", os.listdir(r"D:\AlarmApp\NCKH\server\models"))
try:
    import tensorflow as tf
    print("tensorflow version:", tf.__version__)
    m = tf.keras.models.load_model(p, compile=False)
    print("✅ tf.keras.models.load_model: success")
except Exception as e:
    print("❌ Exception while loading model:")
    traceback.print_exc()
