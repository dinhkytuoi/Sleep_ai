# create_dummy_model.py
import tensorflow as tf, os
out = r"D:\AlarmApp\NCKH\server\models\best_model_fold_1.keras"
print("Saving dummy model to:", out)
# model very small
model = tf.keras.Sequential([
    tf.keras.Input(shape=(100,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
# save as .keras (zip)
model.save(out, include_optimizer=False)
print("Saved.")
