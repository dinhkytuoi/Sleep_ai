# inspect_keras.py
import zipfile, json, os
p = r"D:\AlarmApp\NCKH\server\models\best_model_fold_1.keras"
print("Path:", p)
print("exists:", os.path.exists(p))
print("isfile:", os.path.isfile(p))
print("size:", os.path.getsize(p) if os.path.exists(p) else None)
print("zipfile.is_zipfile:", zipfile.is_zipfile(p))

if zipfile.is_zipfile(p):
    with zipfile.ZipFile(p, 'r') as z:
        names = z.namelist()
        print("Number of entries in zip:", len(names))
        print("First 30 entries:", names[:30])
        for fname in ["keras_metadata.json", "saved_model.pb", "model.h5", "keras_metadata.pb"]:
            if fname in names:
                print(f"Found: {fname}")
                if fname.endswith(".json"):
                    try:
                        print("JSON preview:", json.loads(z.read(fname).decode()) )
                    except Exception as e:
                        print("Cannot parse json:", e)
                break
        else:
            print("None of common metadata files found inside the zip.")
else:
    print("Not a zip file or corrupted archive.")
