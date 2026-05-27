import pickle, numpy as np, xgboost as xgb, sklearn

print("numpy:", np.__version__)
print("xgboost:", xgb.__version__)
print("sklearn:", sklearn.__version__)

MODEL_DIR = r"C:/telemetry-producer/models/miami_models"

models = ["winner", "pit", "laptime"]

for name in models:
    # Load existing (numpy 2.x) pickle
    with open(f"{MODEL_DIR}\\{name}_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{MODEL_DIR}\\{name}_feats.pkl", "rb") as f:
        feats = pickle.load(f)

    # Re-save with protocol 4 (numpy 1.x compatible)
    with open(f"{MODEL_DIR}\\{name}_model.pkl", "wb") as f:
        pickle.dump(model, f, protocol=4)
    with open(f"{MODEL_DIR}\\{name}_feats.pkl", "wb") as f:
        pickle.dump(feats, f, protocol=4)

    print(f"saved {name}")