import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ── Load data ───────────────────────────────────────────────────
df = pd.read_csv("data/real_training_data.csv")
print(f"Loaded dataset: {df.shape}")

# ── Features (X) and Target (y) ─────────────────────────────────
X = df[[
    # Core movement
    "distance_km",                  # how far to next stop
    "speed_kmh",                    # current vehicle speed

    # Time context
    "hour",                         # time of day (rush hour effect)
    "weekday",                      # day of week (Mon vs weekend)

    # Delay & prediction signals
    "delay_seconds",                # already running late?
    "l_predicted_eta_seconds",      # log-level system prediction
    "p_predicted_eta_seconds",      # stop-level system prediction
    "prediction_error",             # how wrong was old model (engineered)

    # Real elapsed time (engineered from timestamps)
    "actual_travel_time_seconds",   # reached_at - created_at
]]

y = df["actual_eta_seconds"]

# ── Train / Test Split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ── Model ───────────────────────────────────────────────────────
model = RandomForestRegressor(
    n_estimators=200,       # more trees = more stable (was 100)
    max_depth=20,           # prevent overfitting
    min_samples_split=10,   # smoother decisions
    n_jobs=-1,              # use all CPU cores
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

# ── Evaluation ──────────────────────────────────────────────────
y_pred = model.predict(X_test)

mae   = mean_absolute_error(y_test, y_pred)
r2    = r2_score(y_test, y_pred)
score = model.score(X_test, y_test)

print("\n── Model Evaluation ──────────────────")
print(f"MAE (avg error in seconds) : {mae:.2f}s")
print(f"R² Score                   : {r2:.4f}")
print(f"Model Score                : {score:.4f}")

# ── Feature Importance ──────────────────────────────────────────
print("\n── Feature Importance ────────────────")
features     = X.columns.tolist()
importances  = model.feature_importances_

for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"{feat:<35s} {bar} {imp:.3f}")

# ── Save Model ──────────────────────────────────────────────────
joblib.dump(model, "models/real_eta_model.pkl")
print("\nModel saved to models/real_eta_model.pkl")