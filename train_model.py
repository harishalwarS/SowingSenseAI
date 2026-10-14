
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("Preparing demonstration data...")

rng = np.random.default_rng(42)
n = 2000

rainfall = rng.uniform(0, 180, n)
temperature = rng.uniform(15, 42, n)
dry_days = rng.integers(0, 15, n)

score = (
    0.4 * np.clip(rainfall / 80 * 100, 0, 100)
    + 0.35 * np.clip(100 - np.abs(temperature - 28) * 8, 0, 100)
    + 0.25 * np.clip(100 - dry_days * 8, 0, 100)
    - 0.25 * np.clip(np.maximum(rainfall - 100, 0), 0, 100)
)

y = (score >= 65).astype(int)

X = pd.DataFrame({
    "rainfall_mm": rainfall,
    "temperature_c": temperature,
    "dry_spell_days": dry_days
})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Demo model training completed!")
print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))

joblib.dump(model, "sowing_model_demo.joblib")
print("Saved model: sowing_model_demo.joblib")
