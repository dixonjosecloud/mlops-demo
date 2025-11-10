import os, pathlib

# Always use /tmp for MLflow tracking on GitHub Actions
mlruns_dir = "/tmp/mlruns"
os.environ["MLFLOW_TRACKING_URI"] = f"file:{mlruns_dir}"
pathlib.Path(mlruns_dir).mkdir(parents=True, exist_ok=True)

# --- now normal imports ---
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# force MLflow to write inside the repo
mlflow.set_tracking_uri(f"file:{safe_home}/mlruns")
mlflow.set_experiment("diabetes_experiment")

with mlflow.start_run():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print(f"✅ Run logged with MSE: {mse:.2f}")

