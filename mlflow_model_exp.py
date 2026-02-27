import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Set the tracking URI and experiment name as in the original code
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")
mlflow.set_experiment("MLFLOW Experiment")
mlflow.autolog()

with mlflow.start_run():
    # Load data
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # train the model
    n_estimators = 100
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    
    # Log metrics
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    mlflow.log_metric("rmse", rmse)
    
    # Log the model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print(f"RMSE: {rmse}")
