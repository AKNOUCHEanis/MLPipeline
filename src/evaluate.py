import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
from urllib.parse import urlparse
import mlflow
from utils import set_mlflow_tracking_config

def evaluate(x_test_path, y_test_path, model_path):

    x_test = pd.read_csv(x_test_path).values
    y_test = pd.read_csv(y_test_path).values

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment("evaluating-experiment")

    with mlflow.start_run():
        model = pickle.load(open(model_path, 'rb'))

        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_metric("accuracy", accuracy)
        print(f"Model accuracy: {accuracy}")


if __name__ == "__main__":

    # set mlflow-tracking-credentials 
    set_mlflow_tracking_config()

    # load parameters
    params = yaml.safe_load(open('params.yaml'))['evaluate']

    x_test_path = params['x_test']
    y_test_path = params['y_test']
    model_path = params['model']

    evaluate(x_test_path, y_test_path, model_path)