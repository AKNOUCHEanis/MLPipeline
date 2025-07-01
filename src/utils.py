import os
from configparser import ConfigParser
from dagshub import init


def set_mlflow_tracking_config():
    config = ConfigParser()
    config.read("config.init")

    os.environ['MLFLOW_TRACKING_URI'] = config['mlflow-tracking-credentials']['uri']
    os.environ['MLFLOW_TRACKING_USERNAME'] = config['mlflow-tracking-credentials']['username']
    os.environ['MLFLOW_TRACKONG_PASSWORD'] = config['mlflow-tracking-credentials']['password']
    os.environ['MLFLOW_TRACKING_REPO_NAME'] = config['mlflow-tracking-credentials']['repo_name']

    os.environ['DAGSHUB_TOKEN'] = config['dagshub-config']['token']

    init(repo_owner=os.environ['MLFLOW_TRACKING_USERNAME'], repo_name=os.environ['MLFLOW_TRACKING_REPO_NAME'], mlflow=True)
