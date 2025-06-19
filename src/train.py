import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse
import os
import yaml
import pickle
from configparser import ConfigParser

import dagshub



def set_mlflow_tracking_config():
    config = ConfigParser()
    config.read("config.ini")

    os.environ['MLFLOW_TRACKING_URI'] = config['mlflow-tracking-credentials']['uri']
    os.environ['MLFLOW_TRACKING_USERNAME'] = config['mlflow-tracking-credentials']['username']
    os.environ['MLFLOW_TRACKONG_PASSWORD'] = config['mlflow-tracking-credentials']['password']
    os.environ['MLFLOW_TRACKING_REPO_NAME'] = config['mlflow-tracking-credentials']['repo_name']


def hyperparameter_tuning(x_train, y_train, param_grid):

    rfc = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(x_train, y_train)
    return grid_search


def train(data_path, model_path):
    
    data = pd.read_csv(data_path)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
   
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment("training-experiment")

    with mlflow.start_run():

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        
        signature = infer_signature(x_train, y_train)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        grid_search  = hyperparameter_tuning(x_train, y_train, param_grid)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(x_test)

        accuracy = accuracy_score(y_pred, y_test)
        print(f"Accuracy score: {accuracy}")

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_param('best_n_estimators', grid_search.best_params_['n_estimators'])
        mlflow.log_param('best_max_depth', grid_search.best_params_['max_depth'])
        mlflow.log_param('best_min_samples_split', grid_search.best_params_['min_samples_split'])
        mlflow.log_param('best_min_samples_leaf', grid_search.best_params_['min_samples_leaf'])

        cm = confusion_matrix(y_pred, y_test)
        cr = classification_report(y_pred, y_test)

        mlflow.log_text(str(cm), 'confusion_matrix.txt')
        mlflow.log_text(str(cr), 'classification_report.txt')

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(mlflow.get_tracking_uri())
        print(tracking_url_type_store)

        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(best_model, "model", registered_model_name='Best Model')
        else:
            mlflow.sklearn.log_model(best_model, 'model', signature=signature)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        pickle.dump(best_model, open(model_path, 'wb'))

        print(f"Model saved to: {model_path}")


if __name__ == '__main__':

    # set mlflow-tracking-credentials 
    set_mlflow_tracking_config()

    # load parameters
    params = yaml.safe_load(open('params.yaml'))['train']

    data_path = params['data']
    model_path = params['model']

    train(data_path=data_path, model_path=model_path)









