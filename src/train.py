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
from utils import set_mlflow_tracking_config


def hyperparameter_tuning(x_train, y_train, param_grid):

    rfc = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(x_train, y_train)
    return grid_search


def train(x_train_path, y_train_path, model_path):
    
    x = pd.read_csv(x_train_path)
    y = pd.read_csv(y_train_path)

    print(x.shape)
    print(y.shape)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    x_train = x_train.values
    y_train = y_train.values
    x_val = x_val.values
    y_val = y_val.values
   
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment("training-experiment")

    with mlflow.start_run():

        signature = infer_signature(x_train, y_train)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        grid_search  = hyperparameter_tuning(x_train, y_train, param_grid)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(x_val)

        accuracy = accuracy_score(y_pred, y_val)
        print(f"Accuracy score: {accuracy}")

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_param('best_n_estimators', grid_search.best_params_['n_estimators'])
        mlflow.log_param('best_max_depth', grid_search.best_params_['max_depth'])
        mlflow.log_param('best_min_samples_split', grid_search.best_params_['min_samples_split'])
        mlflow.log_param('best_min_samples_leaf', grid_search.best_params_['min_samples_leaf'])

        cm = confusion_matrix(y_pred, y_val)
        cr = classification_report(y_pred, y_val)

        mlflow.log_text(str(cm), 'confusion_matrix.txt')
        mlflow.log_text(str(cr), 'classification_report.txt')

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        pickle.dump(best_model, open(model_path, 'wb'))
        print(f"Model saved to: {model_path}")


if __name__ == '__main__':

    # set mlflow-tracking-credentials ls
    set_mlflow_tracking_config()

    # load parameters
    params = yaml.safe_load(open('params.yaml'))['train']

    train_x = params['x_train']
    train_y = params['y_train']
    model_path = params['model']

    train(train_x, train_y, model_path)