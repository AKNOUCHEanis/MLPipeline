import pandas as pd
import sys
import os
import yaml
from sklearn.model_selection import train_test_split

def preprocess(input_path, output_path):
    data = pd.read_csv(input_path)

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    x_train.to_csv(f"{output_path}/x_train.csv", header=None, index=False)
    y_train.to_csv(f"{output_path}/y_train.csv", header=None, index=False)
    x_test.to_csv(f"{output_path}/x_test.csv", header=None, index=False)
    y_test.to_csv(f"{output_path}/y_test.csv", header=None, index=False)

    print(f"Preprocessed data saved into: {output_path}")


if __name__=='__main__':

    # load params
    params = yaml.safe_load(open('params.yaml'))['preprocess']
    preprocess(params['input'], params['output'])

    