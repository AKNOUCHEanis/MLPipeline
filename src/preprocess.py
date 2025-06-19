import pandas as pd
import sys
import os
import yaml

def preprocess(input_path, output_path):
    data = pd.read_csv(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, header=None, index=False)
    print(f"Preprocessed data saved into: {output_path}")


if __name__=='__main__':

    # load params
    params = yaml.safe_load(open('params.yaml'))['preprocess']
    preprocess(params['input'], params['output'])

    