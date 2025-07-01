dvc stage add -f -n preprocess \
-p preprocess.input \
-p preprocess.output \
-d src/preprocess.py \
-d data/raw/data.csv \
-o data/processed/x_train.csv \
-o data/processed/y_train.csv \
-o data/processed/x_test.csv \
-o data/processed/y_test.csv \
python src/preprocess.py


dvc stage add -f -n train \
-p train.x_train \
-p train.y_train \
-p train.model \
-p train.random_state \
-p train.n_estimators \
-p train.max_depth \
-d src/train.py \
-d data/processed/x_train.csv \
-d data/processed/y_train.csv \
-o models/model.pkl \
python src/train.py

dvc stage add -f -n evaluate \
-p evaluate.x_test \
-p evaluate.y_test \
-p evaluate.model \
-d src/evaluate.py \
-d models/model.pkl \
-d data/processed/x_test.csv \
-d data/processed/y_test.csv \
python src/evaluate.py