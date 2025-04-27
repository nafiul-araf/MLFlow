import mlflow.data as md
import pandas as pd
import mlflow
import mlflow.sklearn as ms
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import dagshub

data = load_diabetes()
X = pd.DataFrame(data=data.data, columns=data.feature_names)
y = pd.Series(data=data.target, name='target')

TEST_SIZE = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

rf = RandomForestRegressor(random_state=42)

params_grid = {
    'n_estimators': [10, 30, 50, 100, 150],
    'max_depth': [None, 3, 5, 10],
    'criterion': ['squared_error', 'friedman_mse']
}

CROSS_VALIDATION_FOLD = 10
grid_search = GridSearchCV(estimator=rf, param_grid=params_grid, cv=CROSS_VALIDATION_FOLD, n_jobs=-1, verbose=2)

# grid_search.fit(X_train, y_train)

# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print(f"Best Paramaters: {best_params}\nBest Score:{best_score}")

dagshub.init(repo_owner='nafiul-araf', repo_name='MLFlow', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/nafiul-araf/MLFlow.mlflow')

mlflow.autolog()
mlflow.set_experiment('diabates_prediction')

with mlflow.start_run(run_name='ht_autolog_2'):
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    mean_sqr_error = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Skip mlflow.log_input because DagsHub MLflow server doesn't support it yet
    
    # train_df = X_train.copy()
    # train_df['target'] = y_train
    # train_df = md.from_pandas(train_df)
    # mlflow.log_input(train_df, 'training_data')

    # test_df = X_test.copy()
    # test_df['target'] = y_test
    # test_df = md.from_pandas(test_df)
    # mlflow.log_input(test_df, 'testing_data')

    mlflow.log_params({'test_size': TEST_SIZE, 'K-Fold': CROSS_VALIDATION_FOLD})
    mlflow.log_metrics({'mean_squared_score': mean_sqr_error, 'R-2': r2})
    mlflow.log_artifact(__file__)
    ms.log_model(best_model, 'random forest model')
    mlflow.set_tags({'Author': 'Nafiul', 'Project': 'Diabates Prediction'})
    
    print(f"Mean Squared Error {mean_sqr_error} and R-2 Score {r2}")