import dill
import os
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise e

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            param = params[model_name]
            
            # Handle CatBoostRegressor separately due to sklearn compatibility issues
            if isinstance(model, CatBoostRegressor):
                model.set_params(**param)
                model.fit(X_train, y_train, verbose=0)
            else:
                gs = GridSearchCV(model, param, cv=3, n_jobs=-1, verbose=0)
                # Train the model
                gs.fit(X_train, y_train)
                # Get the best model after hyperparameter tuning
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)

            # Calculate r2 score
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise e

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise e    