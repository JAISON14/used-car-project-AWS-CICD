import os
import sys
from dataclasses import dataclass
from sklearn.metrics import r2_score

from sklearn.ensemble import BaggingRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,X_train,y_train,X_test,y_test):
        try:
            
            # Define the models to evaluate
            models = {
                '''"Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),'''
                "Bagging Regressor": BaggingRegressor()
            }

            # Evaluate the models
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)


            # Get the best model score from the report
            best_model_score = max(sorted(model_report.values()))

            # Get the best model name from the report
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions using the best model
            predicted=best_model.predict(X_test)

            # Calculate the R-squared score
            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        except Exception as e:
            # Raise a custom exception with the original exception and sys information
            raise CustomException(e,sys)      