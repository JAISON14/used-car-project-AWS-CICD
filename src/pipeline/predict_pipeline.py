import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
from src.logger import logging

from src.components.data_transformation import DataTransformation

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
            try:
                model_path=os.path.join("artifacts","model.pkl")

                logging.info("Before Loading")
                model=load_object(file_path=model_path)
                logging.info("Model Loaded")
                data_transformation=DataTransformation()
                logging.info("Data cleaning started")
                features =data_transformation.impute_numerical_features_test(features)
                features = data_transformation.remove_NaN_from_categorical_features(features)
                logging.info("Feature Engineering started")
                # Feature Enggineering
                features = data_transformation.create_new_features(features)
                columns_to_transform = ['miles','miles_per_year']
                features =data_transformation.log_transform(features,columns_to_transform) 
                logging.info("Feature selection started")
                columns_to_remove = ['street','seller_name','year','zip','trim']
                # Remove columns from the train_df DataFrame
                features = features.drop(columns=columns_to_remove)
                logging.info("Feature encoding started")
                categorical_features = features.select_dtypes(include='object').columns.tolist()
                features = data_transformation.target_encode_regression_test(features)
                logging.info("Feature scaling started")
                features = data_transformation.scaling_features_test(features)
                logging.info("Predicting price")
                preds=model.predict(features)
                return preds


            
            except Exception as e:
                raise CustomException(e,sys)

class CustomData:
    def __init__(  self,
        miles:           float,
        year:            float,
        make:             str,
        model:            str,
        trim:             str,
        body_type:        str,
        vehicle_type:     str,
        drivetrain:       str,
        transmission:     str,
        fuel_type:        str,
        engine_size:     float,
        engine_block:     str,
        seller_name:      str,
        street:           str,
        city:             str,
        state:            str,
        zip:              str):

        self.miles = miles

        self.year = year

        self.make = make

        self.model = model

        self.trim = trim

        self.body_type = body_type

        self.vehicle_type = vehicle_type

        self.drivetrain = drivetrain

        self.transmission = transmission

        self.fuel_type = fuel_type

        self.engine_size = engine_size

        self.engine_block = engine_block

        self.seller_name = seller_name

        self.street = street

        self.city = city

        self.state = state

        self.zip = zip

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "miles" : [self.miles],  

                "year" : [self.year],

                "make" : [self.make], 

                "model" : [self.model], 

                "trim" : [self.trim], 

                "body_type" : [self.body_type],

                "vehicle_type" : [self.vehicle_type],

                "drivetrain" : [self.drivetrain],

                "transmission" : [self.transmission],

                "fuel_type" : [self.fuel_type],

                "engine_size" : [self.engine_size],

                "engine_block" : [self.engine_block],

                "seller_name" : [self.seller_name],

                "street" : [self.street], 

                "city" : [self.city], 

                "state" : [self.state],

                "zip" : [self.zip],

            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)