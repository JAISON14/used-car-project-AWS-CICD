import pandas as pd
import numpy as np
import pandas_profiling as pp
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

from sklearn import metrics, preprocessing
#from keras.models import Model
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

#
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object,load_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def preliminary_data_cleaning(self,data):
        '''
        This function is responsible for preliminary data cleaning.
        
        '''
        try:
            logging.info("Inside preliminary_data_cleaning")
            # Dropping the id and vin columns
            data = data.drop('id', axis=1)
            data = data.drop('vin', axis=1)
            data = data.drop('stock_no', axis=1)
            
            # Remove duplicate rows
            data = data.drop_duplicates()
            
            # Removing rows with price missing since that is our target column.
            data = data.dropna(subset=['price'])
            # Reset the index of the DataFrame
            data = data.reset_index(drop=True)
            
            return data
        
        except Exception as e:
            raise CustomException(e,sys)
    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            pass

            return 
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def impute_numerical_features(self,data):
        '''
        This function is responsible for imputing numerical features
        
        '''
        try:
                # Select the numerical features in the DataFrame
                numerical_features = data.select_dtypes(include='number')
                
                #Create a KNN imputer instance
                imputer = KNNImputer(n_neighbors=5)

                # Fit the imputer on the numerical features
                imputer.fit(numerical_features)

                #Apply the imputer to the numerical features
                imputed_data = imputer.transform(numerical_features)

                # Convert the imputed data back to a DataFrame
                imputed_df = pd.DataFrame(imputed_data, columns=numerical_features.columns)

                # Replace the original numerical features in the data with the imputed values
                data[numerical_features.columns] = imputed_df
                
                # Save the imputer to a pickle file
                save_object(

                        file_path=os.path.join('artifacts',"knn_imputer.pkl"),
                        obj=imputer

                )

    
                return data
        
        except Exception as e:
            raise CustomException(e,sys)

    def impute_numerical_features_test(self,data):
        '''
        This function is responsible for imputing numerical features
        
        '''
        try:
                # Select the numerical features in the DataFrame
                numerical_features = data.select_dtypes(include='number')
                
                # Load the KNNImputer object from joblib file
                imputer = load_object(file_path=os.path.join('artifacts',"knn_imputer.pkl"))

                #Apply the imputer to the numerical features
                imputed_data = imputer.transform(numerical_features)

                # Convert the imputed data back to a DataFrame
                imputed_df = pd.DataFrame(imputed_data, columns=numerical_features.columns)

                # Replace the original numerical features in the data with the imputed values
                data[numerical_features.columns] = imputed_df

    
                return data
        
        except Exception as e:
            raise CustomException(e,sys)


    def remove_NaN_from_categorical_features(self,data):
        '''
        This function is responsible for removing NaN values from categorical features
        
        '''
        try:
            categorical_features = data.select_dtypes(include='object').columns.tolist()
            # Replacing NaN values by "NONE"
            for feature in categorical_features:
                data[feature] = data[feature].fillna("NONE")
            
            return data
        
        except Exception as e:
            raise CustomException(e,sys)

    def create_new_features(self,data):
        try:
            # Calculate the 'miles_per_year' feature
            data['miles_per_year'] = (data['miles'] / (2023 - data['year'])).round(0)
            # Create the 'model_trim' column
            data['model_trim'] = data['model'] + ' ' + data['trim']
            data['region'] = data['zip'].str[:3]
            data['age'] = (2023 - data['year'])
            return data
        except Exception as e:
            raise CustomException(e,sys)
        
    def log_transform(self,df, features):
        try:
            # Create a copy of the DataFrame to avoid modifying the original data
            transformed_df = df.copy()
            
            # Perform logarithmic transformation on the specified features
            for feature in features:
                transformed_df[feature] = np.log(transformed_df[feature])
            df[features] = transformed_df[features]
            
            return df
        except Exception as e:
            raise CustomException(e,sys)

    def scaling_features(self,data,y_train):
        try:    
            numerical_features = data.select_dtypes(include=np.number).columns.tolist()

            # Remove rows containing infinity or extremely large values
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.dropna(subset=numerical_features, how='any')
            
            y_train = y_train.loc[data.index]

            # Scale the numerical features
            scaler = StandardScaler()
            scaler.fit(data[numerical_features])
                       
            # Save the scaler to a pickle file
            save_object(

                        file_path=os.path.join('artifacts',"scaler.pkl"),
                        obj=scaler

                )
            
            scaled_data = scaler.transform(data[numerical_features])
                
            # Create a new DataFrame with the scaled numerical features
            scaled_df = pd.DataFrame(scaled_data, columns=numerical_features)
            data[numerical_features] = scaled_df[numerical_features]
            
            return scaled_df, y_train
        except Exception as e:
            raise CustomException(e,sys)

    def scaling_features_test(self,data):
        try:    
            numerical_features = data.select_dtypes(include=np.number).columns.tolist()

            # Load the encoder from the pickle file
            scaler = load_object(file_path=os.path.join('artifacts',"scaler.pkl"))

            scaled_data = scaler.transform(data[numerical_features])
                
            # Create a new DataFrame with the scaled numerical features
            scaled_df = pd.DataFrame(scaled_data, columns=numerical_features)
            data[numerical_features] = scaled_df[numerical_features]
            
            return scaled_df
        except Exception as e:
            raise CustomException(e,sys)

    def target_encode_regression(self,X_train, y_train):
        try:
            # Initialize a TargetEncoder object
            encoder = ce.TargetEncoder(cols=X_train.select_dtypes(include='object').columns)
            
            # Fit the encoder on the training data
            encoder.fit(X_train, y_train)
            
            with open('encoder.pkl', 'wb') as file:
                pickle.dump(encoder, file)
            joblib.dump(encoder, 'encoder.joblib')

            save_object(

                        file_path=os.path.join('artifacts',"encoder.pkl"),
                        obj=encoder

                )
            

            # Transform the categorical features in the training data
            X_train_encoded = encoder.transform(X_train)
            
            return X_train_encoded
        except Exception as e:
            raise CustomException(e,sys)

    def target_encode_regression_test(self,X_train):
        try:
            
            # Load the encoder from the pickle file
            encoder = load_object(file_path=os.path.join('artifacts',"encoder.pkl"))
            # Transform the categorical features in the training data
            X_train_encoded = encoder.transform(X_train)
            
            return X_train_encoded
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Starting preliminary_data_cleaning")

            train_df=self.preliminary_data_cleaning(train_df)

            logging.info(" preliminary_data_cleaning done")

            X_train =train_df.drop('price', axis=1)
            y_train = train_df['price']
            X_train=self.impute_numerical_features(X_train)
            logging.info("Numerical features imputation completed")
            X_train=self.remove_NaN_from_categorical_features(X_train)
            logging.info("NaN removed from categorical features")

            logging.info(" Starting Feature Engineering")
            X_train = self.create_new_features(X_train)
            columns_to_transform = ['miles','miles_per_year']
            logging.info("New features created")
            X_train =self.log_transform(X_train,columns_to_transform) 
            
            columns_to_remove = ['street','seller_name','year','zip','trim']
            # Remove columns from the train_df DataFrame
            X_train = X_train.drop(columns=columns_to_remove)
            categorical_features = X_train.select_dtypes(include='object').columns.tolist()
            X_train_encoded = self.target_encode_regression(X_train, y_train)
            logging.info("Target Encoding completed")
            X_train, y_train = self.scaling_features(X_train_encoded, y_train)

            
            
            save_object(

                file_path=os.path.join('artifacts',"X_train.pkl"),
                obj=X_train

                        )
            save_object(

                file_path=os.path.join('artifacts',"y_train.pkl"),
                obj=y_train

            )

            

            logging.info(" Preparing Testing Data")

            # Data Cleaning
            test_df=self.preliminary_data_cleaning(test_df)
            X_test =test_df.drop('price', axis=1)
            y_test = test_df['price']
            X_test=self.impute_numerical_features_test(X_test)
            X_test=self.remove_NaN_from_categorical_features(X_test)

            logging.info(" Test data cleaning completed")

            # Feature Enggineering
            X_test = self.create_new_features(X_test)
            columns_to_transform = ['miles','miles_per_year']
            X_test =self.log_transform(X_test,columns_to_transform) 

            columns_to_remove = ['street','seller_name','year','zip','trim']
            # Remove columns from the train_df DataFrame
            X_test = X_test.drop(columns=columns_to_remove)

            categorical_features = X_test.select_dtypes(include='object').columns.tolist()
            X_test = self.target_encode_regression_test(X_test)

            # Remove rows containing infinity or extremely large values
            numerical_features = X_test.select_dtypes(include=np.number).columns.tolist()
            X_test = X_test.replace([np.inf, -np.inf], np.nan)
            X_test = X_test.dropna(subset=numerical_features, how='any')          
            y_test = y_test.loc[X_test.index]

            X_test = self.scaling_features_test(X_test)
            logging.info("Read train and test data completed")

            save_object(

                file_path=os.path.join('artifacts',"X_test.pkl"),
                obj=X_test

            )
            save_object(

                file_path=os.path.join('artifacts',"y_test.pkl"),
                obj=y_test

            )  
            


            return (
                X_train,
                y_train,
                X_test,
                y_test
            )
                
            
        except Exception as e:
            raise CustomException(e,sys)