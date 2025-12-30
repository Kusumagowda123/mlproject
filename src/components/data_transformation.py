import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer   #column transformer is used to create pipeline for specific columns
from sklearn.impute import SimpleImputer      #used for handling missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler  #one hot encoder is used for categorical columns and standard scaler is used for numerical columns

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass # decorator to create data classes...imp
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl') # path to save the preprocessor object
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['reading score', 'writing score']
            categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            #numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # handling missing values by median imputation
                ('scaler', StandardScaler())                     # feature scaling 
            ])

            #categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), # handling missing values by most frequent imputation
                ('one_hot_encoder', OneHotEncoder()),                 # one hot encoding for categorical variables
                ('scaler', StandardScaler(with_mean=False))           # feature scaling
            ])

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")  
            #combine both numerical and categorical pipeline using column transformer

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            logging.error("Error occurred in data transformation")
            raise CustomException(e, sys)
        
                                                                            #Data transformation techniques will be implemented in the next steps
    def initiate_data_transformation(self, train_path, test_path):
        try:
            #reading training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessor object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math score'
            numerical_columns = ['reading score', 'writing score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            #transforming using preprocessor object
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            #combining transformed input features and target column
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Data transformation completed")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path, 
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error("Error occurred in initiate_data_transformation")
            raise CustomException(e, sys)
