from src.utils import save_object,load_object
import sys
from src.exception import CustomException
from src.logger import logging
import os

import pandas as pd
import numpy as np

if __name__=="__main__":
    df=pd.read_csv('notebook\data\ca-dealers-used.csv')
    column_data_types = df.dtypes

    df = load_object(file_path=os.path.join('artifacts',"X_train.pkl"))
    column_list = df.columns.tolist()
    print(column_list) 



    