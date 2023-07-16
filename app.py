import sys
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.exception import CustomException

application=Flask(__name__) 

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        try:
            # Create a CustomData instance with form data
            data = CustomData(
            miles=float(request.form.get('miles')),
            year=float(request.form.get('year')),
            make=request.form.get('make'),
            model=request.form.get('model'),
            trim=request.form.get('trim'),
            body_type=request.form.get('body_type'),
            vehicle_type=request.form.get('vehicle_type'),
            drivetrain=request.form.get('drivetrain'),
            transmission=request.form.get('transmission'),
            fuel_type=request.form.get('fuel_type'),
            engine_size=float(request.form.get('engine_size')),
            engine_block=request.form.get('engine_block'),
            seller_name= "seller_name",
            street= "street",
            city="city",
            state=request.form.get('state'),
            zip= "zip"
        )

            # Get the input data as a pandas DataFrame
            pred_df=data.get_data_as_data_frame()
            print(pred_df)

            # Create a PredictPipeline instance
            predict_pipeline=PredictPipeline()

            # Make predictions using the pipeline
            results=predict_pipeline.predict(pred_df)
            return render_template('home.html',results=results[0])
        except Exception as e:
            raise CustomException(e,sys)
    

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)        
