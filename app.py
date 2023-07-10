from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

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

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        
