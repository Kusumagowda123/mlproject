# here we are focusing on creating prediction pipeline
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__,template_folder="templates")  #initializing the flask app

app=application  #to make it work with AWS lambda

@app.route('/')
def index():
    return render_template('index.html')  #rendering the home page

@app.route('/predictdata', methods=['GET','POST'])  #defining the predict route to handle POST requests
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])  #rendering the home page with prediction results
       
if __name__=="__main__":
    app.run(host="0.0.0.0")  #running the app on local server