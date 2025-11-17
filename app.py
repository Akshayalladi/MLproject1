from flask import Flask, render_template, request
from typing import cast
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import PredictPipeline
from src.pipelines.predict_pipeline import Datapreparation

application = Flask(__name__)

app= application


## home page route
@app.route('/')
def index() -> str:
    # render_template may be typed as Optional[str] in some stubs; cast to str for the type-checker
    return cast(str, render_template('index.html'))

@app.route('/predictdata', methods=['GET','POST'])
def predict_data():
    if request.method == 'GET':
        return cast(str, render_template('home.html'))
    else:
        data = Datapreparation(
            gender = request.form.get('gender', ''),
            race_ethnicity = request.form.get('race_ethnicity', ''),
            parental_level_of_education = request.form.get('parental_level_of_education', ''),
            lunch = request.form.get('lunch', ''),
            test_preparation_course = request.form.get('test_preparation_course', ''),
            reading_score = int(request.form.get('reading_score', 0)),
            writing_score = int(request.form.get('writing_score', 0))
        )
    input_df = data.create_dataFrame()
    pred_pipeline = PredictPipeline()
    prediction = pred_pipeline.predict(input_df)
    # Return rendered template (cast to str for type-checkers that mark render_template Optional)
    return cast(str, render_template('home.html', results=prediction[0]))

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True) 
    
