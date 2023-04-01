# -*- coding: utf-8 -*-
"""
@author: Rameswar
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json


app = FastAPI()

class model_input(BaseModel):
    
    age : int
    sex : int
    cp : int
    trestbps : int
    chol : int
    fbs : float
    restecg : float
    thalach : int
    exang : int
    oldpeak : int
    slope : int
    ca : int
    thal : int       
        
# loading the saved model
classifier_DT = pickle.load(open('classifier_DT.sav', 'rb'))

@app.post('/diabetes_prediction')
def diabetes_predd(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    age = input_dictionary['age']
    sex = input_dictionary['sex']
    cp = input_dictionary['cp']
    chol = input_dictionary['chol']
    fbs = input_dictionary['fbs']
    restecg = input_dictionary['restecg']
    thalach = input_dictionary['thalach']
    exang = input_dictionary['exang']
    oldpeak = input_dictionary['oldpeak']
    slope = input_dictionary['slope']
    ca = input_dictionary['ca']
    thal = input_dictionary['thal']
    
    
    input_list = [age, sex, cp, chol, fbs, restecg, thalach, exang,oldpeak,slope,ca,thal]
    
    prediction = classifier_DT.predict_proba([input_list])
    
    if (prediction[0][0] >= 0.75):
        return 2
    elif (prediction[0][0] < 0.75 and prediction[0][0] >= 0.25):
        return 1
    else:
        return 0

    



