# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://127.0.0.1:5500/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pickle_in = open("model_saved.pkl","rb")
classifier=pickle.load(pickle_in)


s3 = np.array(['abdominal_pain', 'altered_sensorium', 'anxiety', 'blackheads', 'blister', 'blurred_and_distorted_vision', 'breathlessness', 'bruising', 'burning_micturition', 'chest_pain', 'chills', 'cold_hands_and_feets', 'continuous_feel_of_urine', 'cough', 'dark_urine', 'dehydration', 'diarrhoea', 'dischromic _patches', 'dizziness', 'extra_marital_contacts', 'fatigue', 'foul_smell_of urine', 'headache', 'high_fever', 'hip_joint_pain', 'joint_pain', 'knee_pain', 'lethargy', 'loss_of_appetite', 'mood_swings', 'movement_stiffness', 'nausea', 'neck_pain', 'nodal_skin_eruptions', 'obesity', 'red_sore_around_nose', 'restlessness', 'scurring', 'silver_like_dusting', 'skin_peeling', 'spinning_movements', 'stomach_pain', 'sweating', 'swelling_joints', 'swelling_of_stomach', 'ulcers_on_tongue', 'vomiting', 'watering_from_eyes', 'weakness_of_one_body_side', 'weight_loss', 'yellowish_skin'])
s2 = np.array(['abdominal_pain', 'acidity', 'anxiety', 'blackheads', 'bladder_discomfort', 'blister', 'breathlessness', 'bruising', 'chills', 'cold_hands_and_feets', 'cough', 'cramps', 'dehydration', 'fatigue', 'foul_smell_of urine', 'headache', 'high_fever', 'indigestion', 'joint_pain', 'knee_pain', 'lethargy', 'loss_of_appetite', 'mood_swings', 'nausea', 'neck_pain', 'nodal_skin_eruptions', 'patches_in_throat', 'pus_filled_pimples', 'shivering', 'skin_peeling', 'skin_rash', 'stiff_neck', 'stomach_pain', 'sunken_eyes', 'sweating', 'swelling_joints', 'ulcers_on_tongue', 'vomiting', 'weakness_in_limbs', 'weakness_of_one_body_side', 'weight_gain', 'weight_loss', 'yellowish_skin'])
s1 = np.array(['acidity', 'back_pain', 'bladder_discomfort', 'breathlessness', 'burning_micturition', 'chills', 'continuous_sneezing', 'cough', 'cramps', 'fatigue', 'headache', 'high_fever', 'indigestion', 'itching', 'joint_pain', 'mood_swings', 'muscle_wasting', 'muscle_weakness', 'neck_pain', 'patches_in_throat', 'pus_filled_pimples', 'shivering', 'skin_rash', 'stiff_neck', 'stomach_pain', 'sunken_eyes', 'vomiting', 'weakness_in_limbs', 'weight_gain', 'yellowish_skin'])












# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Krish Youtube Channel': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_banknote(data:BankNote):
    print(data)
    data = data.dict()
    variance=data['symptom1']
    skewness=data['symptom2']
    curtosis=data['symptom3']
    prediction = classifier.predict([[variance,skewness,curtosis]])
    print(prediction[0])
    return {'prediction': str(prediction[0])}

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload