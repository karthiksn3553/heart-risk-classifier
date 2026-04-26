import pickle
import streamlit as st
import pandas as pd
import numpy as np


# load the model
model = pickle.load(open('rf_model.pkl','rb'))

# title for app
st.title("Heart Attack Risk Classification App ❤️")


# create input features

age = st.number_input('Age',min_value=20,max_value=100,value=25)
restingbp = st.number_input('RestingBP',min_value=0 , max_value =300,value=100)
cholesterol = st.number_input('Cholesterol',min_value=0 , max_value =700,value=140)
fastingbs = st.selectbox('FastingBS',(0,1))
maxhr = st.number_input('MaxHR',min_value=60 , max_value =250,value=140)
oldpeak = st.number_input('Oldpeak',min_value=-3.0 , max_value =6.6,value=1.0)
gender = st.selectbox('Gender(Male or Female)',('M','F'))
chestpaintype = st.selectbox('ChestPainType',('ATA', 'NAP' ,'ASY' ,'TA'))
restingecg = st.selectbox('RestingECG',('Normal' ,'ST' ,'LVH'))
exerciseangina = st.selectbox('ExerciseAngina',('N' ,'Y'))
st_slope = st.selectbox('ST_Slope',('Up', 'Flat', 'Down'))

# Encoding logic
# The model expects a single 'Sex' column, not two! (Standard encoding: M=1, F=0)
Sex = 1 if gender == 'M' else 0
ExerciseAngina = 1 if exerciseangina == 'Y' else 0

ChestPainType_dict = {'ASY': 3, 'NAP': 2, 'ATA': 1, 'TA': 0}
ChestPainType = ChestPainType_dict[chestpaintype]

RestingECG_dict = {'Normal': 0, 'LVH': 1, 'ST': 2}
RestingECG = RestingECG_dict[restingecg]

ST_Slope_dict = {'Down': 0, 'Up': 1, 'Flat': 2}
ST_Slope = ST_Slope_dict[st_slope]

# Create dataframe with EXACT spelling expected by the model
input_features = pd.DataFrame({
    'Age': [age],
    'Sex': [Sex],
    'ChestPainType': [ChestPainType],
    'RestingBP': [restingbp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fastingbs],
    'RestingECG': [RestingECG],
    'MaxHR': [maxhr],
    'ExerciseAngina': [ExerciseAngina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [ST_Slope]
})

# scaling
scaler = StandardScaler()
input_features[['Age','RestingBP','Cholesterol','MaxHR']]=scaler.fit_transform(input_features[['Age','RestingBP','Cholesterol','MaxHR']])

# predictions
if st.button('Predict'):
  predictions= model.predict(input_features)[0]
  if predictions==1:
    st.error('⚠️High Risk of Heart attack❗')
  else:
    st.success('Low risk of Heart attack😎😊')  
