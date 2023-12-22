import streamlit as st
import os
import pandas as pd
import joblib as jb
import time

heading_style = '''
<div style="color:red;" align='center'>
<h1>Stress Level Detection</h1>
</div>
'''
def return_df(
		snoring_rate,
		respiration_rate, 
	    body_temperature, 
		limb_movement,
        blood_oxygen, 
	    eye_movement, 
	    sleeping_hours, 
	    heart_rate):
		
    data={
    'snoring_rate':[snoring_rate],
    'respiration_rate':[respiration_rate],
    'body_temperature':[body_temperature],
    'limb_movement':[limb_movement],
    'blood_oxygen':[blood_oxygen],
	'eye_movement':[eye_movement],
	'sleeping_hours':[sleeping_hours],
    'heart_rate':[heart_rate],
    
    }   
    final_df=pd.DataFrame(data)
    return final_df

@st.cache_data()
def base_model():
    bmodel=jb.load(os.path.join('Stress_detection.pkl'))
    return bmodel

st.markdown(heading_style, unsafe_allow_html=True)
snoring_rate=st.number_input('Enter snoring rate',min_value=0)
respiration_rate=st.number_input('Enter respiration rate',min_value=0)
body_temperature=st.number_input('Enter body temperature',min_value=0)
limb_movement=st.number_input('Enter limb movement',min_value=0)
blood_oxygen=st.number_input('Enter blood oxygen level',min_value=0)
eye_movement=st.number_input('Enter eye movement',min_value=0)
sleeping_hours=st.number_input('Enter sleeping hours',min_value=0)
heart_rate=st.number_input('Enter heart rate',min_value=0)

df=return_df(
snoring_rate,
respiration_rate,
body_temperature,
limb_movement,
blood_oxygen,
eye_movement,
sleeping_hours,
heart_rate)
if st.button('Submit'):
	model=base_model()
	preds=model.predict(df)
	predictions=preds[0]
	
	with st.spinner():
		time.sleep(0.5)
		st.balloons()
	if predictions==0:
		st.write('No Stress')
	elif predictions==1:
		st.write('Low Stress')
	elif predictions==2:
		st.write('Moderate Stress')
	elif predictions==3:
		st.write('High Stress')
	elif predictions==4:
		st.write('Extreme Stress')
	else:
         st.warning('Cannot predict',icon='⚠️')
			
	
	
