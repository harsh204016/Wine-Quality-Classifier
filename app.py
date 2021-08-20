from pycaret.classification import load_model, predict_model
import streamlit as st
import numpy as np
import pandas as pd


def predict_quality(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    return predictions_data['Label'][0]
    

model = load_model('extra_tree_model')


st.title('Wine Quality Classifier Web App')
st.write('This is a web app to classify the quality of your wine based on\
         several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction of the classifier.')
         
         
        
alcohol = st.sidebar.slider(label = 'Alcohol', min_value = 10.0,
                          max_value = 16.0 ,
                          value = 13.0,
                          step = 0.5)

malic_acid = st.sidebar.slider(label = 'Malic Acid', min_value = 0.6,
                          max_value = 5.00 ,
                          value = 2.33,
                          step = 0.2)
                          
ash = st.sidebar.slider(label = 'Ash', min_value = 1.25,
                          max_value = 3.50 ,
                          value = 2.36,
                          step = 0.1)                          

alcalinity_of_ash = st.sidebar.slider(label = 'Alcalinity of Ash', min_value = 10.0,
                          max_value = 30.0 ,
                          value = 19.5,
                          step = 1.0)

magnesium = st.sidebar.slider(label = 'Magnesium', min_value = 70.000,
                          max_value = 170.000 ,
                          value = 98.0,
                          step = 5.0)
   
total_phenols = st.sidebar.slider(label = 'Total Phenols', min_value = 0.95,
                          max_value = 4.0,
                          value = 2.35,
                          step = 0.2)

flavanoids = st.sidebar.slider(label = 'Flavanoids', min_value = 0.3,
                          max_value = 5.50 ,
                          value = 2.15,
                          step = 0.5)

nonflavanoid_phenols = st.sidebar.slider(label = 'Non Flavanoid Phenols', min_value = 0.1,
                          max_value = 0.7 ,
                          value = 0.35,
                          step = 0.09)

proanthocyanins = st.sidebar.slider(label = 'Proanthocyanins', min_value = 0.4,
                          max_value = 4.00 ,
                          value = 1.55,
                          step = 0.2)
                          
color_intensity = st.sidebar.slider(label = 'Color Intensity', min_value = 1.2,
                          max_value = 13.00,
                          value = 4.69,
                          step = 0.5)

hue = st.sidebar.slider(label = 'Hue', min_value = 0.40,
                          max_value = 1.75,
                          value = 0.96,
                          step = 0.2)
                          
                          

features = {'alcohol':alcohol, 'malic_acid':malic_acid, 'ash':ash, 'alcalinity_of_ash':alcalinity_of_ash,
            'magnesium':magnesium,'total_phenols':total_phenols, 'flavanoids':flavanoids,
            'nonflavanoid_phenols':nonflavanoid_phenols,'proanthocyanins':proanthocyanins, 
            'color_intensity':color_intensity, 'hue':hue
           }
 

features_df  = pd.DataFrame([features])

print(features_df)

st.table(features_df)  

if st.button('Predict'):
    
    prediction = predict_quality(model, features_df)
    
    st.write(' Based on feature values, your wine quality is '+ str(prediction))