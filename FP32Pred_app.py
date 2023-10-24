import streamlit as st
import pandas as pd
import seaborn as sns
import altair as alt
import matplotlib as plt
import plotly.express as px
import pickle
# loading the trained model
pickle_in = open('gbFinal.pkl', 'rb')

regressor = pickle.load(pickle_in)

st.title('FP32 Prediction App')

st.write('This app predicts the FP32 of a given input features')
st.write('Please adjust the sidebar to meet your requirements')

st.sidebar.header('User Input Features')

def user_input_features():
    Process_size = st.sidebar.number_input('Process_size (nm)', 0.0, 1000.0,5.0) 
    TDP = st.sidebar.number_input('TDP (W)', 0.0, 1000.0, 320.0) 
    DieSize = st.sidebar.number_input('DieSize (mm^2)', 0.0, 1000000000000.0, 379.0)
    transistor_count = st.sidebar.number_input('transistor_count (million)', 0.0, 10000000000000000.0, 459000.0)
    die_dens = transistor_count / DieSize
    frequency = st.sidebar.number_input('frequency (MHz)', 0.0, 50000.0, 2505.0)
    release_year = st.sidebar.number_input('release year', 0.0, 2100.0, 2023.0)
    transistor_freq = (transistor_count**0.5) * frequency
    real_year = release_year - 2005
    data = {
            'TDP (W)': TDP,
            'Die Size (mm^2)': DieSize,
            'Transistors (million)': transistor_count,
            'Freq (MHz)': frequency,
            'Release Year' : real_year,
            'die_dens': die_dens,
            'transistor_freq': transistor_freq,}
    
    features = pd.DataFrame(data, index=[0])
    return features



df = user_input_features()


prediction = regressor.predict(df)


data=[9.006*1000, 6.29*1000, 6.671*1000, 29.85*1000, prediction[0]]
columns=["RTX2080", "GTX1070", "RTX2060", "RTX3080", "Your Gpu"]

chart_data = pd.DataFrame(
    data=[[9.006, 6.29, 6.671, 29.85, prediction[0]]],
    columns=["RTX2080","GTX1070","RTX2060", "RTX3080", "Your Gpu"])

# chart = alt.Chart(chart_data).mark_bar().encode(
#     x='columns',
#     y='data'
# )

# fig = plt.figure(figsize=(10,5))
# sns.scatterplot(x=columns, y=data)

st.subheader('FP32 Prediction')
st.write(prediction)


fig = px.scatter(x=columns, y=data,
                 size_max=60,size=data,color=columns)
fig.update_xaxes(title_text='GPU Models')
fig.update_yaxes(title_text='Performance (GFLOPS)')

st.plotly_chart(fig, theme="streamlit", use_container_width=True)

