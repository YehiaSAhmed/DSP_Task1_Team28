import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px
import plotly.graph_objs as go

st.title("Signal Viewer")

uploaded_file = st.file_uploader("Choose a signal to plot", type=["csv"])
if uploaded_file is not None: 
    dataframe = pd.read_csv(uploaded_file)
    # st.write(dataframe)
    t=dataframe.iloc[:,0]
    y=dataframe.iloc[:,1]
    print(len(t),len(y))
    fig = px.line(data_frame= dataframe,x=t,y=y)
    st.plotly_chart(fig)
