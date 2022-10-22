import streamlit as st 
import math
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#add signal -  remove signal - reconstruction - sampling - clean code  

import streamlit as st

st.markdown("<h1 style='text-align: center;'>Signal Studio</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Team 28</h2>", unsafe_allow_html=True)

st.subheader("Frequency for Signal 1")
x = st.slider('A number between 0-100',value=25,min_value=1,max_value=100)
st.write("slider number : ",x)






uploaded_file = st.file_uploader("Please select a csv file", type=["csv"])
if uploaded_file is not None:
    file= pd.read_csv(uploaded_file)
    # x_file = file.iloc[0:x,0].values
    y_file = file.iloc[0:x,1].values
    x_file = np.linspace(0,1,len(y_file))

    
    
    
    fig, ax = plt.subplots()
    # ax.set_xlim(0, 1)
    ax.set_xlim(0,1)
    ax.set_ylim(-0.4,0.4)
    ax.set_xlabel('time')
    ax.set_ylabel('s1')
    ax.grid(True)
    ax.plot(x_file,y_file)






else:
    dt = 0.001
    t = np.arange(0, 1, dt)
    s1 = np.sin(2 * np.pi * x * t)

    nse1 = np.random.randn(len(t))                 # white noise 1

    agree = st.checkbox('Add noise')
    st.write(len(t))

    if agree:
        st.subheader("slider 2")
        y = st.slider('A number between 0-10',value=5)
        st.write("slider number : ",y)
        s1 = np.sin(2 * np.pi * x * t) + y * nse1


    st.subheader("sampling slider")
    z = st.slider('A number between 0-100',value=50)
    st.write("slider number : ",z)

    nsample = z
    num = 1000%nsample
    inc = (1000-num)//nsample
    t = np.arange(0, 1, 1/(1000-num))
    # t = np.linspace(0, 1, 1000-num)

    s1 = np.sin(2 * np.pi * x * t)
    s2 = np.sin(2 * np.pi * (z/2) * t)

    
    st.write("nquist rate : ",2*x)



    sample=[]
    time = []
    sin = []

    for i in range(0,1000-num,inc):
        time.append(t[i])
        sample.append(s1[i])
        sin.append(i)
        
    newtime = np.array([])
    newpoints = np.array([])

    iter= (1000-num) // z

    for i in range(len(time)-1):
        newtime = np.concatenate((newtime,np.linspace(time[i],time[i+1],100)))






    st.write(len(newtime))
    st.write(newtime)
        



    sample = np.array(sample)
    time = np.array(time)


    timemod = np.arange(0,1,dt)

    # sj = np.sin(2*np.pi*(z//2)*time)

    # s2 = np.sin(2 * np.pi * x * time) 

    # f2 = np.sin(2 * np.pi * x * time)
    # mymodel = np.poly1d(np.polyfit(time, sample, 3))
    # myline = np.linspace(1, 22, 100)

    # mymodel = np.poly1d(np.polyfit(time,sample, 3))
    # myline = np.linspace(0, 1, 1000)


    #x,y 10 point 





    # arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)

    ax.set_xlabel('time')
    ax.set_ylabel('s1')
    ax.grid(True)
    ax.plot(t,s1)
    ax.plot(time,sample,'o')
    ax.plot(t,s2)
    # ax.plot(newtime,s2)
    # ax.plot(myline, mymodel(myline))

    # ax.plot(time,sample)

st.pyplot(fig)



