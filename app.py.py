from itertools import count
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import math



# add signal -  remove signal - reconstruction - sampling - clean code  - ui - plotly

Button = False
# signal = np.array([])

#css modification
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


if 'signal' not in st.session_state:
    st.session_state.signal = np.zeros(1000, dtype=float)
if 'total' not in st.session_state:
    st.session_state.total = np.zeros(1000, dtype=float)
if 'all_signals' not in st.session_state:
    st.session_state.all_signals = []
if "pointsnum" not in st.session_state:
    st.session_state.pointsnum = 0
if "count" not in st.session_state:
    st.session_state.count = 0
if "time" not in st.session_state: #to save time
    st.session_state.time = np.zeros(1000,dtype=float)
if "noise" not in st.session_state:  # to save noise 
    st.session_state.noise = np.zeros(1000,dtype=float)
if "dcount" not in st.session_state:
    st.session_state.dcount = 0
if "check" not in st.session_state:
    st.session_state.check = 0
if "check2" not in st.session_state:
    st.session_state.check2 = 0
if "checklastfile" not in st.session_state:
    st.session_state.checklastfile = ""
# if "stack" not in st.session_state:
#     st.session_state.stack = []

# title
st.title("sampling studio")



def trysampling(t,y,MF,SF): 
        i=0
        sampledt=[]
        sampledy=[]
        fulltime= t[-1]-t[0]
        period=1/MF
        cycles=int(fulltime/period)
        pointsincycle=len(t)/cycles
        sampledpoints=SF*float(fulltime)
        
        step= math.ceil(int(len(t)/sampledpoints))
        if sampledpoints > len(t) or step == 0: 
            return t ,y
        else: 
            for i in range(0,len(t),step):
                sampledt.append(t[i])
                sampledy.append(y[i])
            return sampledt, sampledy

def sinc_interp(Ys,Ts , t):
    # if len(nt_array) != len(sampled_amplitude):
    #     raise Exception('x and s must be the same length')
    sampled_amplitude=np.array(Ys)
    sampled_time=np.array(Ts)
    T = (sampled_time[1] - sampled_time[0])
    sincM = np.tile(t, (len(sampled_time), 1)) - np.tile(sampled_time[:, np.newaxis], (1, len(t)))
    yNew = np.dot(sampled_amplitude, np.sinc(sincM/T))
    return yNew
 

def addsignal(type,amp,freq,time):
    if type == "sin":
        st.session_state.total += amp*np.sin(2*np.pi*freq*time)
        # st.session_state.stack.append(st.session_state.total)
        st.session_state.all_signals.append({"type":"sin","freq":freq,"amp":amp})
    elif type == "cos":
        st.session_state.total += amp*np.cos(2*np.pi*freq*time)
        # st.session_state.stack.append(st.session_state.total)
        st.session_state.all_signals.append({"type":"cos","freq":freq,"amp":amp})



def addnoise(snrratio,time):
    nse1 = 1/snrratio *np.random.rand(len(time),1)
    for i in range(len(time)):
        st.session_state.total[i] += nse1[i]   # type: ignore
    return nse1


def removenoise(noise):
    for i in range(len(noise)):
        st.session_state.total[i] -= noise[i]   # type: ignore

def removesignal(type,amp,freq,time):
    if type == "sin":
        st.session_state.total -= amp*np.sin(2*np.pi*freq*time)
        st.session_state.all_signals.remove({"type":"sin","freq":freq,"amp":amp})
    elif type == "cos":
        st.session_state.total -= amp*np.cos(2*np.pi*freq*time)
        st.session_state.all_signals.remove({"type":"cos","freq":freq,"amp":amp})

count = 0
with st.sidebar:
    # uploaded file
    uploaded_file = st.file_uploader("Choose a file")
    
if uploaded_file != st.session_state.checklastfile:
    st.session_state.count =0


    

# signal slider
    # st.subheader("freq")
    # x = st.slider('A number between 1-1000', value=500,
    #               min_value=1, max_value=1000)
    # st.write("slider number : ", x)
    # st.subheader("sampling slider")
    # z = st.slider('A number between 0-100', value=50)
    # st.write("slider number : ", z)

    # if len(st.session_state.total) != st.session_state.pointsnum:
    #     st.session_state.total = np.zeros(
    #         st.session_state.pointsnum, dtype=float)
    #     st.session_state.signal = np.zeros(
    #         st.session_state.pointsnum, dtype=float)


if uploaded_file is not None:
    with st.sidebar:
        maxfrequency = int(st.text_input("Max Frequency", "10"))
        SF = st.slider('sampling frequency', 1, 3*int(maxfrequency), int(2*int(maxfrequency)))
    file = pd.read_csv(uploaded_file)
    # x_file = file.iloc[0:x,0].values
    y = file.iloc[:, 1].values
    t = file.iloc[:, 0].values
    if st.session_state.count ==0:
        # st.session_state.stack.append(y_file)
        st.session_state.count +=1
        st.session_state.total = np.zeros(len(t))
        st.session_state.total += y #type: ignore
        st.session_state.signal = np.zeros(len(t))
        st.session_state.time = t
        st.session_state.noise = np.zeros(len(t))
        st.session_state.checklastfile = uploaded_file


    st.session_state.pointsnum = len(t)
    with st.sidebar:
        snrbutton = st.checkbox("add noise")
        if snrbutton :
            st.session_state.check = 1
            removenoise(st.session_state.noise)
            snrratio = st.slider("snr",value=10)
            st.session_state.noise = addnoise(snrratio,t)
        elif snrbutton==0:
            if st.session_state.check == 1: 
                removenoise(st.session_state.noise)
            st.session_state.check = 0
            # noise =
        agree = st.checkbox('add signal')
        if agree:
            genre = st.radio(
                "What's signal ?",
                ('sin', 'cos'))
            col1, col2, col3 = st.columns(3)
            with col1:
                amp = st.text_input("amplituide")
            with col2:
                freq = st.text_input("frequency")
            with col3:
                st.text("")
                st.text("")
                Button = st.button("add signal")

            if Button and genre == 'sin':
                addsignal("sin",int(amp),int(freq),t)
            elif Button and genre == 'cos':
                addsignal("cos",int(amp),int(freq),t)
            st.write(st.session_state.all_signals)
            if count == 0:
                remove_signal = st.selectbox("remove signal",st.session_state.all_signals)
            removebutton = st.button("remove")
            if removebutton:
                removesignal(remove_signal["type"],remove_signal["amp"],remove_signal["freq"],t)  # type: ignore
                st.experimental_rerun()


    # fig=plt.figure()
    # plt.plot(t,y)
    # xsampled,ysampled=trysampling(t,y,maxfrequency,SF)
    # print(len(xsampled))
    # #plt.scatter(xsampled,ysampled,color='red', marker='x')
    # plt.stem(xsampled, ysampled, linefmt='yellow', markerfmt='x', bottom=0)
    # st.plotly_chart(fig)
    # #markerline.set_markerfacecolor('red')
    # yreconst= sinc_interp(ysampled, xsampled , t)
    # fig2=plt.figure()
    # plt.plot(t,yreconst)
    # st.plotly_chart(fig2)

    if Button:
        st.session_state.total += st.session_state.signal  # type: ignore
    fig=plt.figure()
    plt.plot(t,st.session_state.total)
    xsampled,ysampled=trysampling(t,st.session_state.total,maxfrequency,SF)
    plt.stem(xsampled, ysampled, linefmt='yellow', markerfmt='x', bottom=0)
    st.plotly_chart(fig)
    yreconst= sinc_interp(ysampled, xsampled , t)
    fig2=plt.figure()
    plt.plot(t,yreconst)
    st.plotly_chart(fig2)


    # fig, ax = plt.subplots()
    # # ax.set_xlim(0, 1)
    # ax.set_xlabel('time')
    # ax.set_ylabel('s1')
    # ax.grid(True)
    # ax.plot(x_file, st.session_state.total)
    # st.pyplot(fig)
    with open("newsignal.csv", "w", newline="") as f:
        # create the csv writer
        writer = csv.writer(f)

    # write a row to the csv file
        writer.writerow(["time", "amp"])
        for i in range(len(t)):
            writer.writerow([t[i], st.session_state.total[i]])
    df = pd.read_csv("newsignal.csv")
    with st.sidebar:
        st.download_button("download csv file", df.to_csv(),
                           file_name='signal.csv', mime='text/csv')


# else:
#     dt = 0.001
#     t = np.arange(0, 1, dt)
#     s1 = np.sin(2 * np.pi * x * t)

#     nse1 = np.random.randn(len(t))                 # white noise 1

#     agree = st.checkbox('Add noise')
#     st.write(len(t))

#     if agree:
#         st.subheader("slider 2")
#         y = st.slider('A number between 0-10',value=5)
#         st.write("slider number : ",y)
#         s1 = np.sin(2 * np.pi * x * t) + y * nse1


#     st.subheader("sampling slider")
#     z = st.slider('A number between 0-100',value=50)
#     st.write("slider number : ",z)

#     nsample = z
#     num = 1000%nsample

#     inc = (1000-num)//nsample
#     t = np.arange(0, 1, 1/(1000-num))
#     # t = np.linspace(0, 1, 1000-num)
#     s1 = np.sin(2 * np.pi * x * t)
#     s2 = np.sin(2 * np.pi * (z/2) * t)
#     samt = np.arange(0,1,1/z)
#     samp = np.sin(2 * np.pi * x * samt)


#     st.write("nquist rate : ",2*x)


#     sample=[]
#     time = []
#     sin = []

#     for i in range(0,1000-num,inc):
#         time.append(t[i])
#         sample.append(s1[i])
#         sin.append(i)

#     newtime = np.array([])
#     newpoints = np.array([])

#     iter= (1000-num) // z

#     for i in range(len(time)-1):
#         newtime = np.concatenate((newtime,np.linspace(time[i],time[i+1],100)))


#     st.write(len(newtime))
#     st.write(newtime)


#     sample = np.array(sample)
#     time = np.array(time)


#     timemod = np.arange(0,1,dt)

#     # sj = np.sin(2*np.pi*(z//2)*time)

#     # s2 = np.sin(2 * np.pi * x * time)

#     # f2 = np.sin(2 * np.pi * x * time)
#     # mymodel = np.poly1d(np.polyfit(time, sample, 3))
#     # myline = np.linspace(1, 22, 100)

#     # mymodel = np.poly1d(np.polyfit(time,sample, 3))
#     # myline = np.linspace(0, 1, 1000)


#     #x,y 10 point

#     # tck = interpolate.splrep(time,sample,s=0)
#     # xfit = np.arange(0,100,np.pi/50)
#     # yfit = interpolate.splev(xfit,tck,der=0)
#     # arr = np.random.normal(1, 1, size=100)
#     fig, ax = plt.subplots()
#     ax.set_xlim(0, 1)

#     ax.set_xlabel('time')
#     ax.set_ylabel('s1')
#     ax.grid(True)
#     ax.plot(t,s1)
#     ax.plot(t,s2)
#     # ax.plot(time,sample,'o')
#     ax.plot(samt,samp,'o')
#     # ax.plot(xfit,yfit)
#     st.pyplot(fig)


#     # ax.plot(newtime,s2)
#     # ax.plot(myline, mymodel(myline))

#     # ax.plot(time,sample)
