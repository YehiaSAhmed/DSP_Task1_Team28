from multiprocessing.sharedctypes import Value
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import math
from plotly.subplots import make_subplots






Button = False

# css modification
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# session states variables
if 'signal' not in st.session_state:
    st.session_state.signal = np.zeros(1000, dtype=float)
if 'total' not in st.session_state:
    st.session_state.total = np.zeros(1000, dtype=float)
if 'all_signals' not in st.session_state:
    st.session_state.all_signals = []
if "count" not in st.session_state:
    st.session_state.count = 0
if "count1" not in st.session_state:
    st.session_state.count1 =0
if "noise" not in st.session_state:  # to save noise
    st.session_state.noise = np.zeros(1000, dtype=float)
if "check" not in st.session_state:
    st.session_state.check = 0
if "check2" not in st.session_state:
    st.session_state.check2 = 0
if "checklastfile" not in st.session_state:
    st.session_state.checklastfile = ""
if "count2" not in st.session_state:
    st.session_state.count2 = 0
if "check_upload" not in st.session_state:
    st.session_state.check_upload =0
if "check_csv" not in st.session_state:
    st.session_state.check_csv = 0


# title
st.title("Sampling Studio")


def trysampling(t, y, SF):
    i = 0
    sampledt = []
    sampledy = []
    fulltime = float((t[-1]-t[0]))
    #period = 1/MF
    #cycles = int(fulltime/period)
    #pointsincycle = len(t)/cycles
    sampledpoints = SF*float(fulltime)

    step = (round(len(t)/sampledpoints))
    #print(step)
    if sampledpoints > len(t) or step == 0:
        return t, y
    else:
        for i in range(0, len(t), step):
            sampledt.append(t[i])
            sampledy.append(y[i])
        return sampledt, sampledy


def sinc_interp(Ys, Ts, t):
    # if len(nt_array) != len(sampled_amplitude):
    #     raise Exception('x and s must be the same length')
    sampled_amplitude = np.array(Ys)
    sampled_time = np.array(Ts)
    T = (sampled_time[1] - sampled_time[0])
    sincM = np.tile(t, (len(sampled_time), 1)) - \
        np.tile(sampled_time[:, np.newaxis], (1, len(t)))
    yNew = np.dot(sampled_amplitude, np.sinc(sincM/T))
    return yNew


def addsignal(type, amp, freq, time):
    if type == "sin":
        st.session_state.total += amp*np.sin(2*np.pi*freq*time)
        st.session_state.all_signals.append(
            {"type": "sin", "freq": freq, "amp": amp})
    elif type == "cos":
        st.session_state.total += amp*np.cos(2*np.pi*freq*time)
        st.session_state.all_signals.append(
            {"type": "cos", "freq": freq, "amp": amp})


def addnoise(snrratio, time):
    nse1 = 1/snrratio * np.random.rand(len(time), 1)
    for i in range(len(time)):
        st.session_state.total[i] += nse1[i]   # type: ignore
    return nse1


def removenoise(noise):
    for i in range(len(noise)):
        st.session_state.total[i] -= noise[i]   # type: ignore


def removesignal(type, amp, freq, time):
    if type == "sin":
        st.session_state.total -= amp*np.sin(2*np.pi*freq*time)
        st.session_state.all_signals.remove(
            {"type": "sin", "freq": freq, "amp": amp})
    elif type == "cos":
        st.session_state.total -= amp*np.cos(2*np.pi*freq*time)
        st.session_state.all_signals.remove(
            {"type": "cos", "freq": freq, "amp": amp})

def download(time):
    with open("newsignal.csv", "w", newline="") as f:
    # create the csv writer
        writer = csv.writer(f)

    # write a row to the csv file
        writer.writerow(["time", "amp"])
        for i in range(len(time)):
            writer.writerow([time[i], st.session_state.total[i]])
    df = pd.read_csv("newsignal.csv")
    return df

with st.sidebar:
    select = st.selectbox("type of sampling",["csv file","sin wave"])


if select =="csv file":
    st.session_state.count1 = 0
    with st.sidebar:
    # uploaded file
        uploaded_file = st.file_uploader("Choose a file",type="csv")


    if uploaded_file != st.session_state.checklastfile:
        st.session_state.count = 0
    if uploaded_file is not None:
        with st.sidebar:
            maxFrequency=int(st.text_input("Please enter Max Frequency", value="10" ))
            SF = st.slider('sampling frequency', 1, 3*maxFrequency , 2*maxFrequency)
        file = pd.read_csv(uploaded_file)
        # x_file = file.iloc[0:x,0].values
        y = file.iloc[:, 1].values
        t = file.iloc[:, 0].values
        if st.session_state.count == 0:
            # st.session_state.stack.append(y_file)
            st.session_state.count += 1
            st.session_state.total = np.zeros(len(t))
            st.session_state.total += y  # type: ignore
            st.session_state.signal = np.zeros(len(t))
            st.session_state.noise = np.zeros(len(t))
            st.session_state.checklastfile = uploaded_file
            st.session_state.all_signals.clear()

        with st.sidebar:
            snrbutton = st.checkbox("add noise")
            if snrbutton:
                snrratio = st.slider("snr", value=50, min_value=1, max_value=100)
                removenoise(st.session_state.noise)
                st.session_state.noise = addnoise(snrratio, t)
                st.session_state.check_csv = 1
            elif snrbutton == 0:
                removenoise(st.session_state.noise)
                st.session_state.noise = np.zeros(len(t))
                if st.session_state.check_csv == 1:
                    st.session_state.check_csv = 0
                    st.experimental_rerun()
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
                try:
                    if Button and genre == 'sin':
                        addsignal("sin", int(amp), int(freq), t)
                    elif Button and genre == 'cos':
                        addsignal("cos", int(amp), int(freq), t)
            
                except:
                    st.write("invalid input")
                remove_signal = st.selectbox(
                            "remove signal", st.session_state.all_signals)
                removebutton = st.button("remove")
                if removebutton and len(st.session_state.all_signals)!=0:
                    removesignal(
                        remove_signal["type"], remove_signal["amp"], remove_signal["freq"], t)
                    st.experimental_rerun()
                elif removebutton:
                    st.write("no signal to remove")


           
        
        if Button:
            st.session_state.total += st.session_state.signal  # type: ignore
        #fig,axis = plt.subplots()
        
        
        
        #plt.plot(t, st.session_state.total)
        # fig, (ax1, ax2) = plt.subplots(2)
        # fig.suptitle('Signal And Reconstruction')
        # ax1.plot(t,st.session_state.total )
        # ax1.set_title("Signal")
        # ax1.set_xlabel("Time in second")
        
        
        xsampled, ysampled = trysampling(
            t, st.session_state.total, SF)
        print(len(xsampled))
        yreconst = sinc_interp(ysampled, xsampled, t)

        
        
        #plt.stem(xsampled, ysampled, linefmt='yellow', markerfmt='x', bottom=0)
        
        #plt.plot(t,yreconst)
        #axis.set_xlabel("Time in Seconds")
        #axis.set_xlabel("Amplitude in Volts")

        # ax2.stem(xsampled, ysampled, linefmt='yellow', markerfmt='x', bottom=0)
        # ax2.plot(t,yreconst)
        # ax1.set_title("Samples and reconstruction")
        # ax2.set_xlabel("Time in second")
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(y=st.session_state.total,x=t, mode="lines",name="Signal"), row=1, col=1)
        fig.add_trace(go.Scatter(y=ysampled,x=xsampled, mode="markers",name="samples"), row=1, col=1)
        fig.add_trace(go.Scatter(y=yreconst,x=t, mode="lines",name="reconstruction"), row=1, col=1)
        fig.update_xaxes(title_text='Time (in seconds)')
        fig.update_yaxes(title_text='Amplitude (in volts)')
        fig.update_layout(autosize=True)
    

        
    	
        st.plotly_chart(fig, use_container_width=True)
        with st.sidebar:
            st.download_button("download csv file", download(t).to_csv(),
                        file_name='signal.csv', mime='text/csv')


elif select =="sin wave":
    st.session_state.count = 0
    dt = 0.001
    time = np.arange(0, 1, dt)
    if st.session_state.count1 == 0:
        # st.session_state.stack.append(y_file)
        st.session_state.count1 += 1
        st.session_state.total = np.zeros(len(time))
        st.session_state.signal = np.zeros(len(time))
        st.session_state.noise = np.zeros(len(time))
        st.session_state.all_signals.clear()

    # st.session_state.all_signals.clear()
    with st.sidebar:
        
        freqency = st.slider("frequency", min_value=1, max_value=100)
        sampling_frequency = st.slider(
            'sampling frequency', 1, 3*int(freqency), int(2*int(freqency)))


    # st.session_state.total = np.zeros(len(time))
    # st.session_state.signal = np.zeros(len(time))
    sin_signal = np.sin(2 * np.pi * int(freqency) * time)
    if st.session_state.count2 == 0:
        st.session_state.signal = sin_signal
        st.session_state.total += st.session_state.signal
        st.session_state.count2 = 1

    if st.session_state.signal[10] != sin_signal[10]:
        st.session_state.total -= st.session_state.signal
        st.session_state.total += sin_signal
        st.session_state.signal = sin_signal

    with st.sidebar:
        snrbutton = st.checkbox("add noise")
        if snrbutton:
            snrratio = st.slider("snr", value=50, min_value=1, max_value=100)
            removenoise(st.session_state.noise)
            st.session_state.noise = addnoise(snrratio, time)
            st.session_state.check = 1
        elif snrbutton == 0:
            removenoise(st.session_state.noise)
            st.session_state.noise = np.zeros(len(time))
            if st.session_state.check == 1:
                st.session_state.check = 0
                st.experimental_rerun()

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
        try:
            if Button and genre == 'sin':
                addsignal("sin", int(amp), int(freq), time)
            elif Button and genre == 'cos':
                addsignal("cos", int(amp), int(freq), time)
        except:
            st.write("invalid input")
        remove_signal = st.selectbox(
                    "remove signal", st.session_state.all_signals)
        removebutton = st.button("remove")
        
        if removebutton and len(st.session_state.all_signals)!=0:
                # type: ignore
            removesignal(
                    remove_signal["type"], remove_signal["amp"], remove_signal["freq"], time)
            st.experimental_rerun()
        elif removebutton:
            st.write("no signal to remove")


    x_sampled, y_sampled = trysampling(
        time, st.session_state.total,  sampling_frequency+1)
    y_inter = sinc_interp(y_sampled, x_sampled, time)
    # fig = plt.figure()
    # plt.plot(time, st.session_state.total)
    # plt.stem(x_sampled, y_sampled, linefmt='yellow',
    #          markerfmt='x', bottom=0, use_line_collection=True)
    # plt.plot(time,y_inter)
    # plt.xlabel("time")
    # plt.ylabel("signal")
    # st.plotly_chart(fig, use_container_width=True)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(y=st.session_state.total,x=time, mode="lines",name="Signal"), row=1, col=1)
    fig.add_trace(go.Scatter(y=y_sampled,x=x_sampled, mode="markers",name="samples"), row=1, col=1)
    fig.add_trace(go.Scatter(y=y_inter,x=time, mode="lines",name="reconstruction"), row=1, col=1)
    fig.update_xaxes(title_text='Time (in seconds)')
    fig.update_yaxes(title_text='Amplitude (in volts)')
    fig.update_layout(autosize=True)
    st.plotly_chart(fig, use_container_width=True)
    
    
    with st.sidebar:
        st.download_button("download csv file", download(time).to_csv(),
                    file_name='signal.csv', mime='text/csv')
    