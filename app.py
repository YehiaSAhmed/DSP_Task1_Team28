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


st.set_page_config(layout="wide")


# Button = False

# css modification
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# session states variables

# total amplitude of all signals
if 'total' not in st.session_state:
    st.session_state.total = np.zeros(1000, dtype=float)
    # list of dictionaries for added signals
if 'all_signals' not in st.session_state:
    st.session_state.all_signals = []
# change from sin to csv ( initialize state variables)
if "csv_mode" not in st.session_state:
    st.session_state.csv_mode = 0
# change from csv to sin (initalize sin signal )
if "sin_mode" not in st.session_state:
    st.session_state.sin_mode = 0

if "noise" not in st.session_state:  # to save noise
    st.session_state.noise = np.zeros(1000, dtype=float)

# rerun after removing noise // streamlit bug for checkbox// for sin
if "check" not in st.session_state:
    st.session_state.check = 0
    # check if uploaded file is changed
if "checklastfile" not in st.session_state:
    st.session_state.checklastfile = ""
# # save and reset  original sin signal (@change frequency )
# if "save_reset" not in st.session_state:
#     st.session_state.save_reset = 0

# rerun after removing noise  // streamlit bug for checkbox // for csv
if "check_csv" not in st.session_state:
    st.session_state.check_csv = 0
if "max_freq" not in st.session_state:
    st.session_state.max_freq = 0

# title
st.title("Sampling Studio")


def sampling(t, y, SF):
    i = 0
    sampledt = []
    sampledy = []
    fulltime = float((t[-1]-t[0]))
    #period = 1/MF
    #cycles = int(fulltime/period)
    #pointsincycle = len(t)/cycles
    sampledpoints = SF*fulltime

    step = (round(len(t)/sampledpoints))
    # print(step)
    if sampledpoints > len(t) or step == 0:
        return t, y
    else:
        for i in range(1, len(t), step):
            sampledt.append(t[i])
            sampledy.append(y[i])
        return sampledt, sampledy


def sinc_interp(Ys, Ts, t,SF):
    # if len(nt_array) != len(sampled_amplitude):
    #     raise Exception('x and s must be the same length')
    sampled_amplitude = np.array(Ys)
    sampled_time = np.array(Ts)
    if SF==1:
        T=Ts[0]
        sincM = np.tile(t, (len(sampled_time), 1)) - np.tile(sampled_time[:, np.newaxis], (1, len(t)))
        
        yNew = np.dot(sampled_amplitude, np.sinc(sincM/T))
        return yNew
    T = (sampled_time[1] - sampled_time[0])
    sincM = np.tile(t, (len(sampled_time), 1)) - np.tile(sampled_time[:, np.newaxis], (1, len(t)))
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
    if freq > st.session_state.max_freq:
        st.session_state.max_freq = freq


# def addnoise(snrratio, time):
#     nse1 = 1/snrratio * np.random.rand(len(time), 1)
#     for i in range(len(time)):
#         st.session_state.total[i] += nse1[i]   # type: ignore
#     return nse1

def addnoise(SNR, y):
    # nse1 = 1/snrratio * np.random.rand(len(time), 1)
    # for i in range(len(time)):
    #     st.session_state.total[i] += nse1[i]   # type: ignore
    # return nse1
    signal_power = (np.sum(abs(y)**2))/len(y)
    power_db = 10*np.log10(signal_power)
    noise_db = power_db - SNR
    noisepower = 10**(noise_db/10)
    noise = np.random.normal(0, np.sqrt(noisepower), len(y))
    for i in range(len(y)):
        st.session_state.total[i] += noise[i]
    return noise


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
    select = st.radio("mode",
                      ('sin', 'csv file'),)


if select == "csv file":
    st.session_state.sin_mode = 0
    with st.sidebar:
        # uploaded file
        uploaded_file = st.file_uploader("Choose a file", type="csv")

    if uploaded_file != st.session_state.checklastfile:
        st.session_state.csv_mode = 0
    if uploaded_file is not None:
        with st.sidebar:
            maxFrequency = int(st.text_input(
                "Please enter Max Frequency", value="10"))
            SF = st.slider('sampling frequency', 1, 3 *
                           maxFrequency, 2*maxFrequency)
        file = pd.read_csv(uploaded_file)
        # x_file = file.iloc[0:x,0].values
        y = file.iloc[:, 1].values
        t = file.iloc[:, 0].values
        if st.session_state.csv_mode == 0:
            # st.session_state.stack.append(y_file)
            st.session_state.csv_mode += 1
            st.session_state.total = np.zeros(len(t))
            st.session_state.total += y  # type: ignore
            st.session_state.noise = np.zeros(len(t))
            st.session_state.checklastfile = uploaded_file
            st.session_state.all_signals.clear()

        with st.sidebar:
            snrbutton = st.checkbox("add noise")
            if snrbutton:
                snrratio = st.slider(
                    "snr", value=50, min_value=1, max_value=100)
                removenoise(st.session_state.noise)
                st.session_state.noise = addnoise(
                    snrratio, st.session_state.total)
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
                    "Signal Type:",
                    ('sin', 'cos'))
                col1, col2, col3 = st.columns(3)
                with col1:
                    amp = st.text_input("amplituide")
                with col2:
                    freq = st.text_input("frequency")
                with col3:
                    st.text("")
                    st.text("")
                    Button = st.button("add")
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
                if removebutton and len(st.session_state.all_signals) != 0:
                    removesignal(
                        remove_signal["type"], remove_signal["amp"], remove_signal["freq"], t)
                    st.experimental_rerun()
                elif removebutton:
                    st.write("no signal to remove")

        xsampled, ysampled = sampling(
            t, st.session_state.total, SF)
        print(len(xsampled))
        yreconst = sinc_interp(ysampled, xsampled, t,SF)

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(y=st.session_state.total, x=t,
                      mode="lines", name="Signal"), row=1, col=1)
        fig.add_trace(go.Scatter(y=ysampled, x=xsampled,
                      mode="markers", name="samples"), row=1, col=1)
        fig.add_trace(go.Scatter(y=yreconst, x=t, mode="lines",
                      name="reconstruction"), row=1, col=1)
        fig.update_xaxes(title_text='Time (in seconds)')
        fig.update_yaxes(title_text='Amplitude (in volts)')
        fig.update_layout(autosize=True)

        st.plotly_chart(fig, use_container_width=True)
        with st.sidebar:
            st.download_button("download csv file", download(t).to_csv(),
                               file_name='signal.csv', mime='text/csv')


elif select == "sin":

    st.session_state.csv_mode = 0
    dt = 0.001
    time = np.arange(0, 1, dt)
    if st.session_state.sin_mode == 0:
        # st.session_state.stack.append(y_file)
        st.session_state.sin_mode += 1
        st.session_state.total = np.zeros(len(time))
        st.session_state.noise = np.zeros(len(time))
        st.session_state.all_signals.clear()
        addsignal("sin", 1, 5, time)
    if len(st.session_state.all_signals) == 0:
        st.session_state.total = np.zeros(1000)
    # st.write(st.session_state.max_freq)

    # st.session_state.all_signals.clear()
    with st.sidebar:
        genre = st.radio(
            "Signal type",
            ('sin', 'cos'),)
        col1, col2 = st.columns(2)
        with col1:
            freqency = st.slider("frequency", min_value=1, max_value=100)
        with col2:
            amp = st.slider("amp", min_value=1, max_value=100)

        add_signal_button = st.button("add signal")
        if add_signal_button and genre == 'sin':
            addsignal("sin", int(amp), int(freqency), time)
        elif add_signal_button and genre == 'cos':
            addsignal("cos", int(amp), int(freqency), time)

        sampling_frequency = st.slider(
            'sampling frequency', st.session_state.max_freq, 5*st.session_state.max_freq, 2*st.session_state.max_freq)

    with st.sidebar:
        snrbutton = st.checkbox("add noise")
        if snrbutton:
            snrratio = st.slider("snr", value=50, min_value=1, max_value=100)
            removenoise(st.session_state.noise)
            st.session_state.noise = addnoise(snrratio, st.session_state.total)
            st.session_state.check = 1
        elif snrbutton == 0:
            removenoise(st.session_state.noise)
            st.session_state.noise = np.zeros(len(time))
            if st.session_state.check == 1:
                st.session_state.check = 0
                st.experimental_rerun()

        remove_signal = st.selectbox(
            "remove signal", st.session_state.all_signals)
        removebutton = st.button("remove")

        if removebutton and len(st.session_state.all_signals) != 0:
            # type: ignore
            removesignal(
                remove_signal["type"], remove_signal["amp"], remove_signal["freq"], time)
            st.experimental_rerun()
        elif removebutton:
            st.write("no signal to remove")

    x_sampled, y_sampled = sampling(
        time, st.session_state.total,  sampling_frequency)
    y_inter = sinc_interp(y_sampled, x_sampled, time,sampling_frequency)
    # fig = plt.figure()
    # plt.plot(time, st.session_state.total)
    # plt.stem(x_sampled, y_sampled, linefmt='yellow',
    #          markerfmt='x', bottom=0, use_line_collection=True)
    # plt.plot(time,y_inter)
    # plt.xlabel("time")
    # plt.ylabel("signal")
    # st.plotly_chart(fig, use_container_width=True)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(y=st.session_state.total, x=time,
                  mode="lines", name="Signal"), row=1, col=1)
    fig.add_trace(go.Scatter(y=y_sampled, x=x_sampled,
                  mode="markers", name="samples"), row=1, col=1)
    fig.add_trace(go.Scatter(y=y_inter, x=time, mode="lines",
                  name="reconstruction"), row=1, col=1)
    fig.update_xaxes(title_text='Time (in seconds)')
    fig.update_yaxes(title_text='Amplitude (in volts)')
    fig.update_layout(autosize=True)
    st.plotly_chart(fig, use_container_width=True)

    with st.sidebar:
        st.download_button("download csv file", download(time).to_csv(),
                           file_name='signal.csv', mime='text/csv')
