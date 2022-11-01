# Signal Studio - Task 1 DSP
## About
Signal Studio is a webapp that depicts signal sampling and recovery while emphasising the significance and validation of the Nyquist rate.
## Team Members
Name| Section 
--- | --- |
Aya Sameh Ahmed | 1 
Mohamed Hashem Abdelwareth  | 2 
Mina Safwat Samy  | 2
Yehia Said Ahmed | 2 
## How to run
1. **_Clone the repository_**

```sh
$ git clone https://github.com/YehiaSAhmed/DSP_Task1_Team28
```
2. **_Navigate to repository directory_**
```sh
$ cd DSP_Task1_Team28
```
3. **_install project dependencies_**
```sh
pip install -r requirements.txt
```
4. **_Run the application_**
```sh
streamlit run app.py
```
## libraries
- streamlit
- pandas
- numpy
- plotly.express
- plotly.graph_objs
- matplotlib.pyplot
## Features
This web app allows user to
- Load and plot a CSV Signal or compose and mix their own Sinusoidals.
- Sample a signal with varying sampling frequency and recostruct the sampled points.
- Visualize Interactive plots (zoom , pan, slice, and download as images) . 
- View and Hide each curve on the same graph.
- Add or remove sinusoidal signals (sin or cosine) of varying frequencies and magnitudes.
- Add or remove noise with a variable SNR level.
- Save signal as csv file extension.



