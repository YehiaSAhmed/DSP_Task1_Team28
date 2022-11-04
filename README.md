# Signal Studio - Task 1 DSP
## About
Sampling Studio is a webapp that depicts signal sampling and recovery while emphasising the significance and validation of the Nyquist rate.
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
- Sample a signal with varying sampling frequency and reconstruct the sampled points.
- reconstruct a signal with either normalized frequency (with a range from 1 to 5fmax) or another frequency number (in Hz).
- Visualize Interactive plots (zoom , pan, slice, and download as images). 
- View and Hide each curve on the same graph.
- Add or remove sinusoidal signals (sin or cosine) of varying frequencies and magnitudes.
- Add or remove noise with a variable SNR level.
- Save signal as csv file extension.

## Preview
#### Home Page
![home](https://user-images.githubusercontent.com/93640020/199202350-7acc7cef-380f-49d8-956e-4f6c97b5ebc6.png)

#### Load CSV
![Screenshot (343)](https://user-images.githubusercontent.com/93640020/199202532-08ed2ac9-33ea-4402-a3e6-e7bb50578763.png)

#### Compose and mix sinusoidals
![Screenshot (344)](https://user-images.githubusercontent.com/93640020/199202852-d58c25bc-b5e4-49f8-a185-8a051ec1abb0.png)

#### View and hide different curves
![Screenshot (345)](https://user-images.githubusercontent.com/93640020/199203025-a2c2485c-550f-4e2c-b9da-571bae161b94.png)

#### Zoom and pan
![Screenshot (346)](https://user-images.githubusercontent.com/93640020/199203756-fbe48b11-e0a3-42b3-8f99-8071d748bf29.png)

#### View in fullscreen
![Screenshot (349)](https://user-images.githubusercontent.com/93640020/199203871-918bba24-ea3e-4f0d-a7d1-c20644e451c6.png)

#### Add noise
![Screenshot (347)](https://user-images.githubusercontent.com/93640020/199203974-0d919b4a-684c-46a7-bd45-314b706e945c.png)



