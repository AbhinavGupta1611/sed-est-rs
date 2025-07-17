# sed-est-rs
mTAN-LSTM algorithm to integrate sparse remote sensing data and continuous hydrometeorological data

Raw data and pre-trained models: 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15882343.svg)](https://doi.org/10.5281/zenodo.15882343)

### Overview

This repository contains codes to estimate suspended sediment concentration by integrating sparse and intermittent remote sensing data and continuous hydrometeorological data. The sparse remote sensing data were interpolated using the mTAN algorithm, adapted from Shukla and Marlin (2021).

### Installation instructions
#### 1. Clone the repo
     git clone https://github.com/AbhinavGupta1611/sed-est-rs.git
     cd sed-est-rs

#### 2. Create a virtual environment
    python -m venv env
    # Windows
    env\Scripts\activate
    # macOS/Linux
    source env/bin/activate

#### 3. Install required packages
    pip install -r requirements.txt

### Usage
To get started with these codes, a user can downlaod the raw data file 'dataLSTM_SR.txt' from Zenodo (https://doi.org/10.5281/zenodo.15882343)

A user can use their own data to run these codes. In this case, however, the user has to make sure that either the raw data is contained in the same format as 'dataLSTM_SR.txt', or the modules of the code that load the data (load_data.py) are accordingly changed.

The main scripts to train the models are 'lstm_train.py' and 'mTAN_LSTM_train.py'. To run these scripts, specify the name of the file containing the data and directory containing the file.
Similarly, to use the trained model to predict SSC, the main scripts are 'predict_LSTM.py' and 'predict_mTAN_LSTM.py'.

Note that the script 'lstm_train.py' will train a model without using remote sensing data. 

### Contact information:

Abhinav Gupta  - abhigupta.1611@gmail.com

![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/github/license/AbhinavGupta1611/sed-est-rs)


###  References:

Shukla, S. N., & Marlin, B. M. (2021). Multi-time attention networks for irregularly sampled time series. arXiv preprint arXiv:2101.10318.
