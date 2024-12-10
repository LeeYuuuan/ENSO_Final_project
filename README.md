# Nino3.4 Index Prediction with Deep Learning

This repository contains code and resources for predicting the Nino3.4 index using deep learning models such as CNN, LSTM, and Transformer. It implements various models and provides experiments for their comparison and analysis.



---

## Table of Contents


- [**Introduction**](#introduction)
- [**Explanation of Directories & Files**](#explanation-of-directories--files)  
- [**Requirements**](#requirements)  



---

## Introduction
The EI Nino-Southern Oscillation (ENSO) phenomenon, which occurs in the tropical Pacific
Ocean, is the strongest and most significant interannual climate signal on Earth. It can affect
weather around the world, changing the odds of floods, drought, heatwaves and cold seasons
for different regions even raising global temperatures. 

Thus, accurate prediction of ENSO is
the key to improve climate prediction and disaster prevention and reduction in East Asia and
the world. Nino 3.4 Sea Surface Temperature Anomalies (Nino 3.4 SST, Nino 3.4 Index) is one
of the key indicators of ENSO. If the SST in this area is 0.5°C or more above the longterm
average for several months, it indicates an El Ni˜no event. If the SST is 0.5°C or more below the
average, it indicates a La Ni˜na event. 

There are multiple features that can be used to predict
the Nino 3.4 index, including sea surface temperature (SST, anomalies of SST in the Nino 3.4
region are the primary reference for predicting the index.), wind anomalies, oceanographic data,
atmospheric data, etc. Recent advances in deep learning offer promising approaches to model
complex, non-linear relationships in climate data. Thus, we can use deep learning methods to
predict the Nino 3.4 index

This project aims to preciously predict the Nino 3.4 Surface Temperature Anomalies by
achieving the following objectives. First, find and create a dataset suitable for predicting the
Nino 3.4 Index, incorporating diverse climate features. Second, Design appropriate models for
this task. Then, train and evaluate these deep learning model with high accuracy. In the end,
Conduct feature evaluation on the dataset to assess the impact of different features on prediction
outcomes.

##  **Explanation of Directories & Files**

### **model/**
This folder contains the implementation of various machine learning models:

- **CNNModel.py**: Implementation of CNN model.  
- **LSTMModel.py**: Implementation of LSTM model.  
- **TransformerModel.py**: Implementation of Transformer model.

---

### **ENSODATA.py**

This file contains:
- **EarthDataSet class** (inherits from `torch.utils.data.Dataset`)  
- **Load_Data() method**: Used to load climate datasets for Nino3.4 index prediction.

---

### **ENSOTrain.ipynb**

An experimental Jupyter Notebook to:
- Train machine learning models.  
- Visualize training results.  
- Analyze and debug experiments.

---

### **train_model.py**

This script contains the model training logic. It allows users to specify:
- Model type (CNN, LSTM, Transformer, etc.).  
- Dataset and training configurations.  

It provides a unified interface to train multiple models.

---

### **utils.py**

A utility script that contains:
- Functions for drawing and saving loss figures.  
- Path-checking utilities for ensuring directories exist.  
- Additional common helper functions.

---

##  **Requirements**

To run this project, you will need the following dependencies:
```bash
torch>=1.x.x
tqdm
matplotlib
jupyter
xarray
```


