# Nino3.4 Index Prediction with Machine Learning Models

This repository provides code and models for predicting the **Nino3.4 index** using machine learning models like CNN, LSTM, and Transformer.

---

## 🚀 **Table of Contents**

- [**Explanation of Directories & Files**](#explanation-of-directories--files)  
- [**Getting Started**](#getting-started)  
- [**Requirements**](#requirements)  
- [**Setup Instructions**](#setup-instructions)  
- [**Training & Usage**](#training--usage)  

---

## 🗂️ **Explanation of Directories & Files**

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

## 🛠️ **Getting Started**

### Requirements
To run this project, you will need the following dependencies:
```bash
torch>=1.x.x
tqdm
matplotlib
jupyter
