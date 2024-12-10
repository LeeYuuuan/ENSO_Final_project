# Nino3.4 Index Prediction with Deep Learning

This repository provides a deep learning pipeline for predicting the Nino 3.4 Index using CNN, LSTM, and Transformer models. The provided models are trained and evaluated using two main climate datasets: SODA (Simple Ocean Data Assimilation) and CMIP (Coupled Model Intercomparison Project).

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Directory Structure](#directory-structure)  
3. [Setup & Installation](#setup--installation)  
4. [Dataset Description](#dataset-description)  
5. [Model Overview](#model-overview)  
6. [How to Train Models](#how-to-train-models)  
7. [Experimentation with Multiple Models](#experimentation-with-multiple-models)  
8. [Results & Visualization](#results--visualization)  
9. [Acknowledgments](#acknowledgments)  

---

## Introduction

The goal of this project is to implement machine learning models (CNN, LSTM, Transformer, and combinations thereof) to predict the **Nino 3.4 Index**, which is a key climate indicator for ENSO (El Niño-Southern Oscillation) events. The results are achieved by leveraging climate datasets and testing the performance of these machine learning models under different experimental settings.

The implemented pipeline supports training and evaluation on different combinations of the SODA and CMIP datasets using various model types. This repository provides scripts and utilities for model training, feature analysis, and result visualization.

---

## Directory Structure

```plaintext
.
├── model/                      # Model definitions folder
│   ├── CNNModel.py             # CNN model implementation
│   ├── LSTMModel.py            # LSTM model implementation
│   └── TransformerModel.py     # Transformer model implementation
│
├── utils.py                     # Utility functions for plotting and validation
├── ENSODATA.py                  # Dataset class and data loading methods
│
├── ENSOTrain.ipynb              # Jupyter Notebook for training experiments
│
├── train_model.py               # Main model training script
│
├── results/                     # Output directory for storing results
│
└── README.md                    # Documentation file
