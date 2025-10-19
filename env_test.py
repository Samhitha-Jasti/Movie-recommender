# env_test.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask
import nltk
import requests
import tqdm
import yaml
import os

print("✅ All core packages imported successfully!")

# Simple sanity test
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)

# Quick model test
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])
model = LinearRegression().fit(X, y)
print("Model prediction for 5:", model.predict([[5]])[0])