#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 22:48:20 2022

@author: liamsweeney
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import category_encoders as ce
import matplotlib.pyplot as pl
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from pdpbox import pdp, info_plots
from sklearn.pipeline import make_pipeline
import sklearn
import streamlit as st 


st.title("My First Dashboard")

num_rows = st.sidebar.number_input('Select Number of Rows to Load', 
                                   min_value=1000,
                                   max_value=50000,
                                   step=1000)

df = pd.read_csv('/Users/liamsweeney/dat-11-15/Homework/Unit2/data/insurance_premiums.csv')
categorical = ['sex', 'smoker']
df_ohe = pd.get_dummies(df, columns=categorical)
ore = ce.OrdinalEncoder(mapping = [
    {
        'col': 'region',
        'mapping': {'northeast': 1, 'northwest': 2, 'southeast': 3, 'southwest': 4}
    }
])
df_ce = ore.fit_transform(df_ohe)
df_ce["charges"] = df_ce["charges"].astype(float)
fig = px.histogram(df_ce, x="bmi", title = "Body Mass Index Histogram")
fig.show()