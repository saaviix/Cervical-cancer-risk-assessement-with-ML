
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("risk_factors_cervical_cancer.csv")


df.drop('STDs: Time since last diagnosis', axis=1, inplace=True)
df.drop('STDs: Time since first diagnosis', axis=1, inplace=True)

df = df.loc[:, ~((df.columns.str.startswith('STDs:')) & (df.columns != 'STDs: Number of diagnosis'))]

df = df.apply(pd.to_numeric, errors='coerce')