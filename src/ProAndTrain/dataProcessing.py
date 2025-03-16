
import numpy as np
import pandas as pd

from math import floor

import seaborn as sns

import matplotlib.pyplot as plt



def fill_nan_with_group_mean(row, df_filtered, column):
    if pd.isna(row[column]):

        mean_value = df_filtered[df_filtered['Age'] == row['Age']][column].mean()
        if column != 'Hormonal Contraceptives (years)':
            return floor(mean_value)
        else:
            return mean_value
    else:
        return row[column]

def process_data(df : pd.DataFrame):
    df.drop('STDs: Time since last diagnosis', axis=1, inplace=True)
    df.drop('STDs: Time since first diagnosis', axis=1, inplace=True)

    df = df.loc[:, ~((df.columns.str.startswith('STDs:')) & (df.columns != 'STDs: Number of diagnosis'))]

    df = df.apply(pd.to_numeric, errors='coerce')

    age_counts = df['Age'].value_counts()

    valid_ages = age_counts[age_counts > 1].index

    df_filtered = df[df['Age'].isin(valid_ages)]

    columns = ['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Hormonal Contraceptives (years)']
    for column in columns:
        df_filtered[column] = df_filtered.apply(lambda row: fill_nan_with_group_mean(row, df_filtered, column), axis=1)

    df.update(df_filtered)
    return df