import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../../data/risk_factors_cervical_cancer.csv")

#pd.set_option('display.max_rows', None)  # Display all rows
#pd.set_option('display.max_columns', None)  # Display all columns
#pd.set_option('display.width', None)  # Allow output to span multiple lines if necessary
#pd.set_option('display.max_colwidth', None)  # Ensure that long text in columns is fully visible

#sexual partneres = mean(the ones who has the same age)
#first sexual intercourse, mean of fsi among all that have the same age
#number of pregnancies : mean of nop of those that have the same age
df.drop('STDs: Time since last diagnosis', axis=1, inplace=True)
df.drop('STDs: Time since first diagnosis', axis=1, inplace=True)

df = df.loc[:, ~((df.columns.str.startswith('STDs:')) & (df.columns != 'STDs: Number of diagnosis'))]

corr_matrix = df.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(8, 6))  # Set the size of the plot
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
