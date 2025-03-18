
import numpy as np
import pandas as pd

from ProAndTrain import dataProcessing, model

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


from imblearn.over_sampling import SMOTE


import joblib

print('f')
df = pd.read_csv("../data/risk_factors_cervical_cancer.csv")
df = dataProcessing.process_data(df)

ChoosenModel = "catboost"
Model = model.MLModel(ChoosenModel)

X = np.array(df.drop(columns = ['Biopsy'])).astype('float32')
y = np.array(df['Biopsy']).astype('float32')

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_test, x_val, y_test, y_val = train_test_split(X, y, test_size = 0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

Model.train(X_resampled, y_resampled)

result_train = Model.get_score(X_train, y_train)
result_test = Model.get_score(X_test, y_test)
print(result_train, result_test)

joblib.dump(Model.get_model(), "../genModels/" + ChoosenModel + ".pkl")
