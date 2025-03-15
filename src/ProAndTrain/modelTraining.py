from dataProcessing import df
from model import MLModel

import numbers as np

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

X = np.array(df.drop(columns = ['Biopsy'])).astype('float32')
y = np.array(df['Biopsy']).astype('float32')


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_test, x_val, y_test, y_val = train_test_split(X, y, test_size = 0.5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


model = MLModel("xgboost")
model.train(X_train, y_train)


result_train = model.get_score(X_train, y_train)
result_test = model.get_score(X_test, y_test)
result_train, result_test
