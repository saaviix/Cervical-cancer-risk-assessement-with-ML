from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
#from catboost import CatBoostClassifier


class MLModel:
    def __init__(self, model_name: str):
        """
        Initializes the ML model based on the model_name passed.
        
        :param model_name: Name of the model to use. Available options: 'random_forest', 'xgboost', 'svm', 'catboost'.
        """
        self.model_name = model_name.lower()
        self.model = self._get_model()

    def _get_model(self):
        """
        Returns the corresponding model based on the provided model_name.
        """
        if self.model_name == 'random_forest':
            return RandomForestClassifier()
        elif self.model_name == 'xgboost':
            return xgb.XGBClassifier()
        elif self.model_name == 'svm':
            return SVC()
        elif self.model_name == 'catboost':
            return# CatBoostClassifier(silent=True)
        else:
            raise ValueError(f"Model '{self.model_name}' is not recognized. Available options: 'random_forest', 'xgboost', 'svm', 'catboost'.")

    def train(self, X_train, y_train):
        """
        Trains the model on the provided training data.
        
        :param X_train: Features for training.
        :param y_train: Target values for training.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Makes predictions on the provided test data.
        
        :param X_test: Features for prediction.
        :return: Predicted values.
        """
        return self.model.predict(X_test)


    def get_model(self):
        """
        Returns the current model instance.
        
        :return: The ML model instance.
        """
        return self.model

    def get_score(self, X_test, y_test):
        """
        Returns the accuracy score of the model on the provided test data.
        
        :param X_test: Features for testing.
        :param y_test: Target values for testing.
        :return: Accuracy score.
        """
        return self.model.score(X_test, y_test)
