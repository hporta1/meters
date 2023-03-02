
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score


from source.data_preprocess import DataPreprocessing


class ModelBuilder2(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder2, self).__init__(*args, **kwargs)

    def dt(self, X_train, X_test, y_train, y_test,maxi):
        #adding things to change

        ANN_classifier = MLPClassifier(hidden_layer_sizes=(5,5), max_iter=maxi,learning_rate_init=.005)

        #Train the model
        ANN_classifier.fit(X_train, y_train)

        #Test the model
        ANN_predicted = ANN_classifier.predict(X_test)



        return ANN_classifier