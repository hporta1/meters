
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score


from source.data_preprocess import DataPreprocessing


class ModelBuilder2(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder2, self).__init__(*args, **kwargs)

    def dt(self, X_train, X_test, y_train, y_test):
        #adding things to change

        ANN_classifier = MLPClassifier(max_iter=20000, hidden_layer_sizes=(4,4),learning_rate_init=.01)

        #Train the model
        ANN_classifier.fit(X_train, y_train)

        #Test the model
        ANN_predicted = ANN_classifier.predict(X_test)

        error = 0
        for i in range(len(y_test)):
            error += np.sum(ANN_predicted != y_test)

        total_accuracy = 1 - error / len(y_test)

        #get performance
        self.accuracy = accuracy_score(y_test, ANN_predicted)

        return ANN_classifier