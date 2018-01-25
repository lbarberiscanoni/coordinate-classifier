import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import scipy as sp
from scipy import stats
import time

#sklearn is a library to implement generic machine learning models
#pandas are data structures that work very well for ML


class Classifier():

    def loadData(self):
        data = pd.read_csv("TokenCoord1a.csv")
        return data

    #saving the model as a pickle
    def saveModel(self, model):
        pickPath = open("kneighbor.pkl", "wb")
        pickle.dump(model, pickPath)
        pickPath.close()

    def run(self):
        data = self.loadData()

        #we need the double squared brackets to indicate we want both
        attributes = data[["Angle", "Dist"]]
        #all the data for each token
        target = data["Token"]

        #separating our data set into the training set and the testing set. We train the model with the former and test its predictive capacity with the latter
        X_train, X_test, y_train, y_test = train_test_split(attributes, target, test_size=0.6, random_state=10)

        #looping through progressively higher numbers of Kneighbors to see which one is optimal
        for i in range(1, 20):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            print i, score
        
        #saving our model with 3 for now bc it's reasonable, but if 5 was the clear winner we would change this number to 5
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        self.saveModel(knn)

Classifier().run()
