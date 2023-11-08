import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn

from sklearn import tree

# Klassifizierungsalgorithmen
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model

from sklearn.model_selection import train_test_split



class MyAnalyse(object):

    def __init__(self,fname,bereinige=False):

        self.fname = fname

        self.data = pd.read_csv(fname)

        if bereinige == True:
            # bereinigen der Daten
            self.data.dropna(inplace=True)
            self.data.drop_duplicates(inplace=True)
            self.data.reset_index(inplace=True,drop=True)

        # Umwandeln in Numerische Werte
        # Das Mapping wird in diesen Dictionaries definiert, in den späteren Zeilen
        # dann angewandt
        d_gender = {"Male": 0, "Female": 1}
        d_education = {"High School Diploma": 0, "Associate's Degree": 1, 
                       "Bachelor's Degree":2, "Master's Degree": 3,
                        "Doctorate": 4 }
        d_MartialStatus = {"Single": 0, "Married": 1}
        d_HomeOwnership = {"Rented": 0, "Owned": 1}
        d_CreditScore = {"Low": 0, "Average": 1, "High": 2}


        self.data["Gender"] = self.data["Gender"].map(d_gender)
        self.data["Education"] = self.data["Education"].map(d_education)
        self.data["Marital Status"] = self.data["Marital Status"].map(d_MartialStatus)
        self.data["Home Ownership"] = self.data["Home Ownership"].map(d_HomeOwnership)
        self.data["Credit Score"] = self.data["Credit Score"].map(d_CreditScore)

        # Extrahieren der Daten, macht das Arbeiten mit den Algorithmen leichter
        self.gender = self.data["Gender"].to_numpy()
        self.education = self.data["Education"].to_numpy()
        self.maritalStatus=self.data["Marital Status"].to_numpy()
        self.homeOwner = self.data["Home Ownership"].to_numpy()
        self.age = self.data["Age"].to_numpy()
        self.income = self.data["Income"].to_numpy()
        self.children = self.data["Number of Children"].to_numpy()

        self.creditScore = self.data["Credit Score"].to_numpy()

        # Skalieren des Einkommens
        #self.income = (1/1000) * self.income 

        # Zusammenfassen der Features und definieren der Klassennamen als X und Y
        self.X = np.column_stack((self.age,self.gender,self.income,self.education,self.maritalStatus,self.children,self.homeOwner))
        # Speichern der Spaltenüberschriften für die Visualisierung des Decision-Trees
        self.XName = ["Age", "Gender", "Income", "Education", "Marital Status", "Children", "HomeOwner"]
        self.Y = self.creditScore

        # Aufsplitten in Test- und Trainingsdaten
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,self.Y,test_size=0.2,random_state=42)

    def classifyViaKMeans(self):
        # Keine Ahnung wie viele Cluster sinnvoll sind, daher eine Schleife über mehrere Möglichkeiten
        for k in range(2,15):
            
            myKMeans = KMeans(n_clusters=k,n_init="auto")
            myKMeans.fit(self.X_train)
            # Spezialfall bei KMeans: Da mir die Ergebnisse komisch vorkamen 
            # auch auf Trainingsdaten anwenden
            y_pred_train = myKMeans.predict(self.X_train)
            y_pred_test = myKMeans.predict(self.X_test)

            correct = 0
            for i in range(len(y_pred_test)):
                if y_pred_test[i] == self.Y_test[i]:
                    correct += 1
            print("N_clusters: ", k, " Accuracy KMeans auf Testdaten: ", 100*correct/len(y_pred_test))

            correct = 0
            for i in range(len(y_pred_train)):
                if y_pred_train[i] == self.Y_train[i]:
                    correct += 1
            print("N_clusters: ", k, " Accuracy KMeans auf Trainingsdaten: ", 100*correct/len(y_pred_train))


    def classifyViaDecisionTree(self):
        dtree = DecisionTreeClassifier()
        dtree.fit(self.X_train,self.Y_train)
        tree.plot_tree(dtree,feature_names=self.XName)
        plt.savefig("DecisionTree.pdf")

        # testen
        correct = 0
        y_pred = dtree.predict(self.X_test)
        for i in range(len(y_pred)):
            if y_pred[i] == self.Y_test[i]:
                correct += 1

        print("Accuracy DecisionTree: ", 100*correct/len(y_pred))

    def classifyViaRandomForest(self):
        rforest = RandomForestClassifier()
        rforest.fit(self.X_train,self.Y_train)

        # testen
        correct = 0
        y_pred = rforest.predict(self.X_test)
        for i in range(len(y_pred)):
            if y_pred[i] == self.Y_test[i]:
                correct += 1

        print("Accuracy Random Forest: ", 100*correct/len(y_pred))

    def classifyViaKNN(self):
        KNN = KNeighborsClassifier(n_neighbors=5)
        KNN.fit(self.X_train,self.Y_train)
        # testen
        correct = 0
        y_pred = KNN.predict(self.X_test)
        for i in range(len(y_pred)):
            if y_pred[i] == self.Y_test[i]:
                correct += 1

        print("Accuracy KNN: ", 100*correct/len(y_pred))


    def classifyViaMultipleRegression(self):
        reg = linear_model.LinearRegression()
        reg.fit(self.X_train,self.Y_train)

        y_pred = reg.predict(self.X_test)

        # Runden auf die entsprechenen werte:
        y_pred = np.round(y_pred)

        correct = 0
        for i in range(len(y_pred)):
            if y_pred[i] == self.Y_test[i]:
                correct += 1

        print("Accuracy Multiple Regression: ", 100*correct/len(y_pred))

    def showData(self):
        seaborn.pairplot(self.data,hue="Credit Score")
        plt.savefig("Datenvisualisierung.pdf")
        plt.show()




if __name__ == "__main__":
    fname = "CreditScores.csv"
    A = MyAnalyse(fname,bereinige=False)
    A.classifyViaDecisionTree()
    A.classifyViaRandomForest()
    A.classifyViaKNN()
    A.classifyViaMultipleRegression()
    A.classifyViaKMeans()
    A.showData()