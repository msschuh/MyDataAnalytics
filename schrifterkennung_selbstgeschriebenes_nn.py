import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

import time

'''
Ziel ist es, ein möglichst allgemeines neuronales Netz
zur Erkennung von handschriftlichen Ziffern von Grund
auf zu implementieren.
Zur Vereinfachung wird sich auf das Gradientenverfahren
und die Sigmoid-Funktion beschränkt
'''
class GeneralNeuralNetwork(object):
    
    def __init__(self,dataObject):

        self.dataObject = dataObject

        # Trainingsdaten, Testdaten etc rausfischen
        self.X = dataObject.X
        self.Y = dataObject.Y
        self.XTrain = dataObject.XTrain
        self.XTest = dataObject.XTest
        self.YTrain = dataObject.YTrain
        self.YTest = dataObject.YTest

        # Damit sind die Neuronenzahlen für
        # Eingabe und Ausgabe bereits festgelegt:
        self.NNeuronenInput = len(self.X[0])
        self.NNeuronenOutput = len(self.Y[0])

        # Statische Lernrate
        self.Lernrate = 0.3

        # Zufallszahlengenerator initialisieren
        self.rng = np.random.default_rng(seed=42)

    def setupNeuralNetwork(self,NeuronenHiddenLayer):

        self.NNeuronen = [self.NNeuronenInput]
        for N in NeuronenHiddenLayer:
            self.NNeuronen.append(N)
        self.NNeuronen.append(self.NNeuronenOutput)

        # Intervall in dem die Uebergangsmatritzen initialisiert werden sollen
        a = -0.5
        b = 0.5
        self.uebergangsmatritzen = []
        for i in range(len(self.NNeuronen)-1):
            Nin = self.NNeuronen[i]
            Nout = self.NNeuronen[i+1]
            W = (b-a)*self.rng.random((Nout,Nin)) + a
            self.uebergangsmatritzen.append(W)

    def test(self,x):
        y = x.copy()
        for W in self.uebergangsmatritzen:
            y = np.dot(W,y)
            y = self.sigmoid(y)
        
        return y
    
    def train_oneRun(self,x,y):

        InputInLayer = []
        OutputOfLayer = []
        errorOfLayer = len(self.uebergangsmatritzen)*[None]

        v = x.copy()

        for W in self.uebergangsmatritzen:
            InputInLayer.append(v.copy())
            v = np.dot(W,v)
            v = self.sigmoid(v)
            OutputOfLayer.append(v.copy())

        # Fehler berechnen
        errorFinalOutput = y - OutputOfLayer[-1]

        # Und zurückpropagieren. Initialisierung:
        errorOfLayer[-1] = errorFinalOutput
        for i in reversed(range(len(self.uebergangsmatritzen)-1)):
            W = self.uebergangsmatritzen[i+1]
            errorOfLayer[i] = np.dot(W.T,errorOfLayer[i+1])

        # Updates berechnen
        for i in range(len(self.uebergangsmatritzen)):
            outputError = errorOfLayer[i]
            outputLayer = OutputOfLayer[i]
            inputLayer = InputInLayer[i]
            deltaW = np.dot(outputError*outputLayer*(1-outputLayer),np.transpose(inputLayer))
            #print(deltaW.shape)

            self.uebergangsmatritzen[i] += self.Lernrate*deltaW


        #print(InputInLayer)   
        #print(OutputOfLayer)
        #print(errorOfLayer)
        #print(errorFinalOutput)
        del InputInLayer
        del OutputOfLayer
        del errorOfLayer

    def train(self):
        for i in range(len(self.XTrain)):
            x = self.XTrain[i]
            y = self.YTrain[i]
            self.train_oneRun(x,y)
    
    def testWithAll_Classification(self):
        correct = 0
        for i in range(len(self.XTest)):
        #for i in range(1):
            x = self.XTest[i]
            y = self.YTest[i]

            ypred = self.test(x)
            tmp = np.zeros(y.shape)
            tmp[np.argmax(ypred)] = 1
            if np.array_equal(y,tmp):
                correct += 1
        
        print("Correct: {:d}, d.h. {:.2f}%".format(correct,100*correct/len(self.XTest)))

        return (correct, correct/len(self.XTest))


    def sigmoid(self,x):
        y = 1 / (1+np.exp(-x))
        return y
    
    def ddxSigmoid(self,x):
        tmp = self.sigmoid(x)
        y = tmp*(1 - tmp)
        return y
    
    def testFunktionalitaetSingleRun(self):
        x = self.X[0]
        y = self.Y[0]

        res = self.test(x)
        print("Aus Testfunktionalitaet")
        print(x)
        print(y)
        print(res)

    def trainFunktionalitaetSingleRun(self):
        x = self.X[0]
        y = self.Y[0]

        res = self.train_oneRun(x,y)
        print("Aus Trainfunktionalitaet")
        print(x)
        print(y)
        print(res)

class DataSchrifterkennung(object):
    def __init__(self,fname):
        self.fname = fname
        self.data = pd.read_csv(fname)
        self.dataList = self.data.values.tolist()

       
        self.Y = []
        self.X = []
        for cont in self.dataList:
            numCode = np.zeros(10)
            numCode[cont[0]] = 1

            self.Y.append(np.array(numCode,ndmin=2).T)
            
            self.X.append(np.array(cont[1:],ndmin=2).T)

        # Skalieren und in Numpy umwandeln
        self.X = (np.asfarray(self.X)/255)*0.99 + 0.01
        self.Y = np.asfarray(self.Y)

        self.XTrain, self.XTest, self.YTrain, self.YTest = train_test_split(self.X,self.Y,test_size=0.2,random_state=42)


if __name__ =="__main__":
    fname = "mnist_data_60k.csv"
    A = DataSchrifterkennung(fname)
    lernraten = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    performance = []
    runtime = []
    for lernrate in lernraten:
        print(lernrate)
        starttime = time.time()
        B = GeneralNeuralNetwork(A)
        B.Lernrate = lernrate
        B.setupNeuralNetwork([50])
        B.train()
        p = B.testWithAll_Classification()
        stoptime = time.time()
        runtime.append(stoptime - starttime)
        performance.append(p)
        del B
    
    print(lernraten)
    print(performance)
    print(runtime)
    performanceFrac = [x[1] for x in performance]
    plt.figure(1)
    plt.plot(lernraten,performanceFrac)
    plt.xlabel("Lernrate")
    plt.ylabel("Performance")
    plt.figure(2)
    plt.plot(lernraten,runtime)
    plt.xlabel("Lernrate")
    plt.ylabel("Runtime")
    plt.show()
