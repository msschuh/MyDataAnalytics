import math
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

'''
Aufgabenstellung war, auf zufällig generierten
Daten eine K-Fache Kreuzvalidierung durchzuführen
'''

class MyKreuzvalidierung(object):

    def __init__(self,data, maxPolynomgrad, nFolds,xDataColumn,yDataColumn):

        self.data = data
        self.nRows = data.shape[0]
        self.nColumns = data.shape[1]

        self.maxPolyGrad = maxPolynomgrad

        self.nFolds = nFolds

        self.indiziesTestdaten = []
        self.indizesTrainingsdatenFull = []
        self.indiziesFolds = []

        self.xDataColumn = xDataColumn
        self.yDataColumn = yDataColumn

        # Falls ein bestimmter Prozentsatz an Daten zurückgehalten werden soll
        self.fracTestDaten = 0.0

        self.MQF = []

    def prepareData(self):
        for i in range(self.nFolds):
            self.indiziesFolds.append([])
        
        # Um die Indizes für Trainingsdaten, Testdaten und folds zu verteilen
        # erstelle eine neue Liste mit allen vorhandenen Indizes.
        # Von dieser werden die Indizies einem Zweck zugeschlagen und mittels .pop() entfernt
        # Wenn diese Liste leer ist, sind alle Indizies verteilt
        indiziesAvailable = [i for i in df.index]

        # Testdaten zurückhalten falls gewünscht:
        for i in range(int(self.fracTestDaten*self.nRows)):
            lidxToGo = np.random.randint(0,len(indiziesAvailable))
            idxTestDaten = indiziesAvailable.pop(lidxToGo)            
            self.indiziesTestdaten.append(idxTestDaten)

        # Der Reset sind Trainingsdaten
        self.indizesTrainingsdatenFull = [i for i in indiziesAvailable]

        # Trainingsdaten, die auf die n Folds verteilt werden müssen
        # Der Fold, dem das Element zugeschlagen wird, wird mittels Modulo bestimmt:
        # Der aktuelle Durchlauf Modulo der Anzahl Folds
        # Dadurch sind alle Folds etwa gleich groß
        fold = 0
        while indiziesAvailable != []:
            foldThisRound = fold % self.nFolds
            lidxToGo = np.random.randint(0,len(indiziesAvailable))
            idx = indiziesAvailable.pop(lidxToGo)            
            self.indiziesFolds[foldThisRound].append(idx)

            fold+= 1
        
        #print(self.indiziesFolds)
        #print(self.indiziesTestdaten)
        #print(indiziesAvailable)
        #print(df.loc[self.indiziesTestdaten])

    def bestimmeRegressionen(self,iHoldFold):

        # Bestimme die Trainingsdaten für diese Regression:
        trainingsDatenAkt = []
        for i in range(0,len(self.indiziesFolds)):
            if i != iHoldFold:
                trainingsDatenAkt = np.concatenate([trainingsDatenAkt,self.indiziesFolds[i]])
        
        # Lokalisiere die X- und Y-Werte zu diesen Indizies
        x = self.data.loc[trainingsDatenAkt,self.xDataColumn].to_numpy()
        y = self.data.loc[trainingsDatenAkt,self.yDataColumn].to_numpy()

        # Speichere die Regressionspolynome in einer Liste
        regressionen = []
        # Schleife über alle Polynomgrade die für die Regression verwendet werden sollen
        # Beachten: regressionen[i] enthaelt das Regressionspolynom vom Grad i+1
        for i in range(1,self.maxPolyGrad+1):
            pol = np.polynomial.Polynomial.fit(x,y,i)
            regressionen.append(pol)
        
        # Beachten: regressionen[i] enthaelt das Regressionspolynom vom Grad i+1
        return regressionen
    
    def berechneMQF(self,myPoly,iTestSet):

        # Hole die Testdaten aus dem Gesamtdatensatz heraus
        indiziesTdata = self.indiziesFolds[iTestSet]
        x = self.data.loc[indiziesTdata,self.xDataColumn].to_numpy()
        y_data = self.data.loc[indiziesTdata,self.yDataColumn].to_numpy()

        # Werte das Regressionspolynom in den Datenpunkten aus
        y_regression = myPoly(x)

        # Berechne den MQF gemäß der Formel
        MQF = 0
        for i in range(len(x)):
            err_loc = y_data[i] - y_regression[i]
            MQF += err_loc**2
        
        MQF = MQF / len(x)
        return MQF
    
    def run(self):

        # 1. Schritt: Daten vorbereiten
        self.prepareData()

        # 2. Schritt
        # Zur Initialisierung muss vorher der MQF für jeden Grad
        # Auf Null gesetzt werden
        for i in range(self.maxPolyGrad):
            self.MQF.append(0)

        # 3. Schritt: Bestimme die Regressionen
        # für jedes Hold-Set        
        for iHoldSet in range(self.nFolds):
            regressionen = self.bestimmeRegressionen(iHoldSet)

            # 4. Schritt: Für jede der Regressionen wird der MQF berechnet
            for iRegPoly in range(len(regressionen)):
                MQF_loc = self.berechneMQF(regressionen[iRegPoly],iHoldSet)

                # MQF zu dem jeweiligen Polynomgrad aufaddieren:
                self.MQF[iRegPoly] += MQF_loc
        
        
        return self.MQF


# Für das Erzeugen eines Datensatzes
def datafunc(x,clean=False,eps=0):
    y = 0.25*math.exp(x)
    if clean == False:
        y += eps*(2*np.random.random() - 1)
    return y



if __name__ == "__main__":
    # Erzeugen von Zufallsdaten:
    # Anzahl zufallsdaten:
    nZufall = 10000

    # Bereich festlegen
    xMin = -2
    xMax = 7

    # Störung festlegen
    eps = 4

    x = []
    y = []

    for i in range(nZufall):
        xi = (xMax - xMin)* np.random.random() + xMin
        yi = datafunc(xi,clean=False,eps=eps)
        x.append(xi)
        y.append(yi)
    
    #y_corr = [datafunc(xi,clean=True) for xi in x]
    #diff = np.array(y_corr) - np.array(y)
    #print(diff)

    #plt.scatter(x,y)
    #pltx = np.linspace(xMin,xMax,num=200)
    #cleany = [datafunc(xi,clean=True) for xi in pltx]
    #plt.plot(pltx,cleany)
    #plt.show()

    # Festlegungen für die Aufgabe
    # Höchster Polynomgrad:
    maxGradPoly = 4
    # Anzahl Folds
    nFolds = 6

    # Bezeichnungen im Dataset
    xDataStr = "x"
    yDataStr = "y"

    # Pandas Dataframe zusammenbauen
    d = {xDataStr: x, yDataStr: y}
    df = pd.DataFrame(data=d)

    A = MyKreuzvalidierung(df,maxGradPoly,nFolds,xDataStr,yDataStr)

    print(A.run())
