import pandas as pd
import numpy as np
import statistics as st
from matplotlib import pyplot as plt

import math

'''
Aufgabenstellung: Untersuche mit einfachen Mitteln
den Zusammenhang zwischen Position und Gehalt zu analysieren
und eine Regression zu berechnen
'''


fname = "Salaries.csv"


class MyAnalyse(object):

    def __init__(self,myfile):
        self.myfile = myfile
        self.df = pd.read_csv(myfile)
        #print(self.df)

        # Bereinigen: Nullzeilen entfernen
        self.df.dropna()
        # Sinnvoll Duplikate zu entfernen? Nein, zB wenn es Tarifverträge gibt und alle pro
        # Stufe identische Gehälter haben
        #self.df = self.df.drop_duplicates()
        #print(self.df)
        print(self.df.shape)
    

    def visualisiere(self):
        plt.figure()
        plt.scatter(x=self.df["Level"],y=self.df["Salary"])
        plt.xticks(self.df["Level"],labels=self.df["Position"],rotation="vertical")
        plt.margins(0.1)
        plt.subplots_adjust(bottom=0.35)
        plt.xlabel("Level")
        plt.ylabel("Gehalt")
        plt.title("Gehaltsvergleich")

    def haeufigkeitsanalyse(self):
        self._haeufigkeitsanalyse("Position")
        self._haeufigkeitsanalyse("Salary")

    def _haeufigkeitsanalyse(self,searchkey):
        haeufigkeiten_pos = self.df.value_counts(searchkey)
        idxmax = None
        maxval = -math.inf
        for idx in haeufigkeiten_pos.index:
            if idxmax == None:
                idxmax = idx
                maxval = haeufigkeiten_pos[idx]
            elif haeufigkeiten_pos[idx] > maxval:
                idxmax = idx
                maxval = haeufigkeiten_pos[idx]

        print("Es gibt am meisten {:}, naemlich {:d} mal".format(idxmax,maxval))
    
    def medianbestimmung(self):
        meddf = self.df.median(numeric_only=True)
        med = meddf["Salary"]
        print("Der Median des Gehalts ist: ", med)

    def meanbestimmung(self):
        meandf = self.df.mean(numeric_only=True)
        meanv = meandf["Salary"]
        print("Das Durchschnittliche Gehalts ist {:.2f}".format(meanv))

        return meandf

    def standardAbweichung(self):
        stddevs = self.df.std(numeric_only=True)
        std_gehalt = stddevs["Salary"]
        print("Die Standardabweichung vom Gehalt ist: {:.2f}".format(std_gehalt))

        return stddevs

    
    def korrelation(self):
        korrealationtabelle = self.df.corr(method="pearson",numeric_only=True)
        korr = korrealationtabelle.loc["Level","Salary"]
        print("Der Korrleationskoeffizient zwischen Level und Salary ist: {:.2f}".format(korr))
        return korrealationtabelle

    def linRegression(self):
        stddevs = self.standardAbweichung()
        sx = stddevs["Level"]
        sy = stddevs["Salary"]

        korrelationen = self.korrelation()
        print(korrelationen)
        korr = korrelationen.loc["Level","Salary"]

        means = self.meanbestimmung()
        meanSalary = means["Salary"] # ymean
        meanLevel = means["Level"]  # xmean

        # Steigung der Regressionsgerade
        slope = (sy / sx)*korr

        y0 = -(sy/sx)*korr*meanLevel + meanSalary
        print("SX: ", sx, "SY: ", sy, " korr: ", korr)
        print("Slope = ",slope," intercept: ", y0)

        regdata = slope*self.df["Level"] + y0

        plt.figure()
        plt.scatter(x=self.df["Level"],y=self.df["Salary"],c="r")

        plt.plot(self.df["Level"],regdata,c="b")

        plt.xticks(self.df["Level"],labels=self.df["Position"],rotation="vertical")
        plt.margins(0.1)
        plt.subplots_adjust(bottom=0.35)
        plt.xlabel("Level")
        plt.ylabel("Gehalt")


    def polynomregression(self):
        x = self.df["Level"]
        y = self.df["Salary"]
        # Plotgrenzen
        xmin = min(x)
        xmax = max(x)
        plt.figure()
        mymodel1 = np.polynomial.Polynomial.fit(x,y,1)
        mymodel2 = np.polynomial.Polynomial.fit(x,y,2)
        mymodel3 = np.polynomial.Polynomial.fit(x,y,3)
        mymodel4 = np.polynomial.Polynomial.fit(x,y,4)


        myline = np.linspace(xmin,xmax,100)

        plt.plot(myline,mymodel1(myline), label="Grad 1")
        plt.plot(myline,mymodel2(myline), label="Grad 2")
        plt.plot(myline,mymodel3(myline), label="Grad 3")
        plt.plot(myline,mymodel4(myline), label="Grad 4")
        plt.legend()
        plt.title("Regression vom Grad 1 bis 4")
        plt.xlabel("Level")
        plt.ylabel("Gehalt")
        plt.scatter(x=self.df["Level"],y=self.df["Salary"])
        plt.xticks(self.df["Level"],labels=self.df["Position"],rotation="vertical")
        plt.margins(0.1)
        plt.subplots_adjust(bottom=0.35)

        ymean = sum(y)/len(y)

        # Bestimmung der Streuung der Originaldaten:
        origStr = 0
        for i in range(len(y)):
            origStr += (y[i]-ymean)**2
        

        print("ymean: ", ymean)
        for m in [mymodel1,mymodel2,mymodel3,mymodel4]:
            strM = 0
            for i in range(len(x)):
                ypred = m(x[i])
                strM += (ypred - ymean)**2
            print("Bestimmheitsmass für Modell: ", strM / origStr)
        



    def showPlots(self):
        plt.show()




A = MyAnalyse(fname)
A.visualisiere()
A.haeufigkeitsanalyse()
A.medianbestimmung()
A.meanbestimmung()
A.standardAbweichung()
A.korrelation()
A.linRegression()
A.polynomregression()
A.showPlots()
