import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

'''
Ziel war es, mit PyTorch ein Neuronales Netz
zu erstellen, verschiedene Parameter zu variieren und zu untersuchen,
wie sich die Performance verändert
'''
# Erstellen eines Neural Networks mittels Torch

# Die Klasse Erbt von nn.Module die ganzen Eigenschaften
class NN(nn.Module):
    def __init__(self,input_size,output_size,AktivierungsFunktion,paramAktFkt=None):
        super(NN,self).__init__()
        self.flatten = nn.Flatten()
        if paramAktFkt == None:
            self.network = nn.Sequential(
                nn.Linear(input_size,512),
                AktivierungsFunktion(),
                nn.Linear(512,512),
                AktivierungsFunktion(),
                nn.Linear(512,output_size)
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(input_size,512),
                AktivierungsFunktion(paramAktFkt),
                nn.Linear(512,512),
                AktivierungsFunktion(paramAktFkt),
                nn.Linear(512,output_size)
            )

        # Hyperparameter festlegen
        self.learning_rate = 0.0001
    
    def forward(self,x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits
    
    def train(self,train_dataloader,num_epochs=1):
        # Fehlerfunktion und Optimierungsverfahren festlegen
        # Vorher war es die quadratische Abweichung und das Gradientenverfahren
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

        # Training des neuronalen Netzes
        for epoch in range(num_epochs):
           for idx, (data,labels) in enumerate(train_dataloader):

               labels = labels.type(torch.LongTensor)

               # Daten durchs Netz
               outputs = self(data)
               loss = loss_function(outputs,labels)

               # Backpropagation
               # Gradienten auf 0 setzen
               optimizer.zero_grad()
               loss.backward()
               optimizer.step() # Gradientenverfahren

               #if idx%1000 == 0:
               #    loss,current = loss.item(), idx*len(data)
               #    print("Loss: {loss:.5f} Durchlauf: {current:}".format(loss=loss,current=current))

    def test(self,test_dataloader):
        # Testen des Netzes
        correct = 0
        for idx, (data,label) in enumerate(test_dataloader):
    
            output = self(data)
            # Output ist zB [0,0,1,0,0,0,0,0,0,0] für 2 - umcodieren
            output = output.argmax()

            if output == label:
                correct += 1

        performance = correct/test_size
        print("#Testdaten: ", test_size, " correct: ", correct, " Performance: ", performance)
        return performance


# Daten importieren
class Data(Dataset):
    def __init__(self):
        xy = np.loadtxt("mnist_data_60k.csv", delimiter=",",dtype=np.float32,skiprows=1)
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,0])
        self.n_samples = xy.shape[0]

    def __getitem__(self,index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
dataset = Data()
train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset,[train_size,test_size])
train_dataloader = DataLoader(dataset=train_dataset,shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset,shuffle=True)
 

# Initialisierung des neuronalen Netzes
input_size = 784
output_size=10


print("Sigmoid")
modelSigmoid = NN(input_size,output_size,nn.Sigmoid)
modelSigmoid.train(train_dataloader)
modelSigmoid.test(test_dataloader)

print("ReLu")
modelReLu = NN(input_size,output_size,nn.ReLU)
modelReLu.train(train_dataloader)
modelReLu.test(test_dataloader)

print("Leaky ReLU mit 0.01")
LeakyReLU = nn.LeakyReLU
modelLeakyReLU001 = NN(input_size,output_size,LeakyReLU,paramAktFkt=0.01)
modelLeakyReLU001.train(train_dataloader)
modelLeakyReLU001.test(test_dataloader)

print("Leaky ReLU mit 0.05")
LeakyReLU = nn.LeakyReLU
modelLeakyReLU005 = NN(input_size,output_size,LeakyReLU,paramAktFkt=0.05)
modelLeakyReLU005.train(train_dataloader)
modelLeakyReLU005.test(test_dataloader)

print("Leaky ReLU mit 0.1")
LeakyReLU = nn.LeakyReLU
modelLeakyReLU01 = NN(input_size,output_size,LeakyReLU,paramAktFkt=0.1)
modelLeakyReLU01.train(train_dataloader)
modelLeakyReLU01.test(test_dataloader)

print("Leaky ReLU mit 0.5")
LeakyReLU = nn.LeakyReLU
modelLeakyReLU05 = NN(input_size,output_size,LeakyReLU,paramAktFkt=0.5)
modelLeakyReLU05.train(train_dataloader)
modelLeakyReLU05.test(test_dataloader)

print("PreLU")
prelu = nn.PReLU
modelprelu = NN(input_size,output_size,prelu)
modelprelu.train(train_dataloader)
modelprelu.test(test_dataloader)

print("ELU mit 0.1")
elu = nn.ELU
modelelu01 = NN(input_size,output_size,elu,0.1)
modelelu01.train(train_dataloader)
modelelu01.test(test_dataloader)

print("ELU mit 0.2")
elu = nn.ELU
modelelu02 = NN(input_size,output_size,elu,0.2)
modelelu02.train(train_dataloader)
modelelu02.test(test_dataloader)

print("ELU mit 0.3")
elu = nn.ELU
modelelu03 = NN(input_size,output_size,elu,0.3)
modelelu03.train(train_dataloader)
modelelu03.test(test_dataloader)