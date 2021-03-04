from torch import nn
from icecream import ic

import matplotlib.pyplot as plt

import pickle
import torch
import csv
import os
import cv2
import numpy as np
from icecream import ic
from model import *


processed_videos = pickle.load(open("C:\\Users\\admin\\PycharmProjects\\gesres\\saved\\par_videos.pkl","rb"))
labels = pickle.load(open("C:\\Users\\admin\\PycharmProjects\\gesres\\saved\\par_labels.pkl","rb"))

#eval_videos = pickle.load(open("C:\\Users\\admin\\PycharmProjects\\gesres\\saved\\.pkl","rb"))
#eval_labels = pickle.load(open("C:\\Users\\admin\\PycharmProjects\\gesres\\saved\\evallabels.pkl","rb"))

def transformY(y,ey):
    av = list(set(labels))
    Y = []
    val_Y = []
    for x in y:
        Y.append(av.index(x))
    for x in ey:
        val_Y.append(av.index(x))
    return np.array(Y),np.array(val_Y)

Y,vY = transformY(labels,eval_labels)
X = np.array(processed_videos)
vX = np.array(eval_videos)
X = X[:,np.newaxis,:,:,:]
vX = vX[:,np.newaxis,:,:,:]
ic(X.shape,Y.shape,vX.shape,vY.shape)


X = torch.tensor(X,dtype=torch.float32)
Y = torch.tensor(Y,dtype=torch.int64)
vX = torch.tensor(vX,dtype=torch.float32)
vY = torch.tensor(vY,dtype=torch.int64)
ic(X.shape)

model = Model(25)
#

if torch.cuda.is_available():
    model = model.cuda()
    X = X.cuda()
    Y = Y.cuda()
    vX = vX.cuda()
    vY = vY.cuda()


optim = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.CrossEntropyLoss()
h_loss = []
h_acc = []
vh_loss = []
vh_acc = []

for epoch in range(200):
    model.train()
    avg_acc = 0.0
    avg_loss = 0.0
    c = 0
    bs =1

    for i in range(0, len(X), bs):
        XX = X[i:i + bs]
        y = model(XX)
        loss = loss_fn(y, Y[i:i + bs])
        avg_loss += loss.item()
        loss.backward()
        optim.step()
        optim.zero_grad()
        accuracy = (torch.argmax(y, -1) == Y[i:i + bs]).sum().float() / XX.shape[0]
        avg_acc += accuracy
        c += 1

    if epoch % 4 == 0:
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        vc = 0

        for i in range(0, len(vX), bs):
            vXX = vX[i:i + bs]
            vy = model(vXX)
            loss = loss_fn(vy, vY[i:i + bs])
            val_loss += loss.item()
            val_acc += (torch.argmax(vy, -1) == vY[i:i + bs]).sum().float() / vXX.shape[0]
            vc += 1
        val_loss /= vc
        val_acc /= vc
        ic(val_loss, val_acc)
        vh_loss.append(val_loss)
        vh_acc.append(val_acc)

        plt.plot(range(len(h_loss)), h_loss)
        plt.plot(range(0, len(vh_loss) * 4, 4), vh_loss)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.show()
        plt.plot(range(len(h_acc)), h_acc)
        plt.plot(range(0, len(vh_acc) * 4, 4), vh_acc)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.show()

    avg_acc /= c
    avg_loss /= c
    ic(epoch, avg_loss, avg_acc)
    h_loss.append(avg_loss)
    h_acc.append(avg_acc)


torch.save(model.state_dict(),"C:\\Users\\admin\\PycharmProjects\\gesres\\models\\tra_mod.pt")
# model(X[)
# rq = model(X)
# rq.shape