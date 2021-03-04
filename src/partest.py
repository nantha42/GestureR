from torch import nn
from icecream import ic
import random
import matplotlib.pyplot as plt
import time
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
print(np.array(processed_videos).shape)
print(len(labels))
print(len(set(labels)))
random.seed(1)
np.random.seed(1)

def shuffle(videos,labels):
    inds=  list(range(len(videos)))
    print(inds)
    random.shuffle(inds)
    shuff_videos = []
    shuff_labels = []
    for i in inds:
        shuff_videos.append(videos[i])
        shuff_labels.append(labels[i])
    return shuff_videos,shuff_labels


processed_videos, labels = shuffle(processed_videos,labels)
ic("Shuffled ",labels[:20])


def transformY(y):
    av = list(set(labels))
    keys_values = str(av)

    with open("label_index.txt","w") as f:
        f.write(keys_values)
        f.close()
    Y = []
    for x in y:
        Y.append(av.index(x))
    return np.array(Y)

all_labels= transformY(labels)
ic("Shuffled ",labels[:20])

all = np.array(processed_videos)
all = all[:,np.newaxis,:,:,:]

all = all[:1104]
all_labels = all_labels[:1104]
ic("index ",all_labels[:20])
k = 8
ic(all.shape[0])
unit = all.shape[0]//8
for fold in range(1,k):
    onefold = time.time()

    if fold ==0:
        vX = all[:unit]
        vY = all_labels[:unit]
        ic("Validation 0:",unit)
    else:
        vX = all[(fold) * unit:(fold + 1) * unit]
        vY = all_labels[(fold) * unit:(fold + 1) * unit]

        ic("Validation :",fold* unit," : ", (fold+1)*unit)

    if fold == 0:
        X = all[unit:]
        Y = all_labels[unit:]
    else:
        X1 = all[:(fold)*unit]
        X2 = all[(fold+1)*unit:]


        Y1 = all_labels[:(fold)*unit]
        Y2 = all_labels[(fold+1)*unit: ]
        ic("Training  0 : ",(fold)*unit," ",(fold+1)*unit," : ")
        X = np.concatenate((X1,X2),axis = 0)
        print(Y1.shape,Y2.shape)
        Y = np.concatenate((Y1,Y2))

    ic(X.shape,Y.shape,vX.shape,vY.shape)


    X = torch.tensor(X,dtype=torch.float32)
    Y = torch.tensor(Y,dtype=torch.int64)
    vX = torch.tensor(vX,dtype=torch.float32)
    vY = torch.tensor(vY,dtype=torch.int64)
    ic(X.shape)


    model = Model(25)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()
    h_loss = []
    h_acc = []
    vh_loss = []
    vh_acc = []

    for epoch in range(25):
        e_start = time.time()
        model.train()
        avg_acc = 0.0
        avg_loss = 0.0
        c = 0
        bs = 46
        for i in range(0, len(X), bs):
            print("="*(i//bs)+">",i,"/",len(X))
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


            plt.clf()
            plt.plot(range(len(h_loss)), h_loss)
            plt.plot(range(0, len(vh_loss) * 4, 4), vh_loss)
            plt.xlabel("epochs")
            plt.ylabel("loss")
            # plt.show()
            plt.savefig("C:\\Users\\student\\PycharmProjects\\gesres\\plots\\loss_"+str(fold)+".png")
            plt.clf()
            plt.plot(range(len(h_acc)), h_acc)
            plt.plot(range(0, len(vh_acc) * 4, 4), vh_acc)
            plt.xlabel("epochs")
            plt.ylabel("accuracy")
            #plt.show()
            plt.savefig("C:\\Users\\student\\PycharmProjects\\gesres\\plots\\acc_" + str(fold) + ".png")
            plt.clf()

        ic("Field : ",fold, " Epoch: ",epoch," Elapsed time: ",(time.time()-e_start)/60)
        avg_acc /= c
        avg_loss /= c
        ic(epoch, avg_loss, avg_acc)
        h_loss.append(avg_loss)
        h_acc.append(avg_acc)
    ic("Finished 1 fold in ",(time.time()- onefold)/60)
    torch.save(model.state_dict(), "C:\\Users\\student\\PycharmProjects\\gesres\\models\\model_"+str(fold)+".pth")

    #
#eval_videos = pickle.load(open("C:\\Users\\student\\PycharmProjects\\gesres\\saved\\evalvideos.pkl","rb"))
#eval_labels = pickle.load(open("C:\\Users\\student\\PycharmProjects\\gesres\\saved\\evallabels.pkl","rb"))
#
#videos = np.array(processed_videos)
#ic(videos.shape )
#ic(type(processed_videos),type(eval_videos))
#processed_videos.extend(eval_videos)
#ic(np.array(processed_videos).shape)