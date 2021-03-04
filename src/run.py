from attensat import *
import pickle
import cv2
import time
import numpy as np
import random

class iterdata(torch.utils.data.IterableDataset):

    def __init__(self,start,end):
        super(iterdata).__init__()
        assert end > start, "end < start"
        self.start = start 
        self.end = end
        self.videos = pickle.load(open("../saved/2FG-V.pkl","rb"))
        self.labels = pickle.load(open("../saved/2FG-L.pkl","rb"))

        self._preproces()

        ic(len(self.images))
        ic(len(self.imglabels))

    def _preproces(self):
        self.images = []
        frames_count = 0
        self.imglabels = []
        lab_dic = ['teacher', 'name', 'word', 'eraser', 'result', 'memorize', 'pen', 'scale', 'paper', 'principal', 'student', 'exam', 'blackboard', 'pass', 'picture', 'education', 'college', 'university', 'pencil', 'title', 'file', 'book', 'fail', 'sentence', 'classroom'] 
        for i in range(len(self.videos)):
            v = self.videos[i]
            l = self.labels[i]
            for f in v:
                if np.count_nonzero(f) > 300:
                    self.images.append(f)
                    self.imglabels.append(lab_dic.index(l))
                frames_count+=1
        self.images = np.array(self.images)
        indices = list(range(len(self.images)))
        np.random.shuffle(indices) 
        ic(indices[0])
        ic(self.imglabels[0])
        shuffle_images = np.array(self.shuffle(self.images,indices))
        shuffle_labels = np.array(self.shuffle(self.imglabels,indices))
        self.X = shuffle_images[:8*len(shuffle_images)//10] 
        self.Y = shuffle_labels[:8*len(shuffle_images)//10] 
        self.VX = shuffle_images[8*len(shuffle_images)//10:] 
        self.VY = shuffle_labels[8*len(shuffle_images)//10:] 
        self.X = torch.tensor(self.X,dtype=torch.float32)
        self.VX = torch.tensor(self.VX,dtype=torch.float32)
        self.Y =  torch.tensor(self.Y,dtype=torch.long) 
        self.VY =  torch.tensor(self.VY,dtype=torch.long) 


        
        ic(self.X.shape,self.VX.shape) 

    def shuffle(self,x,indices):
        r = []
        for i in range(len(indices)):
            r.append(x[indices[i]])
        return r
    def __iter__(self):
        return iter(self.images[self.start:self.end])

    def __len__(self):
        return len(self.images)

def train(model,epochs):
    it = iterdata(0,24971)

    optim = torch.optim.SGD(model.parameters(),lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    h_loss = []
    h_acc = []
    vh_loss = []
    vh_acc = []
    
    for e in range(epochs):
        avg_acc = 0.0
        avg_loss = 0.0
        c = 0
        model.train()
        b_time = time.time()
        for i in range(0,len(it.X),100):
            X = it.X[i:i+100]
            Y = it.Y[i:i+100]
            y = model(X)
            ic(y.shape,Y.shape)
            loss = loss_fn(y,Y)
            avg_loss += loss.item()
            loss.backward()
            optim.step()
            optim.zero_grad()
            accuracy = (torch.argmax(y,-1)==Y).sum().float()/X.shape[0]
            avg_acc += accuracy
            
            c+=1
            print(y)
            break
        ic("Time for one batch: ",time.time()-b_time)
        avg_loss = avg_loss/c 
        avg_acc = avg_acc/c
        ic(avg_acc,avg_loss)
        h_acc.append(avg_acc) 
        h_loss.append(avg_loss)

        if e%4 == 0:
            avg_acc = 0.0
            avg_loss = 0.0
            c = 0
            model.eval()
            for i in range(0,len(it.VX),100):
                VX = it.VX[i:i+100]
                VY = it.VY[i:i+100]
                y = model(VX)
                # ic(y.shape,VY.shape)
                loss = loss_fn(y,VY)
                avg_loss += loss.item()
                accuracy = (torch.argmax(y,-1)==VY).sum().float()/VX.shape[0]
                avg_acc += accuracy
                c+=1
                print(y) 
                break
            avg_loss = avg_loss/c 
            avg_acc = avg_acc/c
            
            ic(avg_acc,avg_loss)
            vh_acc.append(avg_acc) 
            vh_loss.append(avg_loss)
        ic(e)
    return model

                
if __name__ == '__main__':
    it = iterdata(0,100)
    ds = torch.utils.data.DataLoader(it,num_workers=0)
    attensat = Attensat()
    train(attensat,10)

    # u = torch.tensor(list(range(128*128*10)),dtype=torch.float32)
    # u = u.reshape((10,128,128))
    # attensat = Attensat()
    # # conv = ConvModel()
    # import time
    # x = time.time()
    # result = attensat(u)
    # print(time.time()-x)
    # print(result.shape)


    
    # result1 = conv(u.reshape((1,1,128,128)))
    # print(result1.shape)

