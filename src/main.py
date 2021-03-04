# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pickle
import torch
import csv
import os
import cv2
import numpy as np
from icecream import ic


print(torch.cuda.is_available())
dataset = []
labels = []
processed_videos = []


def get_data(data_loca):
    # for folder in os.listdir(data_loc):
    #     if folder[0] != '.':
    seen = []
    prev_tag = None
    not_worked =  0
    for file in os.listdir(data_loca):
        if file[0] == ".":
            continue

        cap = cv2.VideoCapture(data_loca + "/" + file)
        if "Copy" in file:
            name = file.split(" ")[2].split("_")
            if name[0] == 'v':
                tag = name[1].lower()
            else:
                tag = name[0].lower()
            # ic(file)
        else:
            name = file.split("_")
            if name[0] == 'v':
                tag = name[1].lower()
            else:
                tag = name[0].lower()

        print(tag)

        w = 128
        fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        suc, frame = cap.read()
        if not suc:
            continue
        pframe = cv2.resize(frame, (w, w), 0, 0, cv2.INTER_CUBIC)
        pframe = cv2.cvtColor(pframe, cv2.COLOR_BGR2GRAY)
        all_frames = []

        for i in range(1, fcount):
            x, frame = cap.read()
            if not x:
                continue
            frame = cv2.resize(frame, (w, w), 0, 0, cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(frame, pframe)
            diff[diff < 30] = 0
            count = np.count_nonzero(diff)
            if count > 10:
                # cv2.imshow('frame',diff)
                # if cv2.waitKey(1) & 0xFF == ord('q'): break
                all_frames.append(diff)
            pframe = frame
        # print(len(all_frames))
        reduced_frames = []
        req = 40

        i = 0.0
        if len(all_frames) > 20:
            while i < len(all_frames) - len(all_frames) / req:
                reduced_frames.append(all_frames[int(i)])
                # cv2.imshow('frame',reduced_frames[-1])
                # if cv2.waitKey(1) & 0xFF == ord('q'): break
                i += len(all_frames) / req

            while len(reduced_frames) < req:
                reduced_frames.append(reduced_frames[-1])
            # print("Red ",len(reduced_frames))
            ic(len(reduced_frames))
            labels.append(tag)
            seen.append(tag)
            processed_videos.append(reduced_frames)
        else:
            not_worked+=1
    return not_worked

w = 0
w += get_data("C:\\Data\\ram dataset")
ic(w)
#
# pickle.dump(processed_videos, open("C:\\Users\\student\\PycharmProjects\\gesres\\saved\\transvideos.pkl","wb"))
# pickle.dump(labels,open("C:\\Users\\student\\PycharmProjects\\gesres\\saved\\translabels.pkl","wb"))
#
w+= get_data("C:\\Data\\Dataset")
w+= get_data("C:\\Data\\My Dataset")
# #
# ic(np.array(processed_videos).shape)
#
# pickle.dump(processed_videos, open("C:\\Users\\student\\PycharmProjects\\gesres\\saved\\transvideos.pkl","wb"))
# pickle.dump(labels,open("C:\\Users\\student\\PycharmProjects\\gesres\\saved\\translabels.pkl","wb"))
#

w+= get_data("C:\\Data\\tf_dataset")
print(len(processed_videos))
w+= get_data("C:\\Data\\Nanthak Dataset")
ic("Not worked",w)

pickle.dump(processed_videos, open("C:\\Users\\admin\\PycharmProjects\\gesres\\saved\\par_videos.pkl","wb"))
pickle.dump(labels,open("C:\\Users\\admin\\PycharmProjects\\gesres\\saved\\par_labels.pkl","wb"))
#
#
# pickle.dump(processed_videos, open("C:\\Users\\student\\PycharmProjects\\gesres\\saved\\evalvideos.pkl","wb"))
# pickle.dump(labels,open("C:\\Users\\student\\PycharmProjects\\gesres\\saved\\evallabels.pkl","wb"))
#
#
# processed_videos = pickle.load(open("C:\\Users\\student\\PycharmProjects\\gesres\\saved\\transvideos.pkl","rb"))
# labels = pickle.load(open("C:\\Users\\student\\PycharmProjects\\gesres\\saved\\translabels.pkl","rb"))
#
# eval_videos = pickle.load(open("C:\\Users\\student\\PycharmProjects\\gesres\\saved\\evalvideos.pkl","rb"))
# eval_labels = pickle.load(open("C:\\Users\\student\\PycharmProjects\\gesres\\saved\\evallabels.pkl","rb"))
#
# videos = np.array(processed_videos)
#
# print(set(eval_labels))
#
# print(len(set(labels)))
# print(set(labels))
# print(videos.shape)
#
#
# def transformY(y,ey):
#     av = list(set(labels))
#     Y = []
#     val_Y = []
#     for x in y:
#         Y.append(av.index(x))
#     for x in ey:
#         val_Y.append(av.index(x))
#     print(y)
#     print(Y)
#     print(ey)
#     print(val_Y)
#     return np.array(Y),np.array(val_Y)
#
# Y,vY = transformY(labels,eval_labels)
# X = np.array(processed_videos)
# vX = np.array(eval_videos)
# X = X[:,np.newaxis,:,:,:]
# vX = vX[:,np.newaxis,:,:,:]
# print(X.shape,Y.shape,vX.shape,vY.shape)
#
# X = torch.tensor(X,dtype=torch.float32)
# Y = torch.tensor(Y,dtype=torch.int64)
# vX = torch.tensor(vX,dtype=torch.float32)
# vY = torch.tensor(vY,dtype=torch.int64)
# ic(X.shape)
#
#
#
print(np.array(processed_videos).shape)