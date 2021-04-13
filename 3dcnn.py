import numpy as np
import torch
import torch.nn as nn
import os, cv2, sys
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
sys.path.append('../')
from resnetpytorch.models import resnet
from torchvision import transforms

def create_clips(cap, index):
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter('/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/test_BookShelf_S1/clips/output_' + str(index) + '.avi', fourcc, 5.0, (512,384), 0)
    for i in range(5):
        _, frame = cap.read()
        frame = cv2.resize(frame, (512, 384))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        out.write(frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES,150+index-4)

    out.release()

if __name__ == '__main__':
    df = pd.read_csv('/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/train_BookShelf_S1/gaze_file.csv').to_numpy()
    # file3d = '/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/training_images/output_0.avi'
    # cap = cv2.VideoCapture(file3d)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(frame_count)
    # model = resnet.generate_model(50)
    # print(model)
    # transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # last = None
    # _, frame = cap.read()
    # # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #
    # last = frame
    # print(last.shape)
    # last = transforms(last)
    # print(torch.max(last[0]), torch.min(last[0]))
    # last = last.unsqueeze(dim=3)
    # print(last.shape)
    # for i in range(4):
    #     print(i)
    #     _, frame = cap.read()
    #     # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #     # frame = np.expand_dims(frame, axis=2)
    #     frame = transforms(frame)
    #     frame = frame.unsqueeze(dim=3)
    #     last = torch.cat((last, frame), axis=3)
    #     # cv2.imshow('img', frame)
    #     # cv2.waitKey(0)
    #
    # last = last.unsqueeze(dim=0)
    # print(last.shape)
    #
    # x = model(last)
    # print(x.shape)

    for i in range(10, 6, -1):
        print(i)
