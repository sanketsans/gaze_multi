import cv2, os, sys
import numpy as np
from tqdm import tqdm
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
sys.path.append('../')
from loader import JSON_LOADER

def get_num_correct(pred, label):
    return np.logical_and((np.abs(pred[0] - label[0]) <= 100.0), (np.abs(pred[1]-label[1]) <= 100.0)).sum().item()
    # return torch.logical_and((torch.abs(pred[:,0]*1920-label[:,0]*1920) <= 100.0), (torch.abs(pred[:,1]*1080-label[:,1]*1080) <= 100.0)).sum().item()

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = 20 #min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def load_heatmap(img, channels, index):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[:,:,0]
    h, w = img.shape
    y1 = np.zeros((h, w, channels))
    padding = 40
    mask = None
    trim_size = 50
    # x, y, = int(coordinates[0]*512), int(coordinates[1]*288)
    for j in range(5):
        y2 = np.copy(y1)
        try:
            gpts = list(map(literal_eval, df[trim_size-4+index+j, 1:]))
        except Exception as e:
            print(e)

        telem = 4
        for item in gpts:
            if (item[0] + item[1]) == 0:
                telem -= 1

        if telem > 0:
            coordinates = [sum(y) / telem for y in zip(*gpts)]
            center = (int(coordinates[0]*512), int(coordinates[1]*288))
            if j > 0 and coordinates[0] != 0.0:
                mask += create_circular_mask(h, w, center)
            else:
                mask = create_circular_mask(h, w, center)

        if mask is None:
            mask = create_circular_mask(h, w, radius=0)

        heatmap = np.zeros(img.shape)
        heatmap[mask] = 1.0
        if heatmap.sum() > 0 :
            y2[:, :, 0] = decay_heatmap(heatmap, sigma2=30)

        y1 += y2

    return y1

#     for joint_id in range(1, n_joints + 1):
#         heatmap = np.zeros(joints.shape)
#         heatmap[mask] = 1.0
# #         heatmap = (joints == joint_id).astype('float')
#         if heatmap.sum() > 0:
#             y1[:, :, joint_id - 1] = decay_heatmap(heatmap)
#     return y1

def decay_heatmap(heatmap, sigma2=10):
    """
    Args
        heatmap :
           WxH matrix to decay
        sigma2 :
             (Default value = 1)
    Returns
        Heatmap obtained by gaussian-blurring the input
    """
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma2)
    heatmap /= np.sum(heatmap)
    # heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    # heatmap /= np.max(heatmap)  # keep the max to 1
    return heatmap


if __name__ == "__main__":
    from PIL import Image
    import pathlib
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from variables import RootVariables

    var = RootVariables()
    folder = 'train_sanket_washands_S1/'
    uni = None
    os.chdir(var.root + folder)
    cap = cv2.VideoCapture('scenevideo.mp4')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    df = pd.read_csv(var.root + folder + 'gaze_file.csv').to_numpy()
    for i in range(len(df)):
        try:
            _ = (list(map(literal_eval, df[i, 1:])))
        except:
            indexes = []
            for j in range(1, len(df[i])):
                if 'nan' in df[i][j]:
                    df[i][j] = '(0.0, 0.0)'

    cap.set(cv2.CAP_PROP_POS_FRAMES,var.trim_frame_size)
    folder = var.root + 'testing_images/'
    act = nn.Softmax2d()
    currentDirectory = pathlib.Path('')
    for index in range(2, 3000):
        img = cv2.imread('/data/Tobii_dataset/training_images/train_sanket_washands_S1/image_' + str(index) + '.jpg')
        heat = cv2.imread('/data/Tobii_dataset/heatmap_training_images/train_sanket_washands_S1/image_' + str(index-2) + '.jpg')
        img = cv2.addWeighted(heat, 0.3, img, 0.7, 0)
        print(heat.shape)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # for i in tqdm(range(frame_count-300)):
    #     ret, frame = cap.read()
    #     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     pts = [0.5, 0.5]
    #     heatmapshow, y, heatmap = None, None, None
    #     try:
    #         frame = cv2.resize(frame, (512, 288))
    #         x = load_heatmap(frame, 1, i+5)
    #         heatmapshow = cv2.normalize(x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #         heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    #         # heatmap = transforms.ToTensor()(heatmapshow)
    #         # heatmap = heatmap.unsqueeze(dim=0)
    #         # heatmap = act(heatmap)
    #         # heatmap = heatmap.squeeze(dim=0)
    #         # heatmap = heatmap.permute(1, 2, 0)
    #         # heatmap = heatmap.cpu().numpy()
    #         # print(torch.argmax(tx, dim=1), torch.argmax(tx, dim=2))
    #         # coo = list(map(literal_eval, df[i, 1:]))
    #         # avg = [sum(y) / len(y) for y in zip(*coo)]
    #         frame = cv2.addWeighted(heatmapshow, 0.3, frame, 0.7, 0)
    #
    #
    #         # start_point = (int(pts[0]*frame.shape[1]) - 100, int(pts[1]*frame.shape[0]) + 100)
    #         # end_point = (int(pts[0]*frame.shape[1]) + 100, int(pts[1]*frame.shape[0]) - 100)
    #         # pred_start_point = (int(avg[0]*frame.shape[1]) - 100, int(avg[1]*frame.shape[0]) + 100)
    #         # pred_end_point = (int(avg[0]*frame.shape[1]) + 100, int(avg[1]*frame.shape[0]) - 100)
    #
    #         # frame = cv2.circle(frame, (int(pts[0]*1920),int(pts[1]*1080)), radius=5, color=(0, 0, 255), thickness=5)
    #         # frame = cv2.circle(frame, (int(avg[0]*512),int(avg[1]*288)), radius=5, color=(0, 255, 0), thickness=5)
    #         # frame = cv2.addWeighted(heatmapshow, 0.6, frame, 0.3, 0)
    #         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         # cv2.imwrite(folder + 'image_' + str(i) + '.jpg', frame)
    #
    #         # frame = cv2.rectangle(frame, start_point, end_point, color=(0, 0, 255), thickness=5)
    #         # frame = cv2.rectangle(frame, pred_start_point, pred_end_point, color=(0, 255, 0), thickness=5)
    #         # correct += get_num_correct([a*b for a,b in zip(avg,[1920.0, 1080.0])], [a*b for a,b in zip(pts,[1920.0, 1080.0])])
    #         # total_pts += 1
    #
    #     except Exception as e:
    #         print(e)
    #     cv2.imshow('image', frame)
    #     # cv2.waitKey(0)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # print('accuracy: ', 100.0*(correct / total_pts))
