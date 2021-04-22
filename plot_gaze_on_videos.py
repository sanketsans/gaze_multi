import cv2, os, sys
import numpy as np
from tqdm import tqdm
import pandas as pd
from ast import literal_eval
sys.path.append('../')
from loader import JSON_LOADER
from variables import RootVariables

def get_num_correct(pred, label):
    return np.logical_and((np.abs(pred[0] - label[0]) <= 100.0), (np.abs(pred[1]-label[1]) <= 100.0)).sum().item()
    # return torch.logical_and((torch.abs(pred[:,0]*1920-label[:,0]*1920) <= 100.0), (torch.abs(pred[:,1]*1080-label[:,1]*1080) <= 100.0)).sum().item()

if __name__ == "__main__":
    var = RootVariables()
    folder = 'train_Supermarket_S1/'
    uni = None
    os.chdir(var.root + folder)
    cap = cv2.VideoCapture('scenevideo.mp4')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # dataset = JSON_LOADER(folder)
    # dataset.POP_GAZE_DATA(frame_count)
    # gaze_arr = np.array(dataset.var.gaze_data).transpose()
    # temp = np.zeros((frame_count*4-var.trim_frame_size*4*2 - 4, 2))
    # temp[:,0] = gaze_arr[tuple([np.arange(var.trim_frame_size*4 +4, frame_count*4 - var.trim_frame_size*4), [0]])]
    # temp[:,1] = gaze_arr[tuple([np.arange(var.trim_frame_size*4 +4, frame_count*4 - var.trim_frame_size*4), [1]])]

    df = pd.read_csv('/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/train_Outdoor_S1/gaze_file.csv').to_numpy()
    correct, total_pts = 0, 0
    trim_size = 150
    cap.set(cv2.CAP_PROP_POS_FRAMES,trim_size)
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts = [0.5, 0.5]
        try:
            gpts = list(map(literal_eval, df[trim_size+i, 1:]))
            avg = [sum(y) / len(y) for y in zip(*gpts)]

            # start_point = (int(pts[0]*frame.shape[1]) - 100, int(pts[1]*frame.shape[0]) + 100)
            # end_point = (int(pts[0]*frame.shape[1]) + 100, int(pts[1]*frame.shape[0]) - 100)
            # pred_start_point = (int(avg[0]*frame.shape[1]) - 100, int(avg[1]*frame.shape[0]) + 100)
            # pred_end_point = (int(avg[0]*frame.shape[1]) + 100, int(avg[1]*frame.shape[0]) - 100)

            # frame = cv2.circle(frame, (int(pts[0]*1920),int(pts[1]*1080)), radius=5, color=(0, 0, 255), thickness=5)
            frame = cv2.circle(frame, (int(avg[0]*1920),int(avg[1]*1080)), radius=5, color=(0, 255, 0), thickness=5)

            # frame = cv2.rectangle(frame, start_point, end_point, color=(0, 0, 255), thickness=5)
            # frame = cv2.rectangle(frame, pred_start_point, pred_end_point, color=(0, 255, 0), thickness=5)
            correct += get_num_correct([a*b for a,b in zip(avg,[1920.0, 1080.0])], [a*b for a,b in zip(pts,[1920.0, 1080.0])])
            total_pts += 1

        except Exception as e:
            pass
        cv2.imshow('image', frame)
        cv2.waitKey(0)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    print('accuracy: ', 100.0*(correct / total_pts))
