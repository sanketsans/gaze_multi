import sys, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
import argparse
from tqdm import tqdm
sys.path.append('../')
from variables import RootVariables
from helpers import Helpers
from models import All_Models
from create_dataset import All_Dataset
from torch.utils.tensorboard import SummaryWriter
#from skimage.transform import rotate
import random

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':
    var = RootVariables()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sepoch", type=int, default=0)
    # parser.add_argument('--sepoch', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--nepoch", type=int, default=15)
    # parser.add_argument("--tfolder", action='store', help='tensorboard_folder name')
    parser.add_argument("--reset_data", type=int)
    # parser.add_argument("--reset_tboard", type=boolean_string, default=True)
    parser.add_argument("--model", type=int, choices={0, 1, 2}, help="Model index number, 0 : Signal, 1: Vision, 2 : MultiModal ")
    parser.add_argument("--resnet_model", type=int, default=50, choices={18, 34, 50}, help="Resnet3d depth")
    args = parser.parse_args()

    lastFolder, newFolder = None, None
    All_Dataset = All_Dataset()
    models = All_Models()
    for index, subDir in enumerate(sorted(os.listdir(var.root))):
        if 'train_shahid' in subDir:
            newFolder = subDir
            os.chdir(var.root)

            test_folder = 'test_' + newFolder[6:]
            # _ = os.system('mv ' + newFolder + ' test_' + newFolder[6:])
            # if lastFolder is not None:
            #     print('Last folder changed')
            #     _ = os.system('mv test_' + lastFolder[6:] + ' ' + lastFolder)

            print(newFolder, lastFolder)
            trim_frame_size = 150
            utils = Helpers(test_folder)
            imu_training, imu_testing, training_target, testing_target = utils.load_datasets(args.reset_data, repeat=0)

            pipeline, model_checkpoint = models.get_model(args.model, args.resnet_model, test_folder)
            # pipeline.tensorboard_folder = args.tfolder
            optimizer = optim.Adam(pipeline.parameters(), lr=0.0015, amsgrad=True) #, momentum=0.9)
            lambda1 = lambda epoch: 0.95 ** epoch
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
            criterion = nn.MSELoss()
            best_test_loss = 1000.0
            if Path(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint).is_file():
                checkpoint = torch.load(pipeline.var.root + model_checkpoint)
                pipeline.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                best_test_acc = checkpoint['best_test_loss']
                # pipeline.current_loss = checkpoint['loss']
                print('Model loaded')


            os.chdir(pipeline.var.root)
            print(torch.cuda.device_count())
            best_test_acc = 0.0
            for epoch in tqdm(range(args.sepoch, args.nepoch), desc="epochs"):
                print("Epoch :", epoch)
                if epoch > 0:
                    utils = Helpers(test_folder)
                    imu_training, imu_testing, training_target, testing_target = utils.load_datasets(reset_dataset=0, repeat=1)

                # ttraining_target = np.copy(training_target)
                # timu_training = np.copy(imu_training)
                trainDataset = All_Dataset.get_dataset('trainImg', imu_training, 'heatmap_trainImg', args.model)
                trainLoader = torch.utils.data.DataLoader(trainDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=0)
                tqdm_trainLoader = tqdm(trainLoader)
                testDataset = All_Dataset.get_dataset('testImg', imu_testing,  'heatmap_testImg', args.model)
                testLoader = torch.utils.data.DataLoader(testDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=0)
                tqdm_testLoader = tqdm(testLoader)

                # if epoch == 0 and args.reset_tboard:
                #     # _ = os.system('mv runs new_backup')
                #     _ = os.system('rm -rf ' + pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)

                num_samples = 0
                total_loss, total_correct, total_accuracy = [], 0.0, 0.0
                pipeline.train()
                for batch_index, items in enumerate(tqdm_trainLoader):
                    if args.model == 2:
                        frame_feat, imu_feat, labels = items
                        pred = pipeline(frame_feat, imu_feat)
                    else:
                        feat, labels = items
                        preds = pipeline(feat.float())

                    num_samples += labels.size(0)
#                    print(pred, labels)
                    pred, labels = pipeline.get_original_coordinates(pred, labels)
                    loss = criterion(pred.float(), labels.float())
                    optimizer.zero_grad()
                    loss.backward()
                    ## add gradient clipping
#                    nn.utils.clip_grad_value_(pipeline.parameters(), clip_value=1.0)
                    optimizer.step()

                    with torch.no_grad():

                        # pred, labels = pipeline.get_original_coordinates(pred, labels)

#                        dist = torch.cdist(pred, labels.float(), p=2)[0].unsqueeze(dim=0)
#                        if batch_index > 0:
#                            trainPD = torch.cat((trainPD, dist), 1)
#                        else:
#                            trainPD = dist

                        total_loss.append(loss.detach().item())
                        total_correct += pipeline.get_num_correct(pred, labels.float())
                        total_accuracy = total_correct / num_samples
                        tqdm_trainLoader.set_description('training: ' + '_loss: {:.4} correct: {} accuracy: {:.3} lr:{}'.format(
                            np.mean(total_loss), total_correct, 100.0*total_accuracy, optimizer.param_groups[0]['lr']))

                if (epoch+1) % 20 == 0:
                    scheduler.step()

                pipeline.eval()
                with torch.no_grad():
                    # tb = SummaryWriter(pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)
                    # tb.add_scalar("Train Loss", np.mean(total_loss), epoch)
                    # tb.add_scalar("Training Correct", total_correct, epoch)
                    # tb.add_scalar("Train Accuracy", total_accuracy, epoch)

                    num_samples = 0
                    total_loss, total_correct, total_accuracy = [], 0.0, 0.0
                    dummy_correct, dummy_accuracy = 0.0, 0.0
                    for batch_index, items in enumerate(tqdm_testLoader):
                        dummy_pts = (torch.ones(8, 2) * 0.5).to(device)
                        dummy_pts[:,0] *= 1920
                        dummy_pts[:,1] *= 1080
                        if args.model == 2:
                            frame_feat, imu_feat, labels = items
                            pred = pipeline(frame_feat, imu_feat)
                        else:
                            feat, labels = items
                            pred = pipeline(feat.float())
                        num_samples += labels.size(0)
                        labels = labels[:,0,:]
                        pred, labels = pipeline.get_original_coordinates(pred, labels)

                        loss = criterion(pred.float(), labels.float())
                        # pred, labels = pipeline.get_original_coordinates(pred, labels)

#                        dist = torch.cdist(pred, labels.float(), p=2)[0].unsqueeze(dim=0)
#                        if batch_index > 0:
#                            testPD = torch.cat((testPD, dist), 1)
#                        else:
#                            testPD = dist

                        total_loss.append(loss.detach().item())
                        total_correct += pipeline.get_num_correct(pred, labels.float())
                        dummy_correct += pipeline.get_num_correct(dummy_pts.float(), labels.float())
                        dummy_accuracy = dummy_correct / num_samples
                        total_accuracy = total_correct / num_samples
                        tqdm_testLoader.set_description('testing: ' + '_loss: {:.4} correct: {} accuracy: {:.3} DAcc: {:.4}'.format(
                            np.mean(total_loss), total_correct, 100.0*total_accuracy,  np.floor(100.0*dummy_accuracy)))

                # tb.add_scalar("Testing Loss", np.mean(total_loss), epoch)
                # tb.add_scalar("Testing Correct", total_correct, epoch)
                # tb.add_scalar("Testing Accuracy", total_accuracy, epoch)
                # tb.add_scalar("Dummy Accuracy", np.floor(100.0*dummy_accuracy), epoch)
                # tb.close()

                # if total_accuracy >= best_test_acc:
                #     best_test_acc = total_accuracy
                #     torch.save({
                #                 'epoch': epoch,
                #                 'model_state_dict': pipeline.state_dict(),
                #                 'optimizer_state_dict': optimizer.state_dict(),
                #                 'best_test_acc': best_test_acc,
                #                 }, pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint)
                #     print('Model saved')


            lastFolder = newFolder

    # optimizer = optim.Adam([
    #                         {'params': imuModel.parameters(), 'lr': 1e-4},
    #                         {'params': frameModel.parameters(), 'lr': 1e-4},
    #                         {'params': temporalModel.parameters(), 'lr': 1e-4}
    #                         ], lr=1e-3)
