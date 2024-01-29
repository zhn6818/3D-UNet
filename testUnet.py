import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Unet3d_model import UNet3DModel
from datasetUnet import Custom
from datasetSg import CustomSg
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BCE_WEIGHTS, BACKGROUND_AS_CLASS, TRAIN_CUDA
)
import numpy as np
import time
import cv2

if __name__ == "__main__":


    model = UNet3DModel(in_channels=3, num_classes=1)

    if torch.cuda.is_available() and TRAIN_CUDA:
        model = model.cuda()

    model.load_state_dict(torch.load("./checkpoints/epoch1042_train_loss0.04887127736583352.pth"), strict=True)
    
    model.eval()
    
    train_dataset = CustomSg(10)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=1,drop_last=False,pin_memory=True)
    
    with torch.no_grad():
        
        for data in train_dataloader:
            image, ground_truth = data
            image = image.cuda()
            torch.cuda.synchronize()
            start = time.time()
            logits = model(image)
            end = time.time()
            print("inference time:%f s" % (end - start))
            torch.cuda.synchronize()
            logits = torch.squeeze(logits)
            
            
            logitsr = (logits[0,:,:]>logits[1,:,:]).to(torch.float)
            
            logitsr = logitsr.cpu().numpy() * 255
            
            cv2.imwrite("./test.png",logitsr)
            
            print("hello")
            # logits[logits>0.5] = 1
            # logits[logits<=0.5] = 0
            # pred = logits
            # pred = torch.argmax(logits, dim=0)
            # pred = torch.squeeze(pred,0)
            # pred = pred.cpu().numpy()
            # pred = np.asarray(pred,dtype=np.uint8)
            # imgshow[pred == 1,:] = 0  
            # cv2.imwrite(savefolder+'/'+imgname,imgshow)
            