import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

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
from collections import deque

modelSize = 224
cropsize = (modelSize,modelSize)

class MyData:
    
    def __init__(self, size = 16):
        self.length = size
        self.dq = deque(maxlen=self.length)
    
    def putimg(self, img):
        print(img.shape)
        img = cv2.resize(img, cropsize)
        img = img / 255.0
        img = torch.tensor(img)
        img = img.permute(2, 0, 1)
        self.dq.append(img)
    
    def GetTensor(self, imgIn):
        
        self.putimg(imgIn)
        
        img = torch.zeros([3, self.length, modelSize, modelSize], dtype=torch.float32)
        
        if len(self.dq) < self.length:
            return img
        
        for i in range(self.length):
            img[:, i, :, :] = self.dq[i]
        
        return img
    

if __name__ == "__main__":


    model = UNet3DModel(in_channels=3, num_classes=1)

    if torch.cuda.is_available() and TRAIN_CUDA:
        model = model.cuda()

    model.load_state_dict(torch.load("./checkpoints/epoch9_train_loss0.5090109063312411.pth"), strict=True)
    
    model.eval()
    
    video = cv2.VideoCapture('/data1/zhn/fujian/shadow3.mp4')
 
    # 创建输出视频对象
    output_file = '/data1/zhn/fujian/result/shadow3_result.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = video.get(cv2.CAP_PROP_FPS)
    # video.get(cv2.CAP_PROP_FRAME_WIDTH)
    # video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) * 2)
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height), True)
    
    framecount = 0
    
    mydata = MyData()
    
    while True:
        # 从视频中读取每一帧图像
        ret, frame = video.read()
        
        if not ret:
            break
            
        # 在这里进行需要的处理操作（如果有）
        framecount = framecount + 1
        
        print("currentFrame: ", framecount)
        frame2 = np.copy(frame)
        tensorIn = mydata.GetTensor(frame2)
        tensorIn = tensorIn.cuda()
        tensorIn = tensorIn.unsqueeze(dim=0)
        logits = model(tensorIn)
        logits = torch.squeeze(logits)
        
        logitsr = (logits[0,:,:]>logits[1,:,:]).to(torch.float)
            
        logitsr = logitsr.cpu().numpy() * 255
        logitsr = logitsr[:, :,np.newaxis]
        newout = np.concatenate((logitsr, logitsr, logitsr), axis=2)
        newout = newout.astype(np.uint8)
        newout = cv2.resize(newout, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), frame_height))
        newout = np.concatenate((frame, newout), axis=1)
        # 将当前帧写入到输出视频文件中
        out.write(newout)
    
    # 关闭所有相关的对象
    video.release()
    out.release()
    cv2.destroyAllWindows()
        
    # train_dataset = CustomSg(10)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=1,drop_last=False,pin_memory=True)
    
    
    
    
    # with torch.no_grad():
        
    #     for data in train_dataloader:
    #         image, ground_truth = data
    #         image = image.cuda()
    #         torch.cuda.synchronize()
    #         start = time.time()
    #         logits = model(image)
    #         end = time.time()
    #         print("inference time:%f s" % (end - start))
    #         torch.cuda.synchronize()
    #         logits = torch.squeeze(logits)
            
            
    #         logitsr = (logits[0,:,:]>logits[1,:,:]).to(torch.float)
            
    #         logitsr = logitsr.cpu().numpy() * 255
            
    #         cv2.imwrite("./test.png",logitsr)
            
    #         print("hello")