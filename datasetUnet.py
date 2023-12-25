import cv2
import os
import numpy  as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import random
from PIL import Image
#cropsize = (256,256)
cropsize = (512,512)

from torch.utils.tensorboard import SummaryWriter



class Custom(Dataset):
    def __init__(self, numSamples,cropsize=cropsize, *args, **kwargs):
        super(Custom, self).__init__(*args, **kwargs)
 
        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.num = numSamples
        self.indice = 0
        self.slide = 32


    def __getitem__(self, index):

   
        # writer = SummaryWriter("./log/boardtest/")
        
        self.indice = self.indice + 1
        
        img = torch.ones([1, 3, 16, 256, 256], dtype=torch.float32)
        img *= 128
        
        label = torch.zeros([1, 1, 1, 256, 256], dtype=torch.float32)
        
        intdepth = img.shape[2]
        imgw = img.shape[3]
        imgh = img.shape[4]
        
        rand_num = random.randint(0, intdepth - 1)
        
        newValue = torch.ones([1, 3, self.slide, self.slide], dtype=torch.float64) * 200
        
        rand_w = random.randint(0, imgw - self.slide - 1)
        rand_H = random.randint(0, imgh - self.slide - 1)
        
        img[:,:,rand_num,rand_w:(rand_w + self.slide),rand_H:(rand_H + self.slide)] = newValue
        
        label[:,:,:,rand_w:(rand_w + self.slide),rand_H:(rand_H + self.slide)] = torch.ones([1,1,1,self.slide, self.slide], dtype = torch.float32)
        
        # for ii in range(intdepth):
        #     writer.add_image("teestImg"+ str(self.indice), img[0,:,ii,:,:],ii)
        # writer.add_image("testLabel" + str(self.indice), label[0,0,:,:,:])
        # writer.close()
        img = torch.squeeze(img, 0)
        label = torch.squeeze(label, 0)
        return img, label


    def __len__(self):
        return self.num
    
if __name__ == "__main__":
    
    
   
    train_dataset = Custom(1000)
    trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    drop_last=False,
    pin_memory=True)
    
    for step, data in enumerate(trainloader):
        inputs, labels = data
        
    