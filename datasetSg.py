import cv2
import os
import numpy  as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import random
from PIL import Image
#cropsize = (256,256)


from torch.utils.tensorboard import SummaryWriter


modelSize = 128

cropsize = (modelSize,modelSize)


class CustomSg(Dataset):
    def __init__(self, numSamples,cropsize=cropsize, *args, **kwargs):
        super(CustomSg, self).__init__(*args, **kwargs)
 
        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.num = numSamples
        self.indice = 0
        self.k = 16
        self.slide=None
        self.numbers = list(range(1, 7499))
        # self.numbers_L = list(range(1, 7479))
        self.imgpath = "/data1/zhn/fujian/dataset/img/"
        self.labelpath = "/data1/zhn/fujian/dataset/label/"

    # def processImg(self, img, label, label2):
    #     intdepth = img.shape[2]
    #     imgw = img.shape[3]
    #     imgh = img.shape[4]
    #     rand_int = random.randint(0, 1)
    #     self.slide = self.slides[rand_int]
    #     rand_num = random.randint(0, intdepth - 1)
    #     newValue = torch.ones([1, 3, self.slide, self.slide], dtype=torch.float64) * random.randint(100, 250)
    #     rand_w = random.randint(0, imgw - self.slide - 1)
    #     rand_H = random.randint(0, imgh - self.slide - 1)
    #     img[:,:,rand_num,rand_w:(rand_w + self.slide),rand_H:(rand_H + self.slide)] = newValue
    #     label[:,:,:,rand_w:(rand_w + self.slide),rand_H:(rand_H + self.slide)] = torch.ones([1,1,1,self.slide, self.slide], dtype = torch.float32)
    #     label2[:,:,:,rand_w:(rand_w + self.slide),rand_H:(rand_H + self.slide)] = torch.zeros([1,1,1,self.slide, self.slide], dtype = torch.float32)
        

    def __getitem__(self, index):


        writer = SummaryWriter("./log/boardtest/")
        
        # result = random.sample(self.numbers, k = self.k)
        start = random.randint(1, 7400)
        end = start + 16
        result = self.numbers[start:end:1]
        self.indice = self.indice + 1
        img = torch.zeros([3, self.k, modelSize, modelSize], dtype=torch.float32)
        label = torch.zeros([1, 1, modelSize, modelSize], dtype=torch.float32)
        # label2 = torch.ones([1, 1, modelSize, modelSize], dtype=torch.float32)
        
        i = 0
        for ii in result:
            img_path = self.imgpath + str(ii) + ".png"
            gt_path = self.labelpath + str(ii) + ".png"
            imgreal = cv2.imread(img_path)
            imgreal = cv2.resize(imgreal, cropsize)
            imgreal = imgreal / 255.0
            imgreal = torch.tensor(imgreal)
            imgreal = imgreal.permute(2, 0, 1)
            writer.add_image("teestImg" + str(self.indice), imgreal,i)
            img[:, i, :, :] = imgreal
            labelreal = cv2.imread(gt_path)
            labelreal = cv2.resize(labelreal, cropsize)
            labelreal = cv2.cvtColor(labelreal,cv2.COLOR_BGR2GRAY)
            labelreal = labelreal / 255.0 
            labelreal[labelreal>0.5] = 1
            labelreal[labelreal<0.5] = 0
            # writer.add_image("label"+ str(i), labelreal,i)
            label[0, 0, :, :] = label[0, 0, :, :] + labelreal
            i = i + 1
            
        # writer.add_image("testLabel", label[0,:,:,:])
        # self.indice = self.indice + 1
        
        # value = random.randint(10, 90)
        
        # img = torch.ones([1, 3, 16, modelSize, modelSize], dtype=torch.float32)
        # img *= value
        
        # label = torch.zeros([1, 1, 1, modelSize, modelSize], dtype=torch.float32)
        # label2 = torch.ones([1, 1, 1, modelSize, modelSize], dtype=torch.float32)
        
        # for ii in range(5):
        #     self.processImg(img, label, label2)
        
        # intdepth = img.shape[2]
        # imgw = img.shape[3]
        # imgh = img.shape[4]
        
        # rand_int = random.randint(0, 1)
        # self.slide = self.slides[rand_int]
        
        # rand_num = random.randint(0, intdepth - 1)
        
        # newValue = torch.ones([1, 3, self.slide, self.slide], dtype=torch.float64) * 200
        
        # rand_w = random.randint(0, imgw - self.slide - 1)
        # rand_H = random.randint(0, imgh - self.slide - 1)
        
        # img[:,:,rand_num,rand_w:(rand_w + self.slide),rand_H:(rand_H + self.slide)] = newValue
        
        # label[:,:,:,rand_w:(rand_w + self.slide),rand_H:(rand_H + self.slide)] = torch.ones([1,1,1,self.slide, self.slide], dtype = torch.float32)
        
        # label2[:,:,:,rand_w:(rand_w + self.slide),rand_H:(rand_H + self.slide)] = torch.zeros([1,1,1,self.slide, self.slide], dtype = torch.float32)
        
        # for ii in range(img.shape[2]):
        #     writer.add_image("teestImg"+ str(self.indice), img[0,:,ii,:,:],ii)
        # writer.add_image("testLabel" + str(self.indice), label[0,0,:,:,:])
        # writer.close()
        label[label>=1] = 1
        labelN = (~(label >= 1)).float()
        # writer.add_image("testLabel2", labelN[0,:,:,:])
        label_ = torch.cat([label, labelN], dim=0)
        # img = torch.squeeze(img, 0)
        # label_ = torch.squeeze(label_, 0)
        
        writer.flush()
        writer.close()
        return img, label_
        # label = torch.squeeze(label, 0)
        # return img, label 


    def __len__(self):
        return self.num
    
if __name__ == "__main__":

    train_dataset = CustomSg(9000)
    trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    drop_last=False,
    pin_memory=True)

    for step, data in enumerate(trainloader):
        inputs, labels = data

