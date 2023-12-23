import cv2
import os
import numpy  as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
#cropsize = (256,256)
cropsize = (512,512)
class Custom(Dataset):
    def __init__(self, datapath, cropsize=cropsize, *args, **kwargs):
        super(Custom, self).__init__(*args, **kwargs)
 
        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        self.img_paths = []
        self.gt_paths = []

        with open(datapath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip("\r\n")
                imgname = line.split("/")[-1]
                imgfolder = line.split("/")[-2]
                maskfolder = 'mask'+imgfolder.split('image')[-1]
                gtline = '/'.join(line.split("/")[0:-2])+'/'+maskfolder+'/'+imgname
                #gtline = '/'.join(line.split("/")[0:-3])+'/'+'/mask/'+maskfolder+'/'+imgname
                self.img_paths.append(line)
                self.gt_paths.append(gtline)

    def __getitem__(self, index):

        img_path = self.img_paths[index]
        #print('img_path: ',img_path)
        gt_path = self.gt_paths[index]
        img = cv2.imread(img_path)
        label = cv2.imread(gt_path,0)
        img = cv2.resize(img,cropsize)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        label = cv2.resize(label,cropsize)
        img = img / 255.0
        img = torch.from_numpy(img)
        img = img.permute(2,0,1).type(torch.float32)
        label = label / 255.0 
        label[label>0.5] = 1
        label[label<0.5] = 0
        #kernel = np.ones((3, 3), np.uint8)
        #mask = cv2.dilate(label, kernel, iterations=1)
        #new_mask = np.zeros((mask.shape[0],mask.shape[1]),dtype=np.int64)
        #new_mask[(mask==1)] = 1


        # new_mask = np.zeros((2,mask.shape[0],mask.shape[1]),dtype=np.float32)
        # new_mask[0,(mask==0)] = 1
        # new_mask[1,(mask==1)] = 1

        #label = new_mask*255
         #cv2.imwrite('label.jpg',label)
        #labelimg = label * 255
        #cv2.imwrite('label/label_'+str(index)+'.jpg',labelimg)
        label = label.astype(np.int64)
        #return img, new_mask
        return img, label


    def __len__(self):
        return len(self.img_paths)