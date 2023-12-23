from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)
import torch
from dataset import get_train_val_test_Dataloaders
from torch.utils.tensorboard import SummaryWriter




from config import (
    TRAIN_CUDA
)

if torch.cuda.is_available() and TRAIN_CUDA:
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda 

writer = SummaryWriter("./log/boardtest/")

if __name__ == "__main__":
    
    
    train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)

    
    for data in train_dataloader:
        image, ground_truth = data['image'], data['label']
        for ii in range(image.shape[2]):
            writer.add_image("image", image[:,:,ii,:,:].reshape(-1,512,512), ii)
            writer.add_image("ground_truth", ground_truth[:,:,ii,:,:].reshape(-1,512,512), ii)
            
        print("hello")


