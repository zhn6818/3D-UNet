import math
import torch
import torch.nn as nn
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BCE_WEIGHTS, BACKGROUND_AS_CLASS, TRAIN_CUDA
)
from torch import optim
from torch.nn import CrossEntropyLoss
from dataset import get_train_val_test_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from Unet3d_model import UNet3DModel
from datasetUnet import Custom
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)

if BACKGROUND_AS_CLASS: NUM_CLASSES += 1

writer1 = SummaryWriter("./log/boardtest1/")

model = UNet3DModel(in_channels=3, num_classes=1)
train_transforms = train_transform
val_transforms = val_transform

if torch.cuda.is_available() and TRAIN_CUDA:
    model = model.cuda()
    # train_transforms = train_transform_cuda
    # val_transforms = val_transform_cuda 
elif not torch.cuda.is_available() and TRAIN_CUDA:
    print('cuda not available! Training initialized on cpu ...')


# train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)
train_dataset = Custom(10)
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=1,drop_last=False,pin_memory=True)

# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

# optimizer = Adam(params=model.parameters())
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=5e-4)

min_valid_loss = math.inf

for epoch in range(TRAINING_EPOCH):
    
    train_loss = 0.0
    model.train()
    for data in train_dataloader:
        image, ground_truth = data
        image = image.cuda()
        ground_truth = ground_truth.cuda()
        target = model(image)
        loss = criterion(target, ground_truth)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    valid_loss = 0.0
    model.eval()
    # for data in val_dataloader:
    #     image, ground_truth = data['image'], data['label']
        
    #     target = model(image)
    #     loss = criterion(target,ground_truth)
    #     valid_loss = loss.item()
    print("Loss/Train: ", train_loss / len(train_dataloader), epoch)
    writer1.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
    # writer.add_scalar("Loss/Validation", valid_loss / len(val_dataloader), epoch)
    
    # print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(val_dataloader)}')
    
    # if min_valid_loss > valid_loss:
    #     # print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
    #     min_valid_loss = valid_loss
    #     # Saving State Dict
    torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_train_loss{train_loss}.pth')

writer1.flush()
writer1.close()

