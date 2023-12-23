'''
tensorboard --logdir=log_path
:return:
'''
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# logger = SummaryWriter(log_dir='./log/boardtest/')

# loss = [5.5, 4.1, 4.2, 3.2, 3.3, 2.9, 2.5, 1.2, 0.8, 0.6]
# steps = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

# for i in range(len(loss)):

#     logger.add_scalar('train_loss', loss[i], steps[i])
#     logger.add_scalar('train_steps', loss[i], steps[i])

#     logger.add_scalar('val_loss', loss[i], steps[i])
#     logger.add_scalar('val_steps', loss[i], steps[i])

writer = SummaryWriter("./log/boardtest/")
img = Image.open("image/dogjpg.jpeg")
tensor_tool = transforms.ToTensor()
img_tensor = tensor_tool(img)


writer.add_image("ToTensor",img_tensor)

#normalized的使用,需要输入均值和标准差

trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize",img_norm,0)

norm_tool = transforms.Normalize([1,3,5],[3,1,2])
img_norm = norm_tool(img_tensor)
writer.add_image("Normalize",img_norm,1)
 
norm_tool = transforms.Normalize([2,5,3],[1,5,6])
img_norm = norm_tool(img_tensor)
writer.add_image("Normalize",img_norm,2)


writer.close()
