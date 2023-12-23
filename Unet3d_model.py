import torch
from torch import nn
from torchsummary import summary
import time

class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels)
        # self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        # self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

    
    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        # res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out#, res


class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels) -> None:
        super(UpConv3DBlock, self).__init__()
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 2, 2), stride=2)
            
        
    def forward(self, input):
        # sizeHW = input.shape[-1]
        # batch = input.shape[0]
        # input = input.view(batch, -1, sizeHW, sizeHW)
        out = self.upconv1(input)
        # if residual!=None: out = torch.cat((out, residual), 1)
        # out = self.relu(self.bn(self.conv1(out)))
        # out = self.relu(self.bn(self.conv2(out)))
        # if self.last_layer: out = self.conv3(out)
        return out

class UNet3DModel(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels 
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """
    
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512) -> None:
        super(UNet3DModel, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.a_block4 = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel)
        
        self.s_block5 = UpConv3DBlock(in_channels=bottleneck_channel, out_channels=bottleneck_channel//2)
        self.s_block6 = UpConv3DBlock(in_channels=bottleneck_channel//2, out_channels=bottleneck_channel//4)
        self.s_block7 = UpConv3DBlock(in_channels=bottleneck_channel//4, out_channels=bottleneck_channel//8)
        self.s_block8 = UpConv3DBlock(in_channels=bottleneck_channel//8, out_channels=bottleneck_channel//16)
        
        self.conv9 = nn.Conv3d(in_channels= bottleneck_channel//16, out_channels=1, kernel_size=(3,3,3), padding=1)
        
    def forward(self, input):
        out = self.a_block1(input)
        out = self.a_block2(out)
        out = self.a_block3(out)
        out = self.a_block4(out)
        
        out = self.s_block5(out)
        out = self.s_block6(out)
        out = self.s_block7(out)
        out = self.s_block8(out)
        out = self.conv9(out)
        return out
    


if __name__ == '__main__':
    
    
    model = UNet3DModel(in_channels=3, num_classes=1)
    start_time = time.time()
    summary(model=model, input_size=(3, 16, 256, 256), batch_size=-1, device="cpu")