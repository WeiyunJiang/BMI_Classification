import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn, alexnet

class VGG_16(nn.Module):
    """ Naive VGG-16
     
    """
    def __init__(self):
        super(VGG_16, self).__init__()
        self.vgg = vgg16_bn(pretrained=True)
        self.transform = torch.nn.functional.interpolate
        self.out_layer = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        image = self.transform(image, mode='bilinear', size=(224, 224), align_corners=False)
        out = self.vgg(image)
        out = self.out_layer(out)
        out = self.sigmoid(out)
        return out


class Alex_Net(nn.Module):
    """ Alex Net
    
    """
    def __init__(self):
        super(Alex_Net, self).__init__()
        self.alexNet = alexnet(pretrained=True)
        self.transform = torch.nn.functional.interpolate
        self.out_layer = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        image = self.transform(image, mode='bilinear', size=(224, 224), align_corners=False)
        out = self.alexNet(image)
        out = self.out_layer(out)
        out = self.sigmoid(out)
        return out

if __name__ == '__main__':
    model = Alex_Net()
    input_image= torch.zeros((60, 3, 128, 128))
    out = model(input_image)
    print(out.shape) 