import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn, alexnet
import torch.hub

class VGG_16(nn.Module):
    '''
    VGG-16 with batch norm
    Feature concatnation
    '''
    def __init__(self, num_features=None, cat_features=False):
        super(VGG_16, self).__init__()
        self.vgg = vgg16_bn(pretrained=True)
        self.transform = torch.nn.functional.interpolate
        self.cat_features = cat_features
        if cat_features is True:
            self.out_layer1 = nn.Linear(1000 + num_features, 500)
            self.out_layer2 = nn.Linear(500, 1)
        else:
            self.out_layer1 = nn.Linear(1000, 500)
            self.out_layer2 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, features=None):
        image = self.transform(image, mode='bilinear', size=(224, 224), align_corners=False)
        out = self.vgg(image)
        if self.cat_features is True: # feature concatnation
            out = torch.cat((out, features), dim=-1)
            out = self.out_layer1(out)
            out = F.relu(out)
            out = self.out_layer2(out)
        else:
            out = self.out_layer1(out)
            out = F.relu(out)
            out = self.out_layer2(out)
        out = self.sigmoid(out)
        return out


class Alex_Net(nn.Module):
    """ Alex Net
    
    """
    def __init__(self, num_features=None, cat_features=False):
        super(Alex_Net, self).__init__()
        self.alexNet = alexnet(pretrained=True)
        self.transform = torch.nn.functional.interpolate
        self.cat_features = cat_features
        if cat_features is True:
            self.out_layer1 = nn.Linear(1000 + num_features, 500)
            self.out_layer2 = nn.Linear(500, 1)
        else:
            self.out_layer1 = nn.Linear(1000, 500)
            self.out_layer2 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, features=None):
        image = self.transform(image, mode='bilinear', size=(224, 224), align_corners=False)
        out = self.alexNet(image)
        if self.cat_features is True:
            out = torch.cat((out, features), dim=-1)
            out = self.out_layer1(out)
            out = F.relu(out)
            out = self.out_layer2(out)
        else:
            out = self.out_layer1(out)
            out = F.relu(out)
            out = self.out_layer2(out)
        out = self.sigmoid(out)
        return out
    
class Efficient_Net(nn.Module):
    '''
    Efficient Net with batch norm
    Feature concatnation
    '''
    def __init__(self, num_features=None, cat_features=False):
        super(Efficient_Net, self).__init__()
        self.effnet = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b5_ap', pretrained=True)
        self.transform = torch.nn.functional.interpolate
        self.cat_features = cat_features
        if cat_features is True:
            self.out_layer1 = nn.Linear(1000 + num_features, 500)
            self.out_layer2 = nn.Linear(500, 1)
        else:
            self.out_layer1 = nn.Linear(1000, 500)
            self.out_layer2 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, image, features=None):
        image = self.transform(image, mode='bilinear', size=(224, 224), align_corners=False)
        out = self.effnet(image)
        if self.cat_features is True: # feature concatnation
            out = torch.cat((out, features), dim=-1)
            out = self.out_layer1(out)
            out = F.relu(out)
            out = self.out_layer2(out)
        else:
            out = self.out_layer1(out)
            out = F.relu(out)
            out = self.out_layer2(out)
        out = self.sigmoid(out)
        return out
        
class SE_Net(nn.Module):
    '''
    Squeeze-and-Excitation Net with batch norm
    Feature concatnation
    '''
    def __init__(self, num_features=None, cat_features=False):
        super(SE_Net, self).__init__()
        self.senet = torch.hub.load('moskomule/senet.pytorch', 'se_resnet50', pretrained=True)
        self.transform = torch.nn.functional.interpolate
        self.cat_features = cat_features
        if cat_features is True:
            self.out_layer1 = nn.Linear(1000 + num_features, 500)
            self.out_layer2 = nn.Linear(500, 1)
        else:
            self.out_layer1 = nn.Linear(1000, 500)
            self.out_layer2 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, image, features=None):
        image = self.transform(image, mode='bilinear', size=(224, 224), align_corners=False)
        out = self.senet(image)
        if self.cat_features is True: # feature concatnation
            out = torch.cat((out, features), dim=-1)
            out = self.out_layer1(out)
            out = F.relu(out)
            out = self.out_layer2(out)
        else:
            out = self.out_layer1(out)
            out = F.relu(out)
            out = self.out_layer2(out)
        out = self.sigmoid(out)
        return out
        
if __name__ == '__main__':
    num_features = 10
    features = torch.zeros((60, num_features))
    cat_features = True
    
    input_image= torch.zeros((60, 3, 256, 256))
    # model = Efficient_Net(num_features, cat_features)
    model = Alex_Net(num_features, cat_features)
    
    if cat_features is True:
        out = model(input_image, features)
    else:
        out = model(input_image)

    print(out.shape) 