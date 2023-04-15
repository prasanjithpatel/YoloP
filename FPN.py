import torch
import torch.nn as nn
from torchvision import models 
import io
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

def load_backbone(arct=None):
    if arct==None:
        return None
    elif arct=="resnet18":
        #load the model for torchvision
        model= models.resnet18(pretrained=True)
        layer0=nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool)
        layer1=model.layer1
        layer2=model.layer2
        layer3=model.layer3
        layer4=model.layer4
        #making the gradient false for the parameters for all layers
        for p in layer0[0].parameters(): p.requires_grad = False
        for p in layer0[1].parameters():p.requires_grad=False
        for p in layer1.parameters():p.requires_grad=False
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False
        layer0.apply(set_bn_fix)
        layer1.apply(set_bn_fix)
        layer2.apply(set_bn_fix)
        layer3.apply(set_bn_fix)
        layer4.apply(set_bn_fix)
        features = [layer0, layer1, layer2, layer3, layer4]
        return nn.Sequential(*features) 

class FeaturePyramidNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #loading the backbone
        self.backbone=load_backbone("resnet18")
        #initializing the conv layer for skip connections 
        self.lateral1=nn.Conv2d(512,256,1,1)
        self.lateral2=nn.Conv2d(256,256,1,1)
        self.lateral3=nn.Conv2d(128,256,1,1)
        self.lateral4=nn.Conv2d(64,256,1,1)
        #initializing the conv layes for aliasing effect (smoothing)
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        
    def forward(self,x):
        features=[]
        for i,layer in enumerate(self.backbone):
            x=layer(x)
            if i==0:
                continue
            else:
                features.append(x)
                
        p5=self.lateral1(features[3])
        p4=self._upsample(p5,self.lateral2(features[2]))
        p3=self._upsample(p4,self.lateral3(features[1]))
        p2=self._upsample(p3,self.lateral4(features[0]))
        
        #applying the 3*3 filter for to reduce the aliasing effect of upsampling
        p4=self.smooth1(p4)
        p3=self.smooth1(p3)
        p2=self.smooth1(p2)
        
        return [p5,p4,p3,p2]
    def _upsample(self,x,y):
        _,_,h,w=y.shape
        input1=nn.Upsample(scale_factor=2,mode="nearest")
        output=input1(x)
        return output+y


Fpn=FeaturePyramidNetwork()
test=torch.randn(1,3,224,224)