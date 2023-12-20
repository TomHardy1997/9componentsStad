import torchvision.models as models
import torch 
import torch.nn as nn


class CustomNet(nn.Module):
    def __init__(self, num_classes=9, pretrained=True):
        super(CustomNet, self).__init__()
        
        self.conv_features = models.resnet50(pretrained=pretrained)
        self.conv_features = nn.Sequential(*list(self.conv_features.children())[:-2])
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2048, num_classes) 

    def forward(self, x):
        x = self.conv_features(x) 
        x = self.pool(x)
        x = x.squeeze()
        x = self.dropout(x)        
        out = self.fc(x)
        return out

if __name__ == "__main__":
    net = CustomNet()
    x = torch.randn(9,3,224,224)
    y = net(x)
    import ipdb;ipdb.set_trace()

