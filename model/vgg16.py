from torchvision.models import vgg16
from torch import nn

def decom_vgg16():
    model = vgg16(pretrained=True)
    features = list(model.features)[:30]
    classifier = list(model.classifier)
    # remove last layer and dropout layer
    del classifier[6]
    del classifier[5]
    del classifier[2]
    # free top layer params
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    return nn.Sequential(*features), nn.Sequential(*classifier)