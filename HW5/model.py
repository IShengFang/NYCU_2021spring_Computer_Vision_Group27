from torch import nn
from torchvision import models


class CNN_Model(nn.Module):
    def __init__(self, model_name, pretrained, num_classes=15):
        '''
        torchvision
        [Implemented]
        resnet18, resnet34, resnet50, resnet101, resnet152
        resnext50_32x4d, resnext101_32x8d
        wide_resnet50_2, wide_resnet101_2
        densenet121, densenet169, densenet161, densenet201
        inception_v3, googlenet,
        [NotImplementedYet]
        alexnet, vgg16, vgg16_bn, vgg19, vgg19_bn
        squeezenet1_0, squeezenet1_1
        shufflenet_v2_x0_5 shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
        mobilenet_v2,
        '''
        super().__init__()
        # out name
        # fc: resnet, inception, googlenet, shufflenet, resnext50_32x4d, wide_resnet50_2
        # classifier: alexnet, vgg, squeezenet, mobilenet, mnasnet, classifier

        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes

        self.last_fc = 'res' in self.model_name or \
                        self.model_name == 'googlenet'
        self.densenet = 'densenet' in self.model_name
        self.inception = self.model_name == 'inception_v3'
        exec(f'self.model = models.{self.model_name}(pretrained=pretrained)', {'self': self, 'models': models, 'pretrained': self.pretrained})

        if self.last_fc:
            have_bias = self.model.fc.bias is not None
            self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes, bias=have_bias)
        elif self.densenet:
            have_bias = self.model.classifier.bias is not None
            self.model.classifier = nn.Linear(in_features=self.model.classifier.in_features, out_features=self.num_classes, bias=have_bias)
        elif self.inception:
            self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes, bias=have_bias)
            raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.model(x)
