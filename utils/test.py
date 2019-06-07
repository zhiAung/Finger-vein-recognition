from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append('../')
import tarfile
from IPython.display import display, Image
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch
import torchvision
import time
from torch.autograd import Variable
from config import config
from models import resnet
from models import loss_head
from models import focal_loss_base
from datasets import dataset_loader
import numpy as np
import time
from tools import test
from utils import visualizer
import os
import cv2

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        print(self.submodule._modules.items())
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            print(name)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


if __name__ == "__main__":
    '''
    opt = config.opt
    train_dataset = dataset_loader.get_dataset_for_loader(opt.train_img_pickle)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=1,
                                                    shuffle=True,
                                                    num_workers=1)
    img, label = iter(train_data_loader).next()
    # print(img.numpy()[0, :, :, :].shape)
    plt.figure()
    plt.imshow(img.numpy()[0, :, :, :].transpose((1, 2, 0)), cmap='gray')
    # 将原来的tensor[1,3,227,227]数据转换为array[227,227,3]
    plt.show()
    img, label = Variable(img.cuda()), Variable(label.cuda())
    model_ft = models.resnet18(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False
    model_ft.fc = torch.nn.Linear(2048, 64)
    # print(myresnet)
    exact_list = ["layer1", "layer2", "layer3", 'layer4']
    myexactor = FeatureExtractor(model_ft, exact_list)
    x = myexactor(img)
    '''
    img = cv2.imread('test.jpg', 0)
    print(img.shape)
    color_map = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imshow(color_map)
    print(color_map.shape)
