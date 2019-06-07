# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
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
from PIL import Image


def configure_model(opt):
    # 模型
    if opt.backbone == 'resnet18_finger':
        model = resnet.resnet_finger(use_se=opt.use_se)
    elif opt.backbone == 'resnet18':
        model = resnet.resnet18(pretrained=False)
    elif opt.backbone == 'resnet34':
        model = resnet.resnet34(pretrained=False)
    elif opt.backbone == 'resnet50':
        model = resnet.resnet50(pretrained=False)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(opt.test_model_pretrain_parameter)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if torch.cuda.is_available():
        model = model.cuda()
    #model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # 多GPU并行数据加载
    model.eval()
    return model


class FeatureExtractor(torch.nn.Module):
    def __init__(self, module):
        super(FeatureExtractor, self).__init__()
        self.module = module
    def forward(self, x):
        outputs = []
        x = self.module.conv1(x)
        x = self.module.bn1(x)
        x = self.module.relu(x)
        outputs.append(x)
        x = self.module.maxpool(x)
        x = self.module.layer1(x)
        outputs.append(x)
        x = self.module.layer2(x)
        outputs.append(x)
        x = self.module.layer3(x)
        outputs.append(x)
        x = self.module.layer4(x)
        outputs.append(x)
        x = self.module.avgpool(x)
        return outputs


def load_data_and_featuremaps_from_dataloader(model):
    train_dataset = dataset_loader.get_dataset_for_loader(opt.train_img_pickle)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=1,
                                                    shuffle=True,
                                                    num_workers=1)

    img, label = iter(train_data_loader).next()

    img_g, label_g = Variable(img.cuda()), Variable(label.cuda())
    myexactor = FeatureExtractor(model)
    x = myexactor(img_g)
    print(img.size())
    img = transforms.ToPILImage()(img.squeeze())
    print(img)
    return img, x


def load_data_and_featuremaps_from_file(model, img_name):
    img = Image.open('vis_raw_images/' + img_name)
    #print(img.size)
    img = img.resize((320, 240), Image.ANTIALIAS)
    img = img.convert("RGB")
    #print(img.size)
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.resize_(1, 3, 240, 320)
    #print(img_tensor.size())
    img_g = Variable(img_tensor.cuda())
    myexactor = FeatureExtractor(model)
    x = myexactor(img_g)
    return img, x


def superimposed_heat_map(feature_map, image, save_path):
    heatmap = torch.mean(feature_map, dim=1).squeeze().data.cpu().numpy()
    #print(heatmap.shape)
    #heatmap = heatmap.swapaxes(1,0)
    #print(heatmap.shape)
    heatmap = cv2.resize(heatmap, (320, 240))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = heatmap * 255
    #print(image.size)
    #image = cv2.cvtColor(np.asarray(image.resize((240, 320))), cv2.COLOR_BGR2RGB)
    #print(image.shape)
    image = cv2.cvtColor(np.asarray(image) , cv2.COLOR_BGR2RGB)
    #print(image.shape)
    color_map = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    img_res = cv2.addWeighted(image.astype(np.uint8), 0.5, color_map.astype(np.uint8), 0.5, 0)
    print(type(img_res))


    cv2.imwrite(save_path, img_res)


if __name__ == "__main__":
    save_path = '/home/ytt/deeplearning_project/FingerVeinRecognition/utils/vis_heat_images/'
    opt = config.opt
    model = configure_model(opt)

    img_list = os.listdir('vis_raw_images')
    for img_name in img_list:
        img, feature_maps = load_data_and_featuremaps_from_file(model, img_name)
        superimposed_heat_map(feature_maps[2], img, os.path.join(save_path, 'heat' + img_name))
    print('finish')

    
    #img, feature_maps = load_data_and_featuremaps_from_dataloader(model)
    #
    

    
