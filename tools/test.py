import os
import torch
import numpy as np
import time
import sys
sys.path.append('../')
from config import config
from torch.nn import DataParallel
from models import resnet
import datasets.dataset_loader as dataset_loader


def load_model(backbone, device_ids, test_model_path, use_se):
    if backbone == 'resnet18_finger':
        model = resnet.resnet18_finger(use_se)
    elif backbone == 'resnet18':
        model = resnet.resnet18(pretrained=False)
    elif backbone == 'resnet34':
        model = resnet.resnet34(pretrained=False)
    elif backbone == 'resnet50':
        model = resnet.resnet50(pretrained=False)
    if opt.multi_gpus:
        model = DataParallel(model, device_ids=device_ids)
    model.load_state_dict(torch.load(test_model_path))
    #model.to(torch.device("cuda"))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def find_element(img_name, i):  # 排序比较函数
    elem_str = os.path.splitext(img_name)[0]
    elem_arr = elem_str.split("_")
    if i == 1:
        return int(elem_arr[0])
    if i == 2:
        return int(elem_arr[len(elem_arr) - 1])

def tmp_find_element(img_name, i):
    elem_str = os.path.splitext(img_name)[0]
    elem_arr = elem_str.split("_")
    if i == 1:
        return int(elem_arr[0])
    if i == 2:
        return int(elem_arr[1])
    if i == 3:
        return int(elem_arr[2])
    if i == 4:
        return int(elem_arr[3])


def get_img_list(dataset_file):
    '''
    获取验证集、测试集的图片列表，和制作dataset时的顺序一致，所以可以根据dataset的label(index)对应到此列表的图片
    '''
    img_name_list = []
    for finger in os.listdir(dataset_file):
        images = os.listdir(os.path.join(dataset_file, finger))
        for image in images:
            img_name_list.append(image)
    img_name_list.sort()
    return img_name_list

# 这个地方因为需要把图片的名称和特征连接起来,使用pytorch的数据加载机制dataloader,
# 因为在制作dataset之前把图片的序排好了，所以特征和图片对的上


def get_featurs_dict(model, dataset, img_name_list,  batch_size, num_workers):
    '''
    输验证集、测试集图片入模型，得到特征，并且和图片文件名对应保存在字典里
    Args:
        model: 训练好的模型
        dataset: 验证集/测试集
        img_name_list: 图片文件列表
        batchsize: data_loader batch size
        num_workers: 读取图片的线程数
    Returns:
        features_dict：特征字典
        len(dataset)：数据集大小
    '''
    features = None
    labels = None
    features_dict = {}
    dataset = dataset_loader.get_dataset(dataset, 'val_test')
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for i, (datas, indexs) in enumerate(data_loader):
        if torch.cuda.is_available():
            datas = torch.autograd.Variable(datas.cuda())
        else:
            datas = torch.autograd.Variable(datas)
        outputs = model(datas)
        feature = outputs.data.cpu().numpy()
        indexs = indexs.data.cpu().numpy()
        #print(feature.shape)
        if features is None:
            features = feature
            labels = indexs
        else:
            features = np.vstack((features, feature))
            labels = np.vstack((labels, indexs))
    print(features.shape)
    for i, img_name in enumerate(img_name_list):
        features_dict[img_name] = features[i]
    #print(labels)
    return features_dict, len(dataset)


def cal_accuracy(fe_dict, pair_list):
    '''
    按图片对的方式取出特征，计算距离，并将它和是否同一个图片的标签分别保存到两个索引相同的列表里
    '''
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    sims = []
    labels = []
    
    for pair in pairs:
        splits = pair.split('\t')
        fe_1 = fe_dict[splits[0]]  # 第一张图片
        fe_2 = fe_dict[splits[1]]  # 第二张图片
        label = int(splits[2])  # 是否为同一张图片的标签（1，0）
        # 计算余弦距离
        cos_metric = np.dot(fe_1, fe_2) / (np.linalg.norm(fe_1) * np.linalg.norm(fe_2))
        sims.append(cos_metric)
        labels.append(label)

    y_score = np.asarray(sims)
    y_true = np.asarray(labels)
    best_acc = 0
    best_th = 0
    # 按每一对图片特征的距离为阈值，计算其他多有图片的ACC,选出最好ACC，及此时的阈值
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_acc, best_th


def cal_eer(fe_dict, pair_list):
    '''
    计算eer
    '''
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    genuine_sims = []
    imposter_sims = []
    for pair in pairs:
        splits = pair.split('\t')
        fe_1 = fe_dict[splits[0]]  # 第一张图片
        fe_2 = fe_dict[splits[1]]  # 第二张图片
        label = int(splits[2])  # 是否为同一张图片的标签（1，0）
        # 计算余弦距离
        cos_metric = np.dot(fe_1, fe_2) / (np.linalg.norm(fe_1) * np.linalg.norm(fe_2))
        if label == 1:
            genuine_sims.append(cos_metric)
        elif label == 0:
            imposter_sims.append(cos_metric)

    thresld=np.arange(0, 1, 0.001)
    FRR = []
    FAR = []
    eer = 1
    eer_thresld = 0
    for i in range(len(thresld)):        
        frr = np.sum(genuine_sims < thresld[i]) / len(genuine_sims)
        FRR.append(frr)
        far = np.sum(imposter_sims >= thresld[i])/len(imposter_sims)
        FAR.append(far)
        if (abs(frr - far) < 0.002): #frr和far值相差很小时认为相等
            eer = abs(frr+far)/2
            eer_thresld = thresld[i]
            #print('eer:{}'.format(eer))
    return eer, eer_thresld


def main_test(opt, model, flag = 'val'):
    '''
    主函数
    '''
    if flag == 'test':
        # 1、加载模型
        trained_model = load_model(opt.backbone, opt.device_ids, opt.test_model_path, opt.use_se)
        dataset = opt.test_dataset
        image_pairs_file = opt.test_img_pairs
    else:
        trained_model = model
        dataset = opt.val_dataset
        image_pairs_file = opt.val_img_pairs
    
    # 2、获取测试图片路径列表
    img_names_list = get_img_list(dataset)
    
    # 3、获取(图片名字——特征)字典
    s = time.time()
    fe_dict, cnt = get_featurs_dict(trained_model, dataset, img_names_list,  batch_size=opt.test_batch_size,
                                    num_workers=opt.num_workers)
    t = time.time() - s
    # print('total time is {}, average time is {}'.format(t, t / cnt))
    # 4、计算精度及阈值
    acc, th = cal_accuracy(fe_dict, image_pairs_file)
    eer, eer_th = cal_eer(fe_dict, image_pairs_file)
    print('verification accuracy: ', acc, 'threshold: ', th)
    return acc, th, eer


if __name__ == '__main__':
    opt = config.opt
    main_test(opt,  model = null, flag = 'test' )
    print(get_img_list('/home/hza/deeplearning_project/FingerVeinRecognition/datasets/data/FV-USM/val_48'))
    
