import torch
import torchvision
from torchvision import datasets,models,transforms
import time
import sys
sys.path.append('../')
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


def configure_model(opt):
    # 模型
    if opt.backbone == 'resnet18_finger':
        model = resnet.resnet18_finger(use_se=opt.use_se)
    elif opt.backbone == 'resnet18':
        model = resnet.resnet18(pretrained=False)
    elif opt.backbone == 'resnet34':
        model = resnet.resnet34(pretrained=False)
    elif opt.backbone == 'resnet50':
        model = resnet.resnet50(pretrained=False)
    # 防止模型有改变，所以进行选择参数加载
    if opt.finetune:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(opt.model_pretrained_parameters)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        #model.load_state_dict(torch.load(opt.model_pretrained_parameters), strict=False) # 这一句就可以代替上面的一坨

    # 损失函数的头部（按照网络的一层来定义的）
    if opt.metric == 'add_margin':
        metric_fc = loss_head.AddMarginProduct(1024, opt .num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = loss_head.ArcMarginProduct(1024, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = loss_head.SphereProduct(1024, opt.num_classes, m=4)
    else:
        metric_fc = torch.nn.Linear(512, opt.num_classes)
    if torch.cuda.is_available():
        model = model.cuda() # 转换成在gpu上运算，数据和模型都要转。
        metric_fc = metric_fc.cuda()
    # 损失函数的基底
    if opt.loss == 'focal_loss':
        criterion = focal_loss_base.FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # 优化器
    # ignored_params = list(map(id, model.fc5.parameters())) + list(map(id, model.classifier.parameters()))
    # print(model.layer4.parameters())
    # breakpoint()
    ignored_params = list(map(id, model.fc5.parameters())) # id函数返回对象的内存地址
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': base_params, 'lr': 0.01},
            {'params': model.fc5.parameters(), 'lr': 0.05},
            {'params': metric_fc.parameters(), 'lr': 0.05},
        ], momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': 0.01},
            {'params': model.fc5.parameters(), 'lr': 0.05},
            {'params': metric_fc.parameters(), 'lr': 0.05},
        ], lr=opt.lr, weight_decay=opt.weight_decay)

    # 学习率调整策略
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)  # 检查参数设置
    if opt.multi_gpus:
        model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # 多GPU并行数据加载
        metric_fc = torch.nn.DataParallel(metric_fc, device_ids=opt.device_ids)
    model.train()
    metric_fc.train()
    return model, metric_fc, criterion, optimizer, scheduler


def save_model(model, opt, iter_cnt, acc, th):
    if not os.path.isdir(opt.checkpoints_path):
        os.makedirs(opt.checkpoints_path)
    prefix = opt.checkpoints_path + opt.backbone + '_' + str(iter_cnt) + '_' + str(round(acc, 4)) + '_' + str(th) + '_'
    name = time.strftime(prefix + '%m%d_%H:%M:%S.pkl')
    print(name)
    torch.save(model.state_dict(), name)


def train_val_model(opt, model, metric_fc, criterion, optimizer, scheduler):
    train_dataset = dataset_loader.get_dataset(opt.train_dataset, 'train')
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=opt.train_batch_size,
                                                    shuffle=True,
                                                    num_workers=opt.num_workers)
    best_model = model
    best_acc = 0.0
    print("训练开始")
    for i in range(opt.max_epoch):
        scheduler.step()
        model.train()
        metric_fc.train()
        epoch_total_corrects = 0
        epoch_total_loss = 0.0
        start = time.time()
        for ii, data in enumerate(train_data_loader):
            input_data, labels = data
            if torch.cuda.is_available():
                input_data, label = input_data.cuda(), labels.cuda()
            input_data, label = torch.autograd.Variable(input_data), torch.autograd.Variable(label)
            optimizer.zero_grad()
            feature = model(input_data)
            outputs = metric_fc(feature, label)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            outputs = outputs.data.cpu().numpy()
            label = label.data.cpu().numpy()
            pres = np.argmax(outputs, axis=1) # 每行的最大值  也可以直接对output tensor 使用torch的方法 torch.max(outputs.data, 1)
            epoch_total_corrects += np.sum(pres == label)  #(pres == label).sum()
            epoch_total_loss += loss.item() #loss.data[0] # 新版变成 loss.item()
            iters = i * len(train_data_loader) + ii
            if iters % opt.print_freq == 0:
                acc_batch = np.mean((pres == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('time:{} | train epoch:{} | total_iters:{} | iter:{} | speed:{}iters/s | loss:{} | acc:{}'
                      .format(time_str,
                               i+1,
                               iters,
                               ii+1,
                               round(speed, 2),
                               round(loss.item(), 2),
                               acc_batch))
                print("The remaining time is:{}:h,{}minute,{}s".format(
                    round(((opt.max_epoch * len(train_data_loader)-iters)/speed)//3600, 0),
                    round((((opt.max_epoch * len(train_data_loader)-iters)/speed) % 3600)//60, 0),
                    round((((opt.max_epoch * len(train_data_loader) - iters) / speed) % 3600) % 60, 2)))
                if opt.display:
                    visualizer.display_current_results(iters, loss.item(), name='train_batch_loss')
                    visualizer.display_current_results(iters, acc_batch, name='train_batch_acc')
                start = time.time()
        # save_model(model, opt, iters)
        model.eval()
        epoch_test_acc, epoch_test_th, eer = test.main_test(opt, model, flag = 'val')
        # epoch_test_acc = test.acc
        if opt.display:
            # 每个epoch的平均损失
            visualizer.display_current_results(i+1, epoch_total_loss/len(train_dataset), name='train_epoch_loss')
            # 每个epoch训练集的精度
            visualizer.display_current_results(i+1, epoch_total_corrects/len(train_dataset), name='train_epoch_acc')
            # 本epoch的测试集精度
            visualizer.display_current_results(i+1, epoch_test_acc, name='test_epoch_acc')
            visualizer.display_current_results(i+1, eer, name='eer')
        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            best_model = model
            save_model(best_model, opt, iters, epoch_test_acc, epoch_test_th)  # 训练中途停止的话，可以保存下来截止最好的模型
    return best_model, len(train_data_loader)


def main_train(opt):
    model, metric_fc, criterion, optimizer, scheduler = configure_model(opt)
    trained_model, iters_dataloader = train_val_model(opt, model, metric_fc, criterion, optimizer, scheduler)
    # save_model(trained_model, opt, opt.max_epoch*iters_dataloader)


if __name__ == "__main__":
    opt = config.opt
    if opt.display:
        visualizer = visualizer.Visualizer(opt.env)
    main_train(opt)







