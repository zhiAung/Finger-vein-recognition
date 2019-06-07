import os
import copy
import warnings


class DefaultConfig():
    # 一、model
    env = 'main'
    backbone = 'resnet18_finger'
    classify = 'softmax'
    num_classes = 296
    metric = 'arc_margin'
    easy_margin = True
    use_se = True
    loss = 'focal_loss'
    device_ids = [0, 1]
    multi_gpus = True
    display = True
    optimizer = 'sgd'
    num_workers = 10  # how many workers for loading data
    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'


    # 二、train
    finetune = True
    checkpoints_path = '/home/hza/deeplearning_project/FingerVeinRecognition/checkpoints'

    model_pretrained_parameters = '/home/hza/.cache/torch/checkpoints/resnet18-5c106cde.pth'
    #model_pretrained_parameters = '/home/hza/deeplearning_project/FingerVeinRecognition/checkpoints_s_roi/'\
    #'resnet18_719_0.9083_0.853572_0409_09:38:32.pkl'
    train_dataset = "/home/hza/deeplearning_project/FingerVeinRecognition/datasets/data/FV-USM/train_296"  # 训练集序列化文件 2_train_bmp_aug.pickle \2_train_bmp.pickle
    train_batch_size = 32
    print_freq = 100  # print info every N batch
    max_epoch = 100
    #lr = 1e-1  # initial learning rate
    lr = 1e-2
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decayS
    weight_decay = 5e-4


    # 三、val(test)
    val_dataset = "/home/hza/deeplearning_project/FingerVeinRecognition/datasets/data/FV-USM/val_48" # 测试的时候所使用的序列化文件
    val_img_pairs = '/home/hza/deeplearning_project/FingerVeinRecognition/datasets/data/FV-USM/val_img_pair.txt'
    test_model_pretrain_parameter = "/home/hza/deeplearning_project/FingerVeinRecognition/checkpoints_qb/" \
                                    "resnet18_339_0.9922_0.492546_1209_17:29:35.pkl"
    # 验证集（测试集）的数据文件，也是测试代码里面获取图像名字列表，以及制作图像对的来源
    test_dataset = '/home/hza/deeplearning_project/dataset/BMP/2_val_bmp'
    test_img_pairs = '/home/hza/deeplearning_project/FingerVeinRecognition/datasets/data/FV-USM/test_img_pair.txt'
    test_batch_size = 24

    def parse(self, kwargs):
        for k, v in kwargs.iteritems():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
        setattr(self, k, v)
        print('user config:')
        for k, v in self.__class__.__dict__.iteritems():
            if not k.startswith("__"):
                print(k, getattr(self, k))


opt=DefaultConfig()




