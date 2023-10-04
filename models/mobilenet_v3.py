import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import torch.optim as optim
import torch.nn as nn
from utils import *
import time
import argparse


def weights_init(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    print('Initialize network with %s type' % init_type)
    net.apply(init_func)


###========================== MobileNetv3 framework ==========================
def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MobileBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, kernel, stride, model_rate, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and in_channels == out_channels

        if nl == 'RE':
            nonlinear_layer = nn.ReLU  # or ReLU6
        elif nl == 'HS':
            nonlinear_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        in_channels = int(in_channels * model_rate)
        latent_channels = int(latent_channels * model_rate)
        out_channels = int(out_channels * model_rate)

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, latent_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(latent_channels, track_running_stats=True),
            nonlinear_layer(inplace=True),
            # dw
            nn.Conv2d(latent_channels, latent_channels, kernel, stride, padding, groups=latent_channels, bias=False),
            nn.BatchNorm2d(latent_channels, track_running_stats=True),
            SELayer(latent_channels),
            nonlinear_layer(inplace=True),
            # pw-linear
            nn.Conv2d(latent_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=True)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


'''
if mode == 'large':
    # refer to Table 1 in paper
    mobile_setting = [
        # k, exp, c,  se,     nl,  s,
        [3, 16,  16,  False, 'RE', 1],
        [3, 64,  24,  False, 'RE', 2],
        [3, 72,  24,  False, 'RE', 1],
        [5, 72,  40,  True,  'RE', 2],
        [5, 120, 40,  True,  'RE', 1],
        [5, 120, 40,  True,  'RE', 1],
        [3, 240, 80,  False, 'HS', 2],
        [3, 200, 80,  False, 'HS', 1],
        [3, 184, 80,  False, 'HS', 1],
        [3, 184, 80,  False, 'HS', 1],
        [3, 480, 112, True,  'HS', 1],
        [3, 672, 112, True,  'HS', 1],
        [5, 672, 160, True,  'HS', 2],
        [5, 960, 160, True,  'HS', 1],
        [5, 960, 160, True,  'HS', 1],
    ]
elif mode == 'small':
    # refer to Table 2 in paper
    mobile_setting = [
        # k, exp, c,  se,     nl,  s,
        [3, 16,  16,  True,  'RE', 2],
        [3, 72,  24,  False, 'RE', 2],
        [3, 88,  24,  False, 'RE', 1],
        [5, 96,  40,  True,  'HS', 2],
        [5, 240, 40,  True,  'HS', 1],
        [5, 240, 40,  True,  'HS', 1],
        [5, 120, 48,  True,  'HS', 1],
        [5, 144, 48,  True,  'HS', 1], 
        [5, 288, 96,  True,  'HS', 2],
        [5, 576, 96,  True,  'HS', 1],
        [5, 576, 96,  True,  'HS', 1],
    ]
else:
    raise NotImplementedError
'''


class MobileNetV3_large(nn.Module):
    def __init__(self, last_channels=1280, n_class=1000, dropout=0.8, model_rate=1):
        super(MobileNetV3_large, self).__init__()
        # Start Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(16 * model_rate), 3, 2, 1, bias=False),
            nn.BatchNorm2d(int(16 * model_rate), track_running_stats=True),
            Hswish(inplace=True)
        )
        # MobileBottleneck blocks
        self.conv2 = MobileBottleneck(16, 16, 16, 3, 1, model_rate, False, 'RE')
        self.conv3 = MobileBottleneck(16, 24, 64, 3, 2, model_rate, False, 'RE')
        self.conv4 = MobileBottleneck(24, 24, 72, 3, 1, model_rate, False, 'RE')
        self.conv5 = MobileBottleneck(24, 40, 72, 5, 2, model_rate, True, 'RE')
        self.conv6 = MobileBottleneck(40, 40, 120, 5, 1, model_rate, True, 'RE')
        self.conv7 = MobileBottleneck(40, 40, 120, 5, 1, model_rate, True, 'RE')
        self.conv8 = MobileBottleneck(40, 80, 240, 3, 2, model_rate, False, 'HS')
        self.conv9 = MobileBottleneck(80, 80, 200, 3, 1, model_rate, False, 'HS')
        self.conv10 = MobileBottleneck(80, 80, 184, 3, 1, model_rate, False, 'HS')
        self.conv11 = MobileBottleneck(80, 80, 184, 3, 1, model_rate, False, 'HS')
        self.conv12 = MobileBottleneck(80, 112, 480, 3, 1, model_rate, True, 'HS')
        self.conv13 = MobileBottleneck(112, 112, 672, 3, 1, model_rate, True, 'HS')
        self.conv14 = MobileBottleneck(112, 160, 672, 5, 2, model_rate, True, 'HS')
        self.conv15 = MobileBottleneck(160, 160, 960, 5, 1, model_rate, True, 'HS')
        self.conv16 = MobileBottleneck(160, 160, 960, 5, 1, model_rate, True, 'HS')
        # Last Conv
        self.conv17 = nn.Sequential(
            nn.Conv2d(int(160 * model_rate), int(960 * model_rate), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(960 * model_rate), track_running_stats=True),
            Hswish(inplace=True)
        )
        self.conv18 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(int(960 * model_rate), int(last_channels * model_rate), 1, 1, 0),
            Hswish(inplace=True)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(int(last_channels * model_rate), n_class)
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out")
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)


    def forward(self, x):
        # output = {}
        # x = input['img']
        # feature extraction
        x = self.conv1(x)  # out: B * 16 * 112 * 112
        x = self.conv2(x)  # out: B * 16 * 112 * 112
        x = self.conv3(x)  # out: B * 24 * 56 * 56
        x = self.conv4(x)  # out: B * 24 * 56 * 56
        x = self.conv5(x)  # out: B * 40 * 28 * 28
        x = self.conv6(x)  # out: B * 40 * 28 * 28
        x = self.conv7(x)  # out: B * 40 * 28 * 28
        x = self.conv8(x)  # out: B * 80 * 14 * 14
        x = self.conv9(x)  # out: B * 80 * 14 * 14
        x = self.conv10(x)  # out: B * 80 * 14 * 14
        x = self.conv11(x)  # out: B * 80 * 14 * 14
        x = self.conv12(x)  # out: B * 112 * 14 * 14
        x = self.conv13(x)  # out: B * 112 * 14 * 14
        x = self.conv14(x)  # out: B * 160 * 7 * 7
        x = self.conv15(x)  # out: B * 160 * 7 * 7
        x = self.conv16(x)  # out: B * 160 * 7 * 7
        x = self.conv17(x)  # out: B * 960 * 7 * 7
        x = self.conv18(x)  # out: B * 1280 * 1 * 1
        # classifier
        x = x.mean(3).mean(2)  # out: B * 1280 (global avg pooling)
        out = self.classifier(x)  # out: B * 1000
        # output['score'] = out
        # output['loss'] = F.cross_entropy(output['score'], input['label'])
        # return output
        return out


class MobileNetV3_small(nn.Module):
    def __init__(self, last_channels=1280, n_class=1000, dropout=0.8, model_rate=1):
        super(MobileNetV3_small, self).__init__()
        # Start Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(16 * model_rate), 3, 2, 1, bias=False),
            nn.BatchNorm2d(int(16 * model_rate), track_running_stats=True),
            Hswish(inplace=True)
        )

        # MobileBottleneck blocks
        self.conv2 = MobileBottleneck(16, 16, 16, 3, 2, model_rate, True, 'RE')
        self.conv3 = MobileBottleneck(16, 24, 72, 3, 2, model_rate, False, 'RE')
        self.conv4 = MobileBottleneck(24, 24, 88, 3, 1, model_rate, False, 'RE')
        self.conv5 = MobileBottleneck(24, 40, 96, 5, 2, model_rate, True, 'HS')
        self.conv6 = MobileBottleneck(40, 40, 240, 5, 1, model_rate, True, 'HS')
        self.conv7 = MobileBottleneck(40, 40, 240, 5, 1, model_rate, True, 'HS')
        self.conv8 = MobileBottleneck(40, 48, 120, 5, 1, model_rate, True, 'HS')
        self.conv9 = MobileBottleneck(48, 48, 144, 5, 1, model_rate, True, 'HS')
        self.conv10 = MobileBottleneck(48, 96, 288, 5, 2, model_rate, True, 'HS')
        self.conv11 = MobileBottleneck(96, 96, 576, 3, 1, model_rate, True, 'HS')
        self.conv12 = MobileBottleneck(96, 96, 576, 3, 1, model_rate, True, 'HS')
        # Last Conv
        self.conv13 = nn.Sequential(
            nn.Conv2d(int(96 * model_rate), int(576 * model_rate), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(576 * model_rate), track_running_stats=True),
            Hswish(inplace=True)
        )
        self.conv14 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(int(576 * model_rate), int(last_channels * model_rate), 1, 1, 0),
            Hswish(inplace=True)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(int(last_channels * model_rate), n_class)
        )

    def forward(self, input):
        output = {}
        x = input['img']
        # feature extraction
        x = self.conv1(x)  # out: B * 16 * 112 * 112
        x = self.conv2(x)  # out: B * 16 * 56 * 56
        x = self.conv3(x)  # out: B * 24 * 28 * 28
        x = self.conv4(x)  # out: B * 24 * 28 * 28
        x = self.conv5(x)  # out: B * 40 * 14 * 14
        x = self.conv6(x)  # out: B * 40 * 14 * 14
        x = self.conv7(x)  # out: B * 40 * 14 * 14
        x = self.conv8(x)  # out: B * 48 * 14 * 14
        x = self.conv9(x)  # out: B * 48 * 14 * 14
        x = self.conv10(x)  # out: B * 96 * 7 * 7
        x = self.conv11(x)  # out: B * 96 * 7 * 7
        x = self.conv12(x)  # out: B * 96 * 7 * 7
        x = self.conv13(x)  # out: B * 576 * 7 * 7
        x = self.conv14(x)  # out: B * 1280 * 1 * 1
        # classifier
        x = x.mean(3).mean(2)  # out: B * 1280 (global avg pooling)
        out = self.classifier(x)  # out: B * 1000
        output['score'] = out
        output['loss'] = F.cross_entropy(output['score'], input['label'])
        return output

def mobilenetv3(model_rate=1, track=False):
    # data_shape = cfg['data_shape']
    classes_size = 100
    # hidden_size = [int(np.ceil(model_rate * x)) for x in cfg['resnet']['hidden_size']]
    # scaler_rate = model_rate / cfg['global_model_rate']
    scaler_rate = model_rate
    # model = MobileNetV3_small(n_class=classes_size, model_rate=scaler_rate)
    model = MobileNetV3_large(n_class=classes_size, model_rate=scaler_rate)
    # weights_init(model, init_type='normal', init_gain=0.02)
    return model


def train_one_epoch(model, optimizer, loss, lr_schedule, epoch, dataloader, device, printf, batch):
    start = time.time()
    all_loss = 0
    all_accNum = 0
    model.train()
    for idx, (img, labels) in enumerate(dataloader):
        img = img.to(device)
        labels = labels.to(device)
        out = model(img)
        los = loss(out, labels)

        optimizer.zero_grad()
        los.backward()
        optimizer.step()

        all_loss += los.item()
        cur_acc = (out.data.max(dim=1)[1] == labels).sum()
        all_accNum += cur_acc
        # 每prinft输出一次训练效果
        if (idx % printf) == 0:
            print('epoch:{} training:[{}/{}] loss:{:.6f} accuracy:{:.6f}% lr:{}'.format(epoch, idx, len(dataloader),
                                                                                        los.item(),
                                                                                        cur_acc * 100 / len(labels),
                                                                                        optimizer.param_groups[0][
                                                                                            'lr']))

        lr_schedule.step(los.item())

    end = time.time()
    # 训练完一次，输出平均损失以及平均准确率
    all_loss /= len(dataloader)
    acc = all_accNum * 100 / (len(dataloader) * batch)
    print('epoch:{} time:{:.2f} seconds training_loss:{:.6f} training_accuracy:{:.6f}%'.format(epoch, end - start,
                                                                                               all_loss, acc))
    return all_loss

def train_cifar(net, device, opt):


        # 数据预处理
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # 下载数据集
        train_datasets = torchvision.datasets.CIFAR100('../data/CIFAR100/raw', train=True, transform=transform, download=False)
        val_datasets = torchvision.datasets.CIFAR100('../data/CIFAR100/raw', train=False, transform=transform, download=False)

        # 加载数据集
        train_dataloader = DataLoader(train_datasets, batch_size=opt.batch, shuffle=True, num_workers=opt.numworkers,
                                      pin_memory=True)
        val_dataloader = DataLoader(val_datasets, batch_size=opt.batch, shuffle=False, num_workers=opt.numworkers,
                                    pin_memory=True)





        # 定义优化器和损失函数
        optimizer = optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.1, momentum=0.9,
                              weight_decay=5e-4, nesterov=True)
        loss = nn.CrossEntropyLoss()
        lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-6)

        start_epoch = 0
        # 加载权重

        # 开始训练
        for epoch in range(start_epoch, opt.epoches):
            # 训练
            mean_loss = train_one_epoch(net, optimizer, loss, lr_schedule, epoch, train_dataloader, device, opt.printf,
                                        opt.batch)

if __name__ == "__main__":
    # net = MobileNetV3_large()
    parse = argparse.ArgumentParser()
    parse.add_argument('--epoches', type=int, default=30, help='train  epoches')
    parse.add_argument('--batch', type=int, default=48, help='batch size')
    parse.add_argument('--freeze', type=bool, default=False, help='freeze some weights')
    parse.add_argument('--weights', type=str, default='weights/mobilenet_v3_3.pth', help='last weight path')
    parse.add_argument('--numworkers', type=int, default=4)
    parse.add_argument('--savepath', type=str, default='weights', help='model savepath')
    parse.add_argument('--printf', type=int, default=50, help='print training info after 50 batch')
    parse.add_argument('--classNum', type=int, default=10, help='classes num')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using {} to train'.format(device))
    opt = parse.parse_args()
    net = MobileNetV3_large(last_channels=1280, n_class=100, dropout=0).to(device)
    train_cifar(net, device, opt)

