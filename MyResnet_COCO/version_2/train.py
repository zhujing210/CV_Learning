from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  transforms, models
from utils.checkpoint_saver import CheckpointSaver
from utils.resume_checkpoint import *
from utils.visdomPlot import *
from optim import create_optimizer_v2, optimizer_kwargs
from scheduler import create_scheduler

import time
import os
from torch.utils.data import Dataset

import numpy as np

import argparse 
from model.Resnet18_simplified import *
from data.autoaugment import ImageNetPolicy
from data.CustomData import customData

# 基于自己笔记本的一些默认参数
# 记录训练结果的文档
log_path = r'D:\CV\CV论文调研\MyResnet_COCO\version_2\result_log.txt'
#  数据集根路径
root_path = r'E:\data\data'

num_classes = 2
# how many train epochs to eval once
eval_step = 1
# val acc最高的模型权重保存路径
best_model = r'D:\CV\CV论文调研\MyResnet_COCO\version_2\checkppoints\best_resnet.pkl'
# initial_checkpoint
initial_checkpoint = r'D:\CV\CV论文调研\MyResnet_COCO\version_2\checkppoints\resnet18-f37072fd.pth'
# 训练权重文件的保存路径
output_dir = r'D:\CV\CV论文调研\MyResnet_COCO\version_2\output'
#resume checkpoint路径
resume_file = 'last.pth'
resume_from = os.path.join(output_dir, resume_file)


parser = argparse.ArgumentParser(description='Specify parameters through the command line')
# Dataset / Model parameters
parser.add_argument('--data_dir', metavar='DIR',default=root_path,  
                    help='path to dataset')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=True,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default=initial_checkpoint, type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=2, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--mean', type=list, nargs='+', default=[0.485, 0.456, 0.406], metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=list, nargs='+', default=[0.229, 0.224, 0.225], metavar='STD',
                    help='Override std deviation of of dataset')

parser.add_argument('-b', '--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')


# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')


# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')


# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')

parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--reprob', type=float, default=0.5, metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--rodegree', type=int, default=90,nargs='+',
                    help='rotate degrees')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')


# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default=output_dir, type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-step', default='1', type=int, metavar='epoch',
                    help='val once per eval-step train epochs')
parser.add_argument('--eval-metric', default='acc', type=str, metavar='EVAL_METRIC',
help='Best metric (default: "acc"')
parser.add_argument('--save-all', default=True, type=bool, metavar='SAVE_ALL',
                    help='save all parameters, include state_dict, optimizer,epoch,etc')


def _parse_args():
    # Do we have a config file to parse?
    #args_config, remaining = config_parser.parse_known_args()
    #if args_config.config:
        #with open(args_config.config, 'r') as f:
            #cfg = yaml.safe_load(f)
            #parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    #args = parser.parse_args(remaining)
    args = parser.parse_args()
    # Cache the args as a text string to save them in the output dir later
    #args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    #return args, args_text
    return args


def val_one_epoch(epoch, model, criterion, use_gpu, loader_va, val_dataset_size):
    inputs_accumulate = 0
    since = time.time()
    begin_time = time.time()
    model.eval()
    print('Validation')
    print('-' * 20)
    with open(log_path, 'a+') as f:
        print('Validation', file=f)
        print('-' * 20, file=f)
    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0.0
        for iter_idx, data in enumerate(loader_val, start=1):
            inputs, labels = data
            inputs_accumulate += len(inputs)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            loss =  criterion(outputs, labels)
            running_loss += loss.item()
            running_corrects += torch.eq(preds, labels).float().sum().item()

            if (iter_idx) % args.log_interval == 0:  # 打印日志
                batch_loss = running_loss / inputs_accumulate
                batch_acc = running_corrects / inputs_accumulate
                print('val for Epoch_train [{}] iteration [{}] Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'.\
                    format(epoch, iter_idx, batch_loss, batch_acc, time.time()-begin_time))
                with open(log_path, 'a+') as f:
                    print('val for Epoch_train [{}] iteration [{}] Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'.\
                    format(epoch, iter_idx, batch_loss, batch_acc, time.time()-begin_time), file=f)
                begin_time = time.time()
        time_elapsed = time.time() - since
        val_loss = running_loss / val_dataset_size 
        val_acc = running_corrects / val_dataset_size
        plot_val(val_loss, val_acc, epoch)
        print('val Loss: {:.4f} Acc: {:.4f}'.format( val_loss, val_acc))
        print('validate complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        with open(log_path, 'a+') as f:
            print('val Loss: {:.4f} Acc: {:.4f}'.format( val_loss, val_acc), file=f)
            print('validate complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=f)
        return val_acc


def train_model(model, criterion, optimizer, scheduler, num_epochs, use_gpu, eval_step,
                 loader_train, loader_val, train_dataset_size, val_dataset_size, saver=None, resume_epoch=0):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0  #  val最高acc
    val_times = 0  # val 次数
    with open(log_path, 'w') as f:  # w一次，可以每次train都清空上次train的内容，后面再a进行追加
        f.write("Start train\n")

    for epoch in range(resume_epoch ,num_epochs):
        inputs_accumulate = 0   # 累加各个iter一共处理了多少图片，方便计算running_corrects
        begin_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)
        with open(log_path, 'a') as f:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1), file=f)
            print('-' * 20, file=f)

        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        T = 0 # 一个epoch含有多少个iteration，用于visdom绘图
        for iter_idx, data in enumerate(loader_train, start=1):
            inputs, labels = data
            inputs_accumulate += len(inputs)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            if iter_idx==1:
                if train_dataset_size % len(inputs) == 0:
                    T = train_dataset_size / len(inputs)
                else:
                    T = (train_dataset_size // len(inputs)) + 1

            #zero the parameter gradients
            # 若batchsize太大，考虑前向传播几次再更新梯度
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            loss =  criterion(outputs, labels)
            # backward，optimize
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item()
            running_corrects += torch.eq(preds, labels).float().sum().item()
            # print and write txt
            if (iter_idx) % args.log_interval == 0:  
                batch_loss = running_loss / inputs_accumulate
                batch_acc = running_corrects / inputs_accumulate
                plot_train(loss=batch_loss, acc=batch_acc, iteration=iter_idx, T=T, epoch=epoch)
                print('train: Epoch [{}] iteration [{}] Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'. \
                        format(epoch, iter_idx, batch_loss, batch_acc, time.time()-begin_time))
                with open(log_path, 'a+') as f:
                    print('train: Epoch [{}] iteration [{}] Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'. \
                        format(epoch, iter_idx, batch_loss, batch_acc, time.time()-begin_time), file=f)
                begin_time = time.time()

        epoch_loss = running_loss / train_dataset_size
        epoch_acc = running_corrects / train_dataset_size
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        with open(log_path, 'a+') as f:
            print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc), file=f)
        
        #plot train curve 
        #plot(loss=epoch_loss, acc=epoch_acc, epoch= epoch, is_train=True)

        # save model
        #if not os.path.exists('output'):
            #os.makedirs('output')
        if args.save_all:
            saver.save_checkpoint(epoch=epoch, metric=epoch_acc)
        else:
            torch.save(model, os.path.join(args.output, 'resnet_epoch{}.pkl'.format(epoch)))
        #torch.save(model, 'output/resnet_epoch{}.pkl'.format(epoch))
        
        # 每个epoch更新lr
        scheduler.step(epoch+1, eval_metric)
        # 每eval_step个epoch eval一次，以及 最后一个epoch评估一次
        if ((epoch+1) % eval_step==0) or epoch==num_epochs-1:
            
            val_acc = val_one_epoch(epoch, model, criterion, use_gpu, loader_val, val_dataset_size)
            
            # plot val curve
            #plot(loss=val_loss, acc=val_acc, epoch= val_times , is_train=False)
            
            # deep copy the model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = model.state_dict()
            val_times += 1


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    with open(log_path, 'a+') as f:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=f)
        print('Best val Acc: {:4f}'.format(best_acc), file=f)

    # load best model weights
    model.load_state_dict(best_model_wts)
    # save best model
    torch.save(model, best_model)


if __name__ == '__main__':

    #args, args_text = _parse_args()
    args = _parse_args()
    args.prefetcher = not args.no_prefetcher  # 加快数据读取

    # train 图片路径
    train_img_path = os.path.join(args.data_dir, 'train')
    # train 图片txt文档路径   “绝对路径 \t  label”
    train_txt_path = os.path.join(args.data_dir, 'train/train.txt')
    # val 图片路径
    val_img_path = os.path.join(args.data_dir, 'val')
    # train 图片txt文档路径   “绝对路径 \t  label”
    val_txt_path = os.path.join(args.data_dir, 'val/val.txt')


    #  数据集太小了，就多加一些数据增广
    if not args.no_aug:
        train_transforms = transforms.Compose(transforms=[
                transforms.Resize([224, 224]),
                transforms.RandomHorizontalFlip(p=args.hflip),
                transforms.RandomVerticalFlip(p=args.vflip),
                transforms.RandomRotation(degrees=args.rodegree),
                transforms.ColorJitter(),
                ImageNetPolicy(),           # autoagument
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std),
                transforms.RandomErasing(args.reprob)
            ])
        val_transforms = transforms.Compose(transforms=[
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std),
            ])
    else:
        train_transforms = transforms.Compose(transforms=[
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std),
            ])
        val_transforms = transforms.Compose(transforms=[
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std),
            ])

    use_gpu = torch.cuda.is_available()

    train_dataset = customData(img_path=train_img_path,
                                    txt_path=train_txt_path,
                                    data_transforms=train_transforms) 

    train_dataset_size = len(train_dataset)

    val_dataset = customData(img_path=val_img_path,
                                    txt_path=val_txt_path,
                                    data_transforms=val_transforms)
    
    val_dataset_size = len(val_dataset)

    # wrap your data and label into Tensor
    loder_train = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True) 

    loader_val  =  torch.utils.data.DataLoader(val_dataset,
                                                batch_size=args.batch_size*args.validation_batch_size_multiplier,
                                                shuffle=True) 

 
    # 换模型
    #model_ft = models.resnet18(pretrained=True)  
    #num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = resnet18()
    if args.pretrained:
        model_dict = model_ft.state_dict()
        pretrained_dict = torch.load(args.initial_checkpoint)
        pretrained_dict = {k:v for k, v in pretrained_dict.items() if (k in model_dict and not k.count('fc'))}
        # pretrained_dict = {k:v for k, v in pretrained_dict.items() if not k.count('fc')}
        model_dict.update(pretrained_dict)
        model_ft.load_state_dict(model_dict)
    
    # if use gpu
    if use_gpu:
        model_ft = model_ft.cuda()

    # define cost function。
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # lr设置,pretain,batchsize
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum)
    #lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.decay_epochs, gamma=args.decay_rate)
    optimizer_ft = create_optimizer_v2(model_ft, **optimizer_kwargs(cfg=args))
    lr_scheduler, num_epochs = create_scheduler(args, optimizer_ft)

    # optionally resume from a checkpoint
    resume_epoch = 0
    if args.resume:
        resume_epoch = resume_checkpoint(
            model=model_ft,  checkpoint_path=args.resume,
            optimizer=None if args.no_resume_opt else optimizer_ft)
            #loss_scaler=None if args.no_resume_opt else loss_scaler)

    # multi-GPU
    #model_ft = torch.nn.DataParallel(model_ft, device_ids=[0,1])

    # save dict(state_dict,optimizer,loss_scalar...)
    eval_metric = args.eval_metric
    decreasing = True if eval_metric == 'loss' else False 
    saver = CheckpointSaver(
            model=model_ft, optimizer=optimizer_ft, args=args,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)

    # train model
    train_model(model=model_ft,
                criterion=criterion,
                optimizer=optimizer_ft,
                scheduler=lr_scheduler,
                num_epochs=num_epochs,
                use_gpu=use_gpu,
                eval_step=args.eval_step,
                loader_train=loder_train,
                loader_val=loader_val,
                saver=saver,
                resume_epoch=resume_epoch,
                train_dataset_size= train_dataset_size,
                val_dataset_size=val_dataset_size
                )