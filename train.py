import os
import argparse
import time
import random
# import collections
# import lera

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import numpy as np

import config
import utils
from optims import Optim
import lr_scheduler as L
from models import model
from predata_fromnp import prepare_data

#config
parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-config', default='config.py', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[2], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default=None, type=str,
                   help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-model', default='model', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-notrain', default=False, type=bool,
                    help="train or not")
parser.add_argument('-log', default='', type=str,
                    help="log directory")

opt = parser.parse_args()
# config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

# checkpoint
if opt.restore:
    print('loading checkpoint...\n',opt.restore)
    checkpoints = torch.load(opt.restore)

all_spk_num=158
# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
use_cuda = True
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)

print('building model...\n')
model = model.basic_model(config, use_cuda, all_spk_num)

if opt.restore:
    model.load_state_dict(checkpoints['model'])

if use_cuda:
    model.cuda()

if len(opt.gpus) > 1:
    model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

# optimizer
if  opt.restore:
    optim = checkpoints['optim']
else:
    optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                  lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)

# optim.set_parameters(model.parameters())
# if config.schedule:
#     scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)

# log file path
if opt.log == '':
    log_path = config.log + utils.format_time(time.localtime()) + '.log'
else:
    log_path = config.log + opt.log + '.log'
logging = utils.logging(log_path) # 这种方式也值得学习，单独写一个logging的函数，直接调用，既print，又记录到Log文件里。

# for k, v in [i for i in locals().items()]:
#     logging("%s:\t%s\n" % (str(k), str(v)))
logging("\n")
logging(repr(model)+"\n\n")

# total number of parameters
param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]
logging('total number of parameters: %d\n\n' % param_count)

def save_model(path):
    global updates
    model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}

    torch.save(checkpoints, path)

def train(epoch,data):
    print('*'*25,'Train epoch:',epoch,'*'*25)
    random.shuffle(data)
    print('First of data:',data[0])
    for idx,part in enumerate(data):
        speech_path,images_path,duration,spk_name=part['speech_path'],\
                                                  part['images_path'],\
                                                  part['duration'],part['spk_name']
        if duration<config.Min_Len:
            continue
        elif duration>config.Max_Len:
            speech_feats=np.load(config.aim_path+speech_path)
            images_feats=np.load(config.aim_path+images_path)
            assert images_feats.shape[0]*4==speech_feats.shape[0]
            shift_time=np.random.random()*(duration-config.Max_Len) #可供移动的时间长度
            shift_frames_image=int(shift_time*25)
            shift_frames_speech=4*shift_frames_image
            images_feats=images_feats[shift_frames_image:int(shift_frames_image+config.Max_Len*25)]
            speech_feats=speech_feats[shift_frames_speech:int(shift_frames_speech+config.Max_Len*100)]
            assert images_feats.shape[0]*4==speech_feats.shape[0]
        else:
            speech_feats=np.load(config.aim_path+speech_path)
            images_feats=np.load(config.aim_path+images_path)
            assert images_feats.shape[0]*4==speech_feats.shape[0]
        print('Enter into the model:',images_feats.shape,speech_feats.shape)
        images_feats=Variable(torch.tensor(images_feats))
        speech_feats=Variable(torch.tensor(speech_feats))
        if use_cuda:
            images_feats=images_feats.cuda()
            speech_feats=speech_feats.cuda()
        model(images_feats,speech_feats)

    return

def eval(epoch,data):
    print('*'*25,'Eval epoch:',epoch,'*'*25)
    random.shuffle(data)
    print('First of data:',data[0])
    return

def main():
    train_data,eval_data=None,None
    for i in range(1, config.EPOCH_SIZE+1):
        if not opt.notrain:
            if not train_data:
                train_data=prepare_data('once','train')
                print('Train data gets items of: ',len(train_data))
            train(i,train_data)
        else:
            if not eval_data:
                eval_data=prepare_data('once','valid')
                print('EVAL data gets items of: ',len(eval_data))
            eval(i,eval_data)
    for metric in config.metric:
        logging("Best %s score: %.2f\n" % (metric, max(scores[metric])))


if __name__ == '__main__':
    main()
