import os
import argparse
import time
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

#config
parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-config', default='config.py', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[2,3,4], nargs='+', type=int,
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
# 这个用法有意思，实际是 调了model.seq2seq 并且运行了最后这个括号里的五个参数的方法。(初始化了一个对象也就是）
model = getattr(models, opt.model)(config, use_cuda, all_spk_num)

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

optim.set_parameters(model.parameters())
if config.schedule:
    scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)

# log file path
if opt.log == '':
    log_path = config.log + utils.format_time(time.localtime()) + '.log'
else:
    log_path = config.log + opt.log + '.log'
logging = utils.logging(log_path) # 这种方式也值得学习，单独写一个logging的函数，直接调用，既print，又记录到Log文件里。

for k, v in config.items():
    logging("%s:\t%s\n" % (str(k), str(v)))
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

def train(epoch):
    return

def eval(epoch):
    return

def main():
    for i in range(1, config.epoch+1):
        if not config.notrain:
            train(i)
        else:
            eval(i)
    for metric in config.metric:
        logging("Best %s score: %.2f\n" % (metric, max(scores[metric])))


if __name__ == '__main__':
    main()
