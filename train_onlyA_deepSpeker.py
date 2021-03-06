import os
import io
import argparse
import time
import random
# import collections
# import lera
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import numpy as np
from sklearn import metrics
import lera

import config
import utils
from optims import Optim
import lr_scheduler as L
from models import model_deepSpeaker as model
from predata_fromnp import prepare_data

#config
parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-config', default='config.py', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[3], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default=None, type=str,
                   help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-model', default='model', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-notrain', default=0, type=bool,
                    help="train or not")
parser.add_argument('-log', default='', type=str,
                    help="log directory")

opt = parser.parse_args()
# config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

# checkpoint
if opt.restore:
    print('loading checkpoint...\n',opt.restore)
    # checkpoints = torch.load(opt.restore,map_location={'cuda:0':'cuda:2'})
    checkpoints = torch.load(opt.restore,map_location={'cuda:3':'cuda:1'})
    # checkpoints = torch.load(opt.restore,map_location='cpu')

    # checkpoints = torch.load(opt.restore)
    # with open(opt.restore) as f:
    #     buffer = io.BytesIO(f.read())
    # torch.load(buffer)

# all_spk_num=config.size_of_all_spks
all_spk_num=len(config.spks_list_TS3)
config.spks_list=config.spks_list_TS3
# all_spk_num=len(config.spks_list_NoE)
# config.spks_list=config.spks_list_NoE

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
use_cuda = True
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)

print('building model...\n')
model = model.basic_audio_model_deepSpeaker(config, use_cuda, all_spk_num)

loss_func=nn.CrossEntropyLoss()
if opt.restore:
    if not config.mask_conv_bias:
        if 'output_model.mask_conv.bias' in checkpoints['model']:
            checkpoints['model'].pop('output_model.mask_conv.bias')
    model.load_state_dict(checkpoints['model'])

if use_cuda:
    model.cuda()

if len(opt.gpus) > 1:
    model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

updates=0
if opt.restore:
    updates = checkpoints['updates']
# optimizer
if 1 and opt.restore:
    optim = checkpoints['optim']
else:
    optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                  lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)

optim.set_parameters(model.parameters())
if config.schedule:
    scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.EPOCH_SIZE)

# log file path
if opt.log == '':
    save_pat = config.log + utils.format_time(time.localtime())
    if not os.path.exists(save_pat):
        os.mkdir(save_pat)
    log_path = save_pat + '/log'
else:
    save_pat = config.log + opt.log + '_' + utils.format_time(time.localtime())
    if not os.path.exists(save_pat):
        os.mkdir(save_pat)
    log_path = save_pat + '/log'
logging = utils.logging(log_path) # 这种方式也值得学习，单独写一个logging的函数，直接调用，既print，又记录到Log文件里。

# for k, v in [i for i in locals().items()]:
#     logging("%s:\t%s\n" % (str(k), str(v)))
utils.print_all(config,logging)
logging("\n")
logging(repr(model)+"\n\n")

# total number of parameters
param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]
logging('total number of parameters: %d\n\n' % param_count)

total_loss_batch, start_time = 0, time.time()
right_idx_everyXupdates = 0

lera.log_hyperparams({
    'title':'MSC v6 only Audio DeepSpeaker model',
    'updates':updates,
    'log path': log_path,
    'mask_softmax:': config.mask_softmax,
    'image time conv': config.image_time_conv,
    'image time rnn': config.image_time_rnn,
    'mask_conv_bias': config.mask_conv_bias,
    'mask_over_init': config.mask_over_init,
    'only 1 meet:':config.only_1_meet,
    'class_frame:':config.class_frame,
    'mask norm:':config.mask_norm,
    'mask threshold:':config.threshold,
    'masktopk:':config.mask_topk,
    'size sum:':config.size_sum,
})
def save_model(path):
    global updates
    model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        # 'config': config,
        'optim': optim,
        'updates': updates}

    if not config.image_time_conv:
        torch.save(checkpoints, path+'/{}_chechpoint.pt'.format(updates))
    else:
        torch.save(checkpoints, path+'/{}_tmp_chechpoint.pt'.format(updates))


def train(epoch,data):
    global updates,total_loss_batch,right_idx_everyXupdates
    total_loss_batch=0
    right_idx_everyXupdates=0
    print('*'*25,'Train epoch:',epoch,'*'*25)
    random.shuffle(data)
    print('First of data:',data[0])

    if config.schedule:
        scheduler.step()
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])
        lera.log(
            'lr', scheduler.get_lr()[0]
        )
    loss_total=0.0
    right_idx_per_epoch=0
    loss_grad_list=None
    for idx,part in enumerate(data):
        speech_path,images_path,duration,spk_name=part['speech_path'],\
                                                  part['images_path'],\
                                                  part['duration'],part['spk_name']
        if duration<config.Min_Len:
            continue
        elif duration>config.Max_Len:
            speech_feats=np.load(config.aim_path+speech_path)
            shift_time=np.random.random()*(duration-config.Max_Len) #可供移动的时间长度
            shift_frames_image=int(shift_time*25)
            shift_frames_speech=4*shift_frames_image
            speech_feats=speech_feats[shift_frames_speech:int(shift_frames_speech+config.Max_Len*100)]
        else:
            speech_feats = np.load(config.aim_path + speech_path)
        print('Enter into the model:',  speech_feats.shape)
        speech_feats = Variable(torch.tensor(speech_feats))

        try:
            if use_cuda:
                speech_feats = speech_feats.cuda()
            model.zero_grad()
            predict_score, mask = model(speech_feats)
            if not config.class_frame:
                predict_label = torch.argmax(predict_score, 1).squeeze().item()
            else:
                # 这是在一共t个帧里
                predict_idx = torch.argmax(predict_score).item()
                predict_label = predict_idx % len(config.spks_list)

            target_spk = torch.tensor(config.spks_list.index(spk_name)).unsqueeze(0)

            if predict_label == config.spks_list.index(spk_name):
                right_idx_per_epoch += 1
                right_idx_everyXupdates += 1
            if use_cuda:
                target_spk = target_spk.cuda()
                if not config.class_frame:
                    target_spk = target_spk.unsqueeze(0)

            loss = loss_func(predict_score, target_spk)
            if loss_grad_list is None:
                loss_grad_list = loss
            else:
                # print(next(model.parameters()).grad)

                loss_grad_list = loss_grad_list + loss
                pass

                # loss_grad_list.backward()
                # print(next(model.parameters()).grad[0])
            loss_total+=loss.item()
            total_loss_batch+=loss.item()

            print('loss this seq: ',loss.item())
            # if loss.item()<1.1 and 'Overview' in images_path :
            # if loss.item()<1.5 and 'Corner' not in images_path:
            #     print('Low loss in: ',part)
            #     draw_map(images_path,mask.data.cpu().numpy())
            #     1/0

            # else:
            #     continue

                # lera.log(
                #     'Low loss length', duration
                # )

            if 1 and idx%8==0:
                loss_grad_list.backward()
                optim.step()
                loss_grad_list=None # set to 0 every N samples.

            updates += 1


            # count every XXX updates
            count_interval=200
            if updates%count_interval==0: # count the performance every XXX updates
                acc_this_interval=right_idx_everyXupdates/float(count_interval)
                logging("time: %6.3f, epoch: %3d, updates: %8d, train loss this batch: %6.3f,acc this batch: %6.3f\n"
                    % (time.time()-start_time, epoch, updates, total_loss_batch/float(count_interval),right_idx_everyXupdates/float(count_interval)))
                total_loss_batch=0
                right_idx_everyXupdates=0
                lera.log(
                    'Acc',acc_this_interval
                )

            lera.log(
                'loss',loss.item(),
            )

            if updates%config.save_inter==0:
                save_model(save_pat)

        except RuntimeError as RR:
            print('EEE errors here: ',RR)
            loss_grad_list=None # set to 0 every N samples.
            continue
        # except Exception as RR:
        #     print(' errors here: ',RR)
        #     loss_grad_list=None # set to 0 every N samples.
        #     continue

    print('Loss aver for this epoch:',loss_total/len(data))
    print('Acc aver for this epoch:',right_idx_per_epoch/float(len(data)))
    if epoch>=0:
        eval_data=prepare_data('once','all_valid')
        if config.only_1_meet:
            eval_data=[i for i in eval_data if 'TS3' in i['speech_path']]
        print('EVAL data gets items of: ',len(eval_data))
        #assert len(set([one['spk_name'] for  one in eval_data]))==all_spk_num
        eval(epoch,eval_data)

def eval(epoch,data):
    global save_pat,updates,total_loss_batch,right_idx_everyXupdates
    total_loss_batch=0
    right_idx_everyXupdates=0
    print('*'*25,'Eval epoch:',epoch,'*'*25)
    print('Valid begins in epoch',epoch)
    # random.shuffle(data)
    print('First of data:',data[0])
    loss_total=0.0
    right_idx_per_epoch=0
    loss_grad_list=None
    success_counte=0.0
    for idx,part in enumerate(data):
        speech_path,images_path,duration,spk_name=part['speech_path'], \
                                                  part['images_path'], \
                                                  part['duration'],part['spk_name']
        if duration<config.Min_Len:
            continue
        elif duration>config.Max_Len:
            speech_feats=np.load(config.aim_path+speech_path)
            shift_time=np.random.random()*(duration-config.Max_Len) #可供移动的时间长度
            shift_frames_image=int(shift_time*25)
            shift_frames_speech=4*shift_frames_image
            speech_feats=speech_feats[shift_frames_speech:int(shift_frames_speech+config.Max_Len*100)]
        else:
            speech_feats=np.load(config.aim_path+speech_path)
        print('Enter into the model:',speech_feats.shape)
        speech_feats=Variable(torch.tensor(speech_feats))

        try:
            if use_cuda:
                speech_feats=speech_feats.cuda()
            model.zero_grad()
            predict_score,mask=model(speech_feats)
            if not config.class_frame:
                predict_label=torch.argmax(predict_score,1).squeeze().item()
            else:
                #这是在一共t个帧里
                predict_idx=torch.argmax(predict_score).item()
                predict_label=predict_idx%len(config.spks_list)

            target_spk=torch.tensor(config.spks_list.index(spk_name)).unsueeze(0)

            if predict_label==config.spks_list.index(spk_name):
                right_idx_per_epoch+=1
                right_idx_everyXupdates+=1
            success_counte+=1
            # count every XXX updates
            count_interval=100
            if success_counte!=0 and success_counte%count_interval==0: # count the performance every XXX updates
                acc_this_interval=right_idx_everyXupdates/float(count_interval)
                logging("time: %6.3f, epoch: %3d, updates: %8d, valid loss this batch: %6.3f,valid acc this batch: %6.3f\n"
                        % (time.time()-start_time, epoch, updates, total_loss_batch/float(count_interval),right_idx_everyXupdates/float(count_interval)))
                total_loss_batch=0
                right_idx_everyXupdates=0
                lera.log(
                    'Acc_valid',acc_this_interval
                )

            if use_cuda:
                target_spk=target_spk.cuda()
                if not config.class_frame:
                    target_spk=target_spk.unsqueeze(0)

            loss=loss_func(predict_score,target_spk)

            loss_total+=loss.item()
            total_loss_batch+=loss.item()

            print('loss this seq: ',loss.item())

            updates += 1

        except  RuntimeError as RR:
            print('EEE errors here: ',RR)
            continue

        lera.log(
            'loss_valid',loss.item(),
        )

    print('Loss valid aver for this epoch:',loss_total/float(success_counte))
    print('Acc valid aver for this epoch:',right_idx_per_epoch/float(success_counte))
    lera.log(
        'Acc_valid_all',right_idx_per_epoch/float(success_counte))
    lera.log(
        'Loss_valid_all',loss_total/float(success_counte))
    print('Valid ends in epoch',epoch)

def main():
    train_data,eval_data=None,None
    for i in range(1, config.EPOCH_SIZE+1):
        if not opt.notrain:
            if not train_data:
                train_data=prepare_data('once','all_train')
                if config.only_1_meet:
                    train_data=[i for i in train_data if 'TS3' in i['speech_path']]
                    # train_data=[i for i in train_data if i['speech_path'][0]!='E']
                print('Train data gets items of: ',len(train_data))
                # all_spk_list=set([one['spk_name'] for  one in train_data])
                # print(len(all_spk_list))
                # print(all_spk_list)
                # assert len(all_spk_list)==all_spk_num
                assert len(set([one['spk_name'] for  one in train_data]))==all_spk_num
            train(i,train_data)
        else:
            if not eval_data:
                eval_data=prepare_data('once','all_valid')
                if config.only_1_meet:
                    eval_data=[i for i in eval_data if 'TS3' in i['speech_path']]
                print('EVAL data gets items of: ',len(eval_data))
                # print(len(set([one['spk_name'] for  one in eval_data])))
                # assert len(set([one['spk_name'] for  one in eval_data]))==all_spk_num
            eval(i,eval_data)
    for metric in config.metric:
        logging("Best %s score: %.2f\n" % (metric, max(scores[metric])))

if __name__ == '__main__':
    main()
