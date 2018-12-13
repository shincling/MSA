#coding=utf8
import sys
import os
import numpy as np
import time
import random
import config as config
import re
import soundfile as sf
import resampy
import librosa
import shutil
import subprocess
# import Image
from PIL import Image

channel_first=config.channel_first
# np.random.seed(1)#设定种子
# random.seed(1)

def extract_frames(video, dst):
    with open('video_log', "w") as ffmpeg_log:
        video_id = video.split("/")[-1].split(".")[0]
        if os.path.exists(dst):
            print " cleanup: " + dst + "/"
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   '-y',  # (optional) overwrite output file if it exists
                                   '-i', video,  # input file
                                   '-vf', "scale={}:{}".format(config.VideoSize[0],config.VideoSize[1]),  # input file
                                   '-r', str(config.VIDEO_RATE),  # samplling rate of the Video
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%03d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command, stdout=ffmpeg_log, stderr=ffmpeg_log)

def split_forTrainDevTest(spk_list,train_or_test):
    '''为了保证一个统一的训练和测试的划分标准，不得不用通用的一些方法来限定一下,
    这里采用的是用sorted先固定方法的排序，那么不论方法或者seed怎么设置，训练测试的划分标准维持不变，
    也就是数据集会维持一直'''
    length=len(spk_list)
    # spk_list=sorted(spk_list,key=lambda x:(x[1]))#这个意思是按照文件名的第二个字符排序
    # spk_list=sorted(spk_list)#这个意思是按照文件名的第1个字符排序,暂时采用这种
    spk_list=sorted(spk_list,key=lambda x:(x[-1]))#这个意思是按照文件名的最后一个字符排序
    #TODO:暂时用最后一个字符排序，这个容易造成问题，可能第一个比较不一样的，这个需要注意一下
    if train_or_test=='train':
        return spk_list[:int(round(0.7*length))]
    elif train_or_test=='valid':
        return spk_list[(int(round(0.7*length))+1):int(round(0.8*length))]
    elif train_or_test=='test':
        return spk_list[(int(round(0.8*length))+1):]
    else:
        raise ValueError('Wrong input of train_or_test.')

def prepare_datasize(gen):
    data=gen.next()
    #此处顺序是 mix_speechs.shape,mix_feas.shape,aim_fea.shape,aim_spkid.shape,query.shape
    #一个例子：(5, 17040) (5, 134, 129) (5, 134, 129) (5,) (5, 32, 400, 300, 3)
    #暂时输出的是：语音长度、语音频率数量、视频截断之后的长度
    print 'datasize:',data[1].shape[1],data[1].shape[2],data[4].shape[1],data[-1],(data[4].shape[2],data[4].shape[3])
    return data[1].shape[1],data[1].shape[2],data[4].shape[1],data[-1],(data[4].shape[2],data[4].shape[3])

def create_mix_list(train_or_test,mix_k,data_path,all_spk,Num_samples_per_batch):
    list_path=data_path+'/list_mixtures/'
    file_name=open(list_path+'faceemb_mix_{}_spk_{}.txt'.format(mix_k,train_or_test),'w')

    for i_line in range(Num_samples_per_batch):
        aim_spk_k=random.sample(all_spk,mix_k)#本次混合的候选人
        line=''
        ratio=round(5*np.random.rand()-2.5,3)
        for spk in aim_spk_k:
            sample_name=random.sample(os.listdir('{}/face_emb/s2-s35/{}_imgnpy/'.format(data_path,spk)),1)[0]
            sample_name=sample_name.replace('npy','wav')
            if spk==aim_spk_k[0]:
                line+='GRID/data/face_emb/voice/{}/{} {} '.format(spk,sample_name,ratio)
            elif spk==aim_spk_k[-1]:
                line+='GRID/data/face_emb/voice/{}/{} {} '.format(spk,sample_name,-1*ratio)
            else:
                line+='GRID/data/face_emb/voice/{}/{} 0.000 '.format(spk,sample_name)
        line+='\n'
        file_name.write(line)

def convert2(array):
    shape=array.shape
    o=array.real.reshape(shape[0],shape[1],1).repeat(2,2)
    o[:,:,1]=array.imag
    return o

def prepare_data(mode,train_or_test):
    '''
    :param
    mode: type str, 'global' or 'once' ， global用来获取全局的spk_to_idx的字典，所有说话人的列表等等
    train_or_test:type str, 'train','valid' or 'test'
     其中把每个文件夹每个人的按文件名的排序的前70%作为训练，70-80%作为valid，最后20%作为测试
    :return:
    '''
    all_list=[]
    #目标数据集的总data，底下应该存放分目录的文件夹，每个文件夹应该名字是sX
    data_path=config.aim_path
    meeting_ids=os.listdir(data_path)

    if train_or_test=='train':
        aim_meeting_ids=config.TRAIN_LIST
    elif train_or_test=='valid':
        aim_meeting_ids=config.VALID_LIST
    elif train_or_test=='test':
        aim_meeting_ids=config.TEST_LIST

    for aim_meeting in aim_meeting_ids:
        assert aim_meeting in meeting_ids
        meeting_path=data_path+aim_meeting+'/'
        views=os.listdir(meeting_path)
        for view in views: # for every different view
            print('*'*50)
            print('Begin to process:', view_path)
            view_path=meeting_path+view
            for name in os.listdir(view_path):
                if '.npy' in name:
                    break
            else:
                print('No extracted features in ', data_path + '/' + view)
                continue

            files_in_view=os.listdir(view_path)
            cutted_samples=[name.replace('.wav','').replace('.avi','').replace('_Audio.npy','').replace('.npy','') for name in files_in_view]
            ids = sorted(set(cutted_samples))
            for id in ids:  # different part of the meeting
                if (id+'.npy') in files_in_view and (id+'_Audio.npy') in files_in_view:
                    print('Got dual feats for ',id)
                else:
                    continue
                iidx,spk_name,start_time,end_time=id.split('_')
                all_list.append({
                    'speech_path':'/'.join(aim_meeting,view,id)+'_Audio.npy',
                    'images_path':'/'.join(aim_meeting,view,id)+'.npy',
                    'duration':float(end_time)-float(start_time),
                    'spk_name':spk_name
                })
    return all_list


cc=prepare_data('once','train')
print(cc)
time.sleep(10)