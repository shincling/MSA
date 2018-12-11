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


def process_uris(data_path): # func for every uri
    views=os.listdir(data_path)
    for view in views: # every different view
        images_npy_ids=[]
        for name in os.listdir(data_path+'/'+view):
            if '.npy' in name:
                images_npy_ids.append(name.replace('.npy',''))
        ids=sorted(images_npy_ids)
        for id in ids: # different part of the meeting
            speech_name=data_path+'/'+view+'/'+id
            signal, rate = sf.read(speech_name)  # signal 是采样值，rate 是采样频率
            if len(signal.shape) > 1:
                signal = signal[:, 0]
            if rate != config.FRAME_RATE:
                # 如果频率不是设定的频率则需要进行转换
                signal = resampy.resample(signal, rate, config.FRAME_RATE, filter='kaiser_best')
            if signal.shape[0] > config.MAX_LEN_SPEECH:  # 根据最大长度,裁剪
                shift_len=random.randint(0,signal.shape[0]-config.MAX_LEN_SPEECH)
                signal = signal[shift_len:(shift_len+config.MAX_LEN_SPEECH)]
            # 更新混叠语音长度
            if signal.shape[0] > mix_len:
                mix_len = signal.shape[0]

            signal -= np.mean(signal)  # 语音信号预处理，先减去均值
            signal /= np.max(np.abs(signal))  # 波形幅值预处理，幅值归一化

            # 如果需要augment数据的话，先进行随机shift, 以后考虑固定shift
            if config.AUGMENT_DATA and train_or_test=='train':
                random_shift = random.sample(range(len(signal)), 1)[0]
                signal = np.append(signal[random_shift:], signal[:random_shift])

            if signal.shape[0] < config.MAX_LEN_SPEECH:  # 根据最大长度用 0 补齐,
                signal=np.append(signal,np.zeros(config.MAX_LEN_SPEECH - signal.shape[0]))

            if k==0:#第一个作为目标
                ratio=10**(aim_spk_db_k[k]/20.0)
                signal=ratio*signal
                aim_spkname.append(aim_spk_k[0])
                # aim_spk=eval(re.findall('\d+',aim_spk_k[0])[0])-1 #选定第一个作为目标说话人
                #TODO:这里有个问题是spk是从１开始的貌似，这个后面要统一一下　-->　已经解决，构建了spk和idx的双向索引
                aim_spk_speech=signal
                aim_spkid.append(aim_spkname)
                wav_mix=signal
                # print signal.shape
                aim_fea_clean = np.transpose((librosa.core.spectrum.stft(signal, config.FFT_SIZE, config.HOP_LEN,
                                                                         config.WIN_LEN)))
                if config.IS_POWER:
                    aim_fea_clean=pow(aim_fea_clean,0.3)
                aim_fea_clean=convert2(aim_fea_clean)
                # print aim_fea_clean.shape
                #TODO:这个实现出来跟原文不太一样啊，是２５７×３０１（原文是２９８）
                aim_fea.append(aim_fea_clean)
                # 把第一个人顺便也注册进去混合dict里
                multi_fea_dict_this_sample[spk]=aim_fea_clean
                multi_wav_dict_this_sample[spk]=signal

                aim_spk_fea_video_path=data_path+'/face_emb/s2-s35/'+spk+'_imgnpy/'+sample_name+'.npy'
                multi_video_fea_dict_this_sample[spk]=np.load(aim_spk_fea_video_path)

            else:
                ratio=10**(aim_spk_db_k[k]/20.0)
                signal=ratio*signal
                wav_mix = wav_mix + signal  # 混叠后的语音
                #　这个说话人的语音
                some_fea_clean = np.transpose((librosa.core.spectrum.stft(signal, config.FFT_SIZE, config.HOP_LEN,
                                                                          config.WIN_LEN)))
                if config.IS_POWER:
                    some_fea_clean=pow(some_fea_clean,0.3)
                some_fea_clean=convert2(some_fea_clean)
                multi_fea_dict_this_sample[spk]=some_fea_clean
                multi_wav_dict_this_sample[spk]=signal

                aim_spk_fea_video_path=data_path+'/face_emb/s2-s35/'+spk+'_imgnpy/'+sample_name+'.npy'
                multi_video_fea_dict_this_sample[spk]=np.load(aim_spk_fea_video_path)


            multi_spk_fea_list.append(multi_fea_dict_this_sample) #把这个sample的dict传进去
            multi_spk_wav_list.append(multi_wav_dict_this_sample) #把这个sample的dict传进去
            multi_video_list.append(multi_video_dict_this_sample) #把这个sample的dict传进去
            multi_video_fea_list.append(multi_video_fea_dict_this_sample) #把这个sample的dict传进去

            # 这里采用log 以后可以考虑采用MFCC或GFCC特征做为输入
            if config.IS_LOG_SPECTRAL:
                feature_mix = np.log(np.transpose((librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                              config.FRAME_SHIFT,
                                                                              window=config.WINDOWS)))
                                     + np.spacing(1))
            else:
                feature_mix = np.transpose((librosa.core.spectrum.stft(wav_mix, config.FFT_SIZE, config.HOP_LEN,
                                                                       config.WIN_LEN,)))
                if config.IS_POWER:
                    feature_mix=pow(feature_mix,0.3)
                feature_mix=convert2(feature_mix)

            mix_speechs[batch_idx,:]=wav_mix
            mix_feas.append(feature_mix)
            # mix_phase.append(np.transpose(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
            #                                                                      config.FRAME_SHIFT,)))
            mix_phase.append(np.transpose(librosa.core.spectrum.stft(wav_mix, config.FFT_SIZE, config.HOP_LEN,
                                                                     config.WIN_LEN,)))
            if config.IS_POWER:
                mix_phase[-1]=pow(mix_phase[-1],0.3)
            batch_idx+=1
            # print 'batch_dix:{}/{},'.format(batch_idx,config.BATCH_SIZE),
            if batch_idx==config.BATCH_SIZE: #填满了一个batch
                mix_feas=np.array(mix_feas)
                mix_phase=np.array(mix_phase)
                aim_fea=np.array(aim_fea)
                # aim_spkid=np.array(aim_spkid)
                query=np.array(query)
                print 'spk_list_from_this_gen:{}'.format(aim_spkname)
                print 'aim spk list:', [one.keys() for one in multi_spk_fea_list]
                # print '\nmix_speechs.shape,mix_feas.shape,aim_fea.shape,aim_spkname.shape,query.shape,all_spk_num:'
                # print mix_speechs.shape,mix_feas.shape,aim_fea.shape,len(aim_spkname),query.shape,len(all_spk)
                if mode=='global':
                    all_spk=sorted(all_spk)
                    all_spk=sorted(all_spk_train)
                    all_spk_eval=sorted(all_spk_eval)
                    all_spk_test=sorted(all_spk_test)
                    dict_spk_to_idx={spk:idx for idx,spk in enumerate(all_spk)}
                    dict_idx_to_spk={idx:spk for idx,spk in enumerate(all_spk)}
                    yield all_spk,dict_spk_to_idx,dict_idx_to_spk, \
                          aim_fea.shape[1],aim_fea.shape[2],config.MAX_LEN_VIDEO,len(all_spk),batch_total
                    #上面的是：语音长度、语音频率、视频分割多少帧 TODO:后面把这个替换了query.shape[1]
                elif mode=='once':
                    yield {'mix_wav':mix_speechs,
                           'mix_feas':mix_feas,
                           'mix_phase':mix_phase,
                           'aim_fea':aim_fea,
                           'aim_spkname':aim_spkname,
                           'query':query,
                           'num_all_spk':len(all_spk),
                           'multi_spk_fea_list':multi_spk_fea_list,
                           'multi_spk_wav_list':multi_spk_wav_list,
                           'multi_video_list':multi_video_list,
                           'multi_video_fea_list':multi_video_fea_list,
                           'batch_total':batch_total,
                           'top_k':mix_k
                           }

                #下一个batch的混合说话人个数， 先调整一下
                mix_k=random.sample(mix_number_list,1)[0]
                batch_idx=0
                mix_speechs=np.zeros((config.BATCH_SIZE,config.MAX_LEN_SPEECH))
                mix_feas=[]#应该是bs,n_frames,n_fre这么多
                mix_phase=[]
                aim_fea=[]#应该是bs,n_frames,n_fre这么多
                aim_spkid=[] #np.zeros(config.BATCH_SIZE)
                aim_spkname=[]
                query=[]#应该是BATCH_SIZE，shape(query)的形式，用list再转换把
                multi_spk_fea_list=[]
                multi_spk_wav_list=[]
                multi_video_list=[]
                multi_video_fea_list=[]
                sample_idx[mix_k]+=1

prepared_urils=['IN1013', 'IN1014', 'IN1016', 'IS1000a', 'IS1000b', 'IS1000c', 'IS1000d',
                'IS1001a', 'IS1001b', 'IS1001c', 'IS1001d', 'IS1002b', 'IS1002c', 'IS1002d',
                'IS1003a', 'IS1003c', 'IS1003d', 'IS1004a', 'IS1004b', 'IS1004c', 'IS1004d',
                'IS1005a', 'IS1005b', 'IS1005c', 'IS1006a', 'IS1006b', 'IS1006c', 'IS1006d',
                'IS1007a', 'IS1007b', 'IS1007c','TS3003b', 'TS3003c', 'TS3003d', 'TS3004a',
                'TS3004b', 'TS3004c', 'TS3004d', 'TS3005a', 'TS3005b', 'TS3005c', 'TS3005d', 'TS3006a']
for uri in prepared_urils:
    data_path='~/shijing/disk0/aim_sets/'+uri
    process_uris(data_path)

