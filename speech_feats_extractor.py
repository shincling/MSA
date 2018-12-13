#coding=utf8
import sys
import os
import numpy as np
import config as config
import soundfile as sf
import resampy
import librosa
from tqdm import tqdm

def convert2(array):
    shape=array.shape
    o=array.real.reshape(shape[0],shape[1],1).repeat(2,2)
    o[:,:,1]=array.imag
    return o

def process_uris(data_path): # func for every uri
    print('*'*40)
    print('Begin to process',data_path)
    views=os.listdir(data_path)
    for view in views: # every different view
        print('Begin to process:', data_path + '/' + view)
        for name in os.listdir(data_path + '/' + view):
            if '.avi' in name:
                break
        else:
            print('No avi in ', data_path + '/' + view)
            continue

        images_npy_ids=[]
        for name in os.listdir(data_path+'/'+view):
            if '.npy' in name:
                images_npy_ids.append(name.replace('.npy',''))
        ids=sorted(images_npy_ids)
        for id in ids: # different part of the meeting
            speech_name=data_path+'/'+view+'/'+id+'.wav'
            signal, rate = sf.read(speech_name)  # signal 是采样值，rate 是采样频率
            if len(signal.shape) > 1:
                signal = signal[:, 0]
            if rate != config.FRAME_RATE:
                # 如果频率不是设定的频率则需要进行转换
                signal = resampy.resample(signal, rate, config.FRAME_RATE, filter='kaiser_best')
            # if signal.shape[0] < config.MIN_LEN_SPEECH:  # 过滤掉长度少于极限的
            #     continue
            #     shift_len=random.randint(0,signal.shape[0]-config.MAX_LEN_SPEECH)
            #     signal = signal[shift_len:(shift_len+config.MAX_LEN_SPEECH)]
            # 更新混叠语音长度
            # if signal.shape[0] > mix_len:
            #     mix_len = signal.shape[0]
            # if signal.shape[0] < config.MAX_LEN_SPEECH:  # 根据最大长度用 0 补齐,
            #     signal=np.append(signal,np.zeros(config.MAX_LEN_SPEECH - signal.shape[0]))

            signal -= np.mean(signal)  # 语音信号预处理，先减去均值
            signal /= np.max(np.abs(signal))  # 波形幅值预处理，幅值归一化

            # 这里采用log 以后可以考虑采用MFCC或GFCC特征做为输入
            if config.IS_LOG_SPECTRAL:
                feature_speech = np.log(np.transpose((librosa.core.spectrum.stft(signal, config.FRAME_LENGTH,
                                                                              config.FRAME_SHIFT,
                                                                              window=config.WINDOWS)))
                                     + np.spacing(1))
            else:
                feature_speech = np.transpose((librosa.core.spectrum.stft(signal, config.FFT_SIZE, config.HOP_LEN,
                                                                       config.WIN_LEN,)))
                if config.IS_POWER:
                    feature_speech=pow(feature_speech,0.3)
                feature_speech=convert2(feature_speech)

            #print(speech_name,' audio size:',feature_speech.shape)
            #print(speech_name,' video size:',(np.load(speech_name.replace('wav','npy'))).shape)

            try:
                audio_length=feature_speech.shape[0]
                aim_length=4*np.load(speech_name.replace('wav','npy')).shape[0]
                if audio_length<aim_length: #　现在的比较短，就补足
                    for ___ in range(aim_length-audio_length):
                        feature_speech=np.append(feature_speech,feature_speech[-1:],0)
                elif audio_length>aim_length: # 现在的长了，就减短
                    feature_speech=feature_speech[:aim_length]
                assert feature_speech.shape[0]==aim_length
                np.save(data_path+'/'+view+'/'+id+'_Audio.npy',feature_speech)
            except Exception as rr:
                print('EEE',rr)
                continue



prepared_urils=['IN1013', 'IN1014', 'IN1016', 'IS1000a', 'IS1000b', 'IS1000c', 'IS1000d',
                'IS1001a', 'IS1001b', 'IS1001c', 'IS1001d', 'IS1002b', 'IS1002c', 'IS1002d',
                'IS1003a', 'IS1003c', 'IS1003d', 'IS1004a', 'IS1004b', 'IS1004c', 'IS1004d',
                'IS1005a', 'IS1005b', 'IS1005c', 'IS1006a', 'IS1006b', 'IS1006c', 'IS1006d',
                'IS1007a', 'IS1007b', 'IS1007c','TS3003b', 'TS3003c', 'TS3003d', 'TS3004a',
                'TS3004b', 'TS3004c', 'TS3004d', 'TS3005a', 'TS3005b', 'TS3005c', 'TS3005d', 'TS3006a']
prepared_urils=os.listdir('/home/user/shijing/disk0/aim_sets/')
print('All prepared_urils:\n',prepared_urils)
# for uri in tqdm(prepared_urils[43:]):
for uri in tqdm(prepared_urils[43:]):
    if uri not in os.listdir('/home/user/shijing/disk0/aim_sets/'):
        print('No uri',uri,'\n')
        continue
    data_path='/home/user/shijing/disk0/aim_sets/'+uri
    process_uris(data_path)

