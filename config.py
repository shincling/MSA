# <-*- encoding:utf8 -*->

"""
    Configuration Profile
"""

import time
import soundfile as sf
import resampy
import numpy as np
from scipy import signal

def valid_mode_dataset():
    if MODE==1:
        if DATASET not in ['THCHS-30','WSJ0','TIMIT']:
            raise ValueError("Dataset {} is not Speech!!!".format(DATASET))
    if MODE==2:
        if DATASET not in ['MNIST']:
            raise ValueError("Dataset {} is not Image!!!".format(DATASET))
    if MODE==3:
        if DATASET not in ['AVA','GRID']:
            raise ValueError("Dataset {} is not Video!!!".format(DATASET))
    if MODE==4:
        print('Top-down Query.')

def update_max_len(file_path_list, max_len):
    tmp_max_len = 0
    # 用于搜集不同语音的长度
    signal_set = set()
    for file_path in file_path_list:
        file_list = open(file_path)
        for line in file_list:
            line = line.strip().split()
            if len(line) < 3:
                print('Wrong audio list file record in the line:', line)
                continue
            file_str = line[0]
            if file_str in signal_set:
                continue
            signal_set.add(file_str)
            signal, rate = sf.read(file_str)  # signal 是采样值，rate 是采样频率
            if len(signal.shape) > 1:
                signal = signal[:, 0]
            if rate != FRAME_RATE:
                # 如果频率不是设定的频率则需要进行转换
                signal = resampy.resample(signal, rate, FRAME_RATE, filter='kaiser_fast')
            if len(signal) > tmp_max_len:
                tmp_max_len = len(signal)
        file_list.close()
    if tmp_max_len < max_len:
        #原来这个max_len是认为设定的23秒8K的大小，是一个预设的极限值
        #这里就是所有的speech的长度找出最大的和这个比，最大不能超过23秒这个。
        max_len = tmp_max_len
    return max_len



# 判断是否加载
HAS_INIT_CONFIG = False
MAT_ENG = []
# External configuration file
CONFIG_FILE = './config.cfg'
# mode=1 纯净语音刺激, 2 图片刺激, 3 视频刺激, 4 top-down概念刺激
MODE = 3
# 数据集
# 1包括：THCHS-30 或者 WSJ0, TIMIT做为模型调试
# 2包括：ＭNIST
# 3包括：AVA,GRID
# 4包括：
DATASET = 'GRID'
valid_mode_dataset() #判断MODE和数据集是否对应，不对就抛出异常
# aim_path='./Dataset_Multi/'+str(MODE)+'/'+DATASET
# aim_path='/media/sw/Elements/数据集/Grid/Dataset_Multi/'+str(MODE)+'/'+DATASET
# aim_path='../Torch_multi/Dataset_Multi/'+str(MODE)+'/'+DATASET
# aim_path='./Dataset/'+str(MODE)+'/'+DATASET
aim_path='/home/user/aim_sets/
# 日志记录，Record log into this file, such as dl4ss_output.log_20170303_110305
LOG_FILE_PRE = './av4ss_output.'+time.strftime('%Y-%m-%d %H:%M:%S')+'.log'
# 训练文件列表
TRAIN_LIST = aim_path+'/train_list'
# TRAIN_LIST = True
TRAIN_LIST =[
    'EN2001a', 'EN2001b', 'EN2001d', 'EN2001e', 'EN2004a', 'EN2005a', 'EN2006a', 'EN2006b',
    'EN2009b', 'EN2009c', 'ES2002a', 'ES2002b', 'ES2002c', 'ES2002d', 'ES2005a', 'ES2005b',
    'ES2005c', 'ES2005d', 'ES2006a', 'ES2006b', 'ES2006c', 'ES2006d', 'ES2007a', 'ES2007b',
    'ES2007c', 'ES2007d', 'ES2008a', 'ES2008b', 'ES2008c', 'ES2008d', 'ES2009a', 'ES2009b',
    'ES2009c', 'ES2009d', 'ES2010a', 'ES2010b', 'ES2010c', 'ES2010d', 'ES2012a', 'ES2012b',
    'ES2012c', 'ES2012d', 'ES2013a', 'ES2013b', 'ES2013c', 'ES2013d', 'ES2015a', 'ES2015b',
    'ES2015c', 'ES2015d', 'ES2016a', 'ES2016b', 'ES2016c', 'ES2016d', 'IN1001', 'IN1002',
    'IN1005' , 'IN1007' , 'IN1008' , 'IN1009' , 'IN1012' , 'IN1013' , 'IN1014', 'IN1016',
    'IS1000a', 'IS1000b', 'IS1000c', 'IS1000d', 'IS1001a', 'IS1001b', 'IS1001c', 'IS1001d',
    'IS1002b', 'IS1002c', 'IS1002d', 'IS1003a', 'IS1003c', 'IS1003d', 'IS1004a', 'IS1004b',
    'IS1004c', 'IS1004d', 'IS1005a', 'IS1005b', 'IS1005c', 'IS1006a', 'IS1006b', 'IS1006c',
    'IS1006d', 'IS1007a', 'IS1007b', 'IS1007c', 'TS3005a', 'TS3005b', 'TS3005c', 'TS3005d',
    'TS3008a', 'TS3008b', 'TS3008c', 'TS3008d', 'TS3009a', 'TS3009b', 'TS3009d', 'TS3010a',
    'TS3010b', 'TS3010c', 'TS3010d', 'TS3011a', 'TS3011b', 'TS3011c', 'TS3011d', 'TS3012a',
    'TS3012b', 'TS3012c', 'TS3012d' ] #115 in total
# 验证文件列表
VALID_LIST = aim_path+'/valid_list'
# VALID_LIST = True
VALID_LIST =[
    'ES2003a', 'ES2003b', 'ES2003c', 'ES2003d', 'ES2011a', 'ES2011b', 'ES2011c', 'ES2011d',
    'IB4001', 'IS1008a', 'IS1008b', 'IS1008c', 'IS1008d', 'TS3004a', 'TS3004b', 'TS3004c',
    'TS3004d', 'TS3006a', 'TS3006b', 'TS3006c', 'TS3006d'] # 21 in total
# 测试文件列表
TEST_LIST = aim_path+'/test_list'
# TEST_LIST = True
TEST_LIST = [
    'EN2002b', 'EN2002d', 'ES2004a', 'ES2004b', 'ES2004c', 'ES2004d', 'ES2014a', 'ES2014b',
    'ES2014c', 'ES2014d', 'IS1009a', 'IS1009b', 'IS1009c', 'IS1009d', 'TS3003a', 'TS3003b',
    'TS3003c', 'TS3003d', 'TS3007a', 'TS3007b', 'TS3007c', 'TS3007d'] # 22 in total
# 未登录文件列表
UNK_LIST = aim_path+'/unk_list'

Num_samples_per_epoch=2000 #如果没有预订提供的list,则设定一个Epoch里的训练样本数目
# 是否读取参数
Load_param = True
#Load_param = False
Save_param = True
# Load_param = False
# 是否在训练阶段用Ground Truth的分类结果
Ground_truth = True
#query是否经过memory的再次更新
Comm_with_Memory=True
Comm_with_Memory=False
# DNN/RNN隐层的维度 hidden units
HIDDEN_UNITS = 300
HIDDEN_UNITS = 200
FC_UNITS = 600
# DNN/RNN层数
# NUM_LAYERS = 2
NUM_LAYERS = 1 #这里好像是一层
# Embedding大小,主要是给语音第三维那部分的
EMBEDDING_SIZE = 50
# EMBEDDING_SIZE = 40
# 是否丰富数据
AUGMENT_DATA = False
# AUGMENT_DATA = True
# set the max epoch of training
MAX_EPOCH = 600
# epoch size
EPOCH_SIZE = 10
EPOCH_SIZE = 300
# batch size
BATCH_SIZE = 8
# 评估的batch size
BATCH_SIZE_EVAL = 10
# feature frame rate
FRAME_RATE = 2*8000
# 帧时长(ms)
FRAME_LENGTH = int(0.032 * FRAME_RATE)
FRAME_LENGTH = int(0.010 * FRAME_RATE)
# 帧移(ms)
FRAME_SHIFT = int(0.016 * FRAME_RATE)
# 是否shuffle_batch
SHUFFLE_BATCH = True
# 设定最小混叠说话人数，Minimum number of mixed speakers for training
MIN_MIX = 2
# 设定最大混叠说话人数，Maximum number of mixed speakers for training
MAX_MIX = 2
# 设定speech multi acc的阈值alpha
ALPHA = 0.5
quchong_alpha=1
dB=5
# 设置训练/开发/验证模型的最大语音长度(秒)
MAX_LEN = 3
MAX_LEN_SPEECH = FRAME_RATE*MAX_LEN
#MAX_LEN = update_max_len([TRAIN_LIST], MAX_LEN)
# 帧长
WINDOWS = FRAME_LENGTH
FFT_SIZE=512
HOP_LEN=int(0.010 * FRAME_RATE)
WIN_LEN=int(0.025* FRAME_RATE)
 # 训练模型权重的目录
TMP_WEIGHT_FOLDER = aim_path+'/_tmp_weights'
# 未登录说话人, False: 由说话人提取记忆，True: 进行语音抽取，# TODO NONE: 有相似度计算确定是否进行语音更新
# 更新，UNK_SPK用spk idx=0替代
UNK_SPK = False
# 未登录说话人语音的最大额外条数
UNK_SPK_SUPP = 3
START_EALY_STOP = 0
# 特征Spectral of Log Spectral
IS_LOG_SPECTRAL = False
IS_POWER = 1
# DB_THRESHOLD = 40  # None
# 添加背景噪音（Str）
ADD_BGD_NOISE = False
BGD_NOISE_WAV = None
BGD_NOISE_FILE = 'Dataset_Multi/BGD_150203_010_STR.CH1.wav'
Out_Sep_Result=True

# VideoSize=(299,299)
# NUM_ALL_FRAMES=25
# VIDEO_RATE=10
# channel_first=True
if MODE==2:
    '''Params for Image'''
    ImageSize=(28,28)
elif MODE==3:
    '''Params for Video'''
    VideoSize=(299,299)
    NUM_ALL_FRAMES=25
    VIDEO_RATE=25
    channel_first=True
    MAX_LEN_VIDEO=MAX_LEN*VIDEO_RATE

def load_bgd_wav(file_path):
    signal, rate = sf.read(file_path)  # signal 是采样值，rate 是采样频率
    if len(signal.shape) > 1:
        signal = signal[:, 0]
    if rate != FRAME_RATE:
        # 如果频率不是设定的频率则需要进行转换
        signal = resampy.resample(signal, rate, FRAME_RATE, filter='kaiser_fast')
    return signal

print('\n','*'*40)
print('All the params:')
print('*'*40)
cc=locals().items()
print(cc)
print('*'*100,'\n')
