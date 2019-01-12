# <-*- encoding:utf8 -*->

"""
    Configuration Profile
"""

import time
import soundfile as sf
import resampy
import numpy as np
from scipy import signal

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
aim_path='/home/user/shijing/disk0/aim_sets/'
# 日志记录，Record log into this file, such as dl4ss_output.log_20170303_110305
LOG_FILE_PRE = './av4ss_output.'+time.strftime('%Y-%m-%d %H:%M:%S')+'.log'
# 训练文件列表
# TRAIN_LIST = aim_path+'/train_list'
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
    'TS3012b', 'TS3012c', 'TS3012d' ] # 115 in total
# 验证文件列表
# VALID_LIST = aim_path+'/valid_list'
# VALID_LIST = True
VALID_LIST =[
    'ES2003a', 'ES2003b', 'ES2003c', 'ES2003d', 'ES2011a', 'ES2011b', 'ES2011c', 'ES2011d',
    'IB4001', 'IS1008a', 'IS1008b', 'IS1008c', 'IS1008d', 'TS3004a', 'TS3004b', 'TS3004c',
    'TS3004d', 'TS3006a', 'TS3006b', 'TS3006c', 'TS3006d'] # 21 in total
# 测试文件列表
# TEST_LIST = aim_path+'/test_list'
# TEST_LIST = True
TEST_LIST = [
    'EN2002b', 'EN2002d', 'ES2004a', 'ES2004b', 'ES2004c', 'ES2004d', 'ES2014a', 'ES2014b',
    'ES2014c', 'ES2014d', 'IS1009a', 'IS1009b', 'IS1009c', 'IS1009d', 'TS3003a', 'TS3003b',
    'TS3003c', 'TS3003d', 'TS3007a', 'TS3007b', 'TS3007c', 'TS3007d'] # 22 in total
# 未登录文件列表
UNK_LIST = aim_path+'/unk_list'
spks_list=['FEE005', 'FEE013', 'FEE016', 'FEE019', 'FEE021', 'FEE024', 'FEE028', 'FEE029',
           'FEE030', 'FEE032', 'FEE036', 'FEE037', 'FEE038', 'FEE039', 'FEE040', 'FEE041',
           'FEE042', 'FEE043', 'FEE044', 'FEE046', 'FEE047', 'FEE049', 'FEE050', 'FEE051',
           'FEE052', 'FEE055', 'FEE057', 'FEE058', 'FEE059', 'FEE060', 'FEE064', 'FEE078',
           'FEE080', 'FEE081', 'FEE083', 'FEE085', 'FEE087', 'FEE088', 'FEO023', 'FEO026',
           'FEO065', 'FEO066', 'FEO070', 'FEO072', 'FEO079', 'FEO084', 'FIE038', 'FIE073',
           'FIE081', 'FIE088', 'FIO017', 'FIO041', 'FIO074', 'FIO084', 'FIO087', 'FIO089',
           'FIO093', 'FTD019UID', 'MEE006', 'MEE007', 'MEE008', 'MEE009', 'MEE010', 'MEE011',
           'MEE012', 'MEE014', 'MEE017', 'MEE018', 'MEE025', 'MEE027', 'MEE031', 'MEE033', 'MEE034',
           'MEE035', 'MEE045', 'MEE048', 'MEE053', 'MEE054', 'MEE056', 'MEE061', 'MEE063', 'MEE067',
           'MEE068', 'MEE071', 'MEE073', 'MEE089', 'MEE094', 'MEE095', 'MEO015', 'MEO020', 'MEO022',
           'MEO062', 'MEO069', 'MEO082', 'MEO086', 'MIE002', 'MIE029', 'MIE034', 'MIE080', 'MIE083',
           'MIE085', 'MIE090', 'MIO005', 'MIO008', 'MIO012', 'MIO016', 'MIO018', 'MIO019', 'MIO020',
           'MIO022', 'MIO023', 'MIO024', 'MIO025', 'MIO026', 'MIO031', 'MIO035', 'MIO040', 'MIO043',
           'MIO047', 'MIO049', 'MIO050', 'MIO055', 'MIO066', 'MIO072', 'MIO075', 'MIO076', 'MIO077',
           'MIO078', 'MIO082', 'MIO086', 'MIO091', 'MIO092', 'MIO097', 'MIO098', 'MIO099', 'MIO100',
           'MIO101', 'MIO104', 'MIO105', 'MIO106', 'MTD0010ID', 'MTD009PM', 'MTD011UID', 'MTD012ME',
           'MTD013PM', 'MTD014ID', 'MTD015UID', 'MTD016ME', 'MTD017PM', 'MTD018ID', 'MTD020ME',
           'MTD021PM', 'MTD022ID', 'MTD023UID', 'MTD024ME', 'MTD025PM', 'MTD026UID', 'MTD027ID',
           'MTD028ME', 'MTD029PM', 'MTD030ID', 'MTD031UID', 'MTD032ME', 'MTD033PM', 'MTD034ID',
           'MTD035UID', 'MTD036ME', 'MTD037PM', 'MTD038ID', 'MTD039UID', 'MTD040ME', 'MTD041PM',
           'MTD042ID', 'MTD043UID', 'MTD044ME', 'MTD045PM', 'MTD046ID', 'MTD047UID', 'MTD048ME']


optim = 'adam'
# optim = 'adagrad'
# learning_rate = 0.002
learning_rate = 0.0002
# learning_rate = 0.00005
learning_rate = 0.00001
max_grad_norm = 10
learning_rate_decay = 0.5
save_inter=5000

schedule = 1
bidirec = 1
start_decay_at = 5
log ='log/'

mask_softmax = 0

image_time_conv = 1
images_recu=0

mask_conv_bias=1
mask_over_init=1
only_1_meet=1
class_frame=1 #是否每个frame最后的时候单独预测
mask_norm=1
# threshold=0.3
threshold=0
mask_topk=5
size_sum=1


SEED = 1234
Min_Len = 0.5
Max_Len = 3.0

speech_fre=257
image_size=(13,13)
size_of_all_spks=179

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


# print('\n','*'*40)
# print('All the params:')
# print('*'*40)
# cc=locals().items()
# print(cc)
# print('*'*100,'\n')
