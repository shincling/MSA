#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SPEECH_MODEL(nn.Module):
    def __init__(self, config ):
        super(SPEECH_MODEL, self).__init__()
        self.cnn1=nn.Conv2d(2,96,(1,7),stride=[1],padding=(0,3),dilation=(1,1))
        self.cnn2=nn.Conv2d(96,96,(7,1),stride=1,padding=(3,0),dilation=(1,1))
        self.cnn3=nn.Conv2d(96,96,(5,5),stride=1,padding=(2,2),dilation=(1,1))
        self.cnn4=nn.Conv2d(96,96,(5,5),stride=1,padding=(4,2),dilation=(2,1))
        self.cnn5=nn.Conv2d(96,96,(5,5),stride=1,padding=(8,2),dilation=(4,1))

        self.cnn6=nn.Conv2d(96,96,(5,5),stride=1,padding=(16,2),dilation=(8,1))
        self.cnn7=nn.Conv2d(96,96,(5,5),stride=1,padding=(32,2),dilation=(16,1))
        self.cnn8=nn.Conv2d(96,96,(5,5),stride=1,padding=(64,2),dilation=(32,1))
        self.cnn9=nn.Conv2d(96,96,(5,5),stride=1,padding=(2,2),dilation=(1,1))
        self.cnn10=nn.Conv2d(96,96,(5,5),stride=1,padding=(4,4),dilation=(2,2))

        self.cnn11=nn.Conv2d(96,96,(5,5),stride=1,padding=(8,8),dilation=(4,4))
        self.cnn12=nn.Conv2d(96,96,(5,5),stride=1,padding=(16,16),dilation=(8,8))
        self.cnn13=nn.Conv2d(96,96,(5,5),stride=1,padding=(32,32),dilation=(16,16))
        self.cnn14=nn.Conv2d(96,96,(5,5),stride=1,padding=(64,64),dilation=(32,32))
        self.cnn15=nn.Conv2d(96,8,(1,1),stride=1,padding=(0,0),dilation=(1,1))
        self.num_cnns=15
        self.bn1=nn.BatchNorm2d(96)
        self.bn2=nn.BatchNorm2d(96)
        self.bn3=nn.BatchNorm2d(96)
        self.bn4=nn.BatchNorm2d(96)
        self.bn5=nn.BatchNorm2d(96)
        self.bn6=nn.BatchNorm2d(96)
        self.bn7=nn.BatchNorm2d(96)
        self.bn8=nn.BatchNorm2d(96)
        self.bn9=nn.BatchNorm2d(96)
        self.bn10=nn.BatchNorm2d(96)
        self.bn11=nn.BatchNorm2d(96)
        self.bn12=nn.BatchNorm2d(96)
        self.bn13=nn.BatchNorm2d(96)
        self.bn14=nn.BatchNorm2d(96)
        self.bn15=nn.BatchNorm2d(8)

    def forward(self, x):
        # Speech:[4t,257,2]
        x=torch.transpose(torch.transpose(x,0,2),1,2).unsqueeze(0)
        print(x.shape)
        print('\nSpeech layer log:')
        x = x.contiguous()
        for idx in range(self.num_cnns):
            cnn_layer=eval('self.cnn{}'.format(idx+1))
            bn_layer=eval('self.bn{}'.format(idx+1))
            x=F.relu(cnn_layer(x))
            x=bn_layer(x)
            print('speech shape after CNNs:',idx,'', x.size())
        return x


class basic_model(nn.Module):
    def __init__(self, config,tgt_spk_size, use_cuda ):
        super(basic_model, self).__init__()

        self.speech_model=SPEECH_MODEL(config,)
        # self.images_model=IMAGES_MODEL(config,)
        # self.output_model=MERGE_MODEL(config,)
        self.use_cuda = use_cuda
        self.tgt_spk_size = tgt_spk_size
        self.config = config
        self.loss_for_ss= nn.MSELoss()
        self.log_softmax = nn.LogSoftmax()

    def forward(self,input_image,input_speech,):
        # Image:[t,1024,3,13],Speech:[4t,257,2]

        speech_hidden=self.speech_model(input_speech)
        1/0


        return None

