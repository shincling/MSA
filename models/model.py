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

class SPEECH_MODEL_1(nn.Module):
    def __init__(self, config ):
        super(SPEECH_MODEL_1, self).__init__()
        self.cnn1=nn.Conv2d(2,64,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn1=nn.BatchNorm2d(64)
        self.cnn2=nn.Conv2d(64,64,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn2=nn.BatchNorm2d(64)
        self.pool2=nn.MaxPool2d((2,2))

        self.cnn3=nn.Conv2d(64,128,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn3=nn.BatchNorm2d(128)
        self.cnn4=nn.Conv2d(128,128,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn4=nn.BatchNorm2d(128)
        self.pool4=nn.MaxPool2d((2,2))

        self.cnn5=nn.Conv2d(128,256,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn5=nn.BatchNorm2d(256)
        self.cnn6=nn.Conv2d(256,256,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn6=nn.BatchNorm2d(256)
        self.pool6=nn.MaxPool2d((1,2))

        self.cnn7=nn.Conv2d(256,512,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn7=nn.BatchNorm2d(512)
        self.cnn8=nn.Conv2d(512,512,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn8=nn.BatchNorm2d(512)
        self.pool8=nn.MaxPool2d((1,2))

        self.cnn9=nn.Conv2d(512,1024,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn9=nn.BatchNorm2d(1024)
        self.cnn10=nn.Conv2d(1024,1024,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn10=nn.BatchNorm2d(1024)
        self.pool10=nn.MaxPool2d((1,2))

        self.num_cnns=10
        self.pool_list=[2,4,6,8,10]

    def forward(self, x):
        # Speech:[4t,257,2]
        x=torch.transpose(torch.transpose(x,0,2),1,2).unsqueeze(0)
        print(x.shape) #[bs=1, 2, 152, 257]
        print('\nSpeech layer log:')
        x = x.contiguous()
        for idx in range(self.num_cnns):
            cnn_layer=eval('self.cnn{}'.format(idx+1))
            bn_layer=eval('self.bn{}'.format(idx+1))
            x=F.relu(cnn_layer(x))
            x=bn_layer(x)
            if idx+1 in self.pool_list:
                pool_layer=eval('self.pool{}'.format(idx+1))
                x=pool_layer(x)
            print('speech shape after CNNs:',idx,'', x.size())

        print('speech shape final:', x.size(),'\n')
        return x

class MERGE_MODEL(nn.Module):
    def __init__(self, config,tgt_spk_size):
        super(MERGE_MODEL, self).__init__()
        self.tgt_spk_size = tgt_spk_size
        self.config = config

        self.sp_pool=nn.MaxPool2d((1,8))
        self.sp_fc1=nn.Linear(1024,1024)
        self.sp_fc2=nn.Linear(1024,1024)

        self.im_conv1=nn.Conv2d(1024,1024,(1,1),stride=1,padding=(0,0),dilation=(1,1))
        self.im_conv2=nn.Conv2d(1024,1024,(1,1),stride=1,padding=(0,0),dilation=(1,1))

        self.mask_conv=nn.Conv2d(1,1,(1,1),stride=1,padding=(0,0),dilation=(1,1))
        self.pool_over_size=nn.AvgPool2d((config.image_size[0],config.image_size[1]))
    def forward(self, image_hidden,speech_hidden):
        config=self.config
        #Image:[t,1024,13,13],Speech:[1,1024,t,8]
        speech_hidden=self.sp_pool(speech_hidden).squeeze(-1) #[1,1024,t,1]-->[1,1024,t]
        speech_hidden=torch.transpose(speech_hidden,1,2) # [1,t,1024]
        speech_hidden=F.relu(self.sp_fc1(speech_hidden)) # [1,t,1024]
        speech_final=F.relu(self.sp_fc2(speech_hidden)) # [1,t,1024]
        print('Gets speech final: ',speech_final.shape)
        speech_final=speech_final.squeeze() #[t,1024]
        speech_final=speech_final.unsqueeze(1).unsqueeze(1) #[t,1,1,1024]
        speech_final=speech_final.expand(-1,config.image_size[0],config.image_size[1],-1) #[t,1,1,1024]
        speech_final=speech_final.contiguous().view(-1,1024,1) #[t,1,1,1024]
        # print('Gets speech final: ',speech_final.shape)

        image_hidden=self.im_conv1(image_hidden)
        image_final=self.im_conv2(image_hidden) #[t,1024,13,13]
        # print('Gets image final: ',image_final.shape)
        image_final=torch.transpose(torch.transpose(image_final,1,3),1,2) #[t,13,13,1024]
        print('Gets image final: ',image_final.shape)
        image_tmp=image_final.contiguous().view(-1,1,1024)

        mask=F.sigmoid(torch.bmm(image_tmp,speech_final).view(-1,config.image_size[0],config.image_size[1])).unsqueeze(1) #[t,1,13,13]
        mask=self.mask_conv(mask).squeeze().unsqueeze(-1) #[t,13,13,1]
        print('Gets mask final: ',mask.shape)

        images_masked=image_final*mask #[t,13,13,1024)
        print('Gets masked images: ',images_masked.shape)
        images_masked=torch.transpose(torch.transpose(images_masked,1,3),2,3) #[t,1024, 13,13]
        images_masked_aver=self.pool_over_size(images_masked).squeeze() #[t,1024]
        print('Gets masked images aver: ',images_masked_aver.shape)

        feats_final=torch.mean(images_masked_aver,dim=0,keepdim=True) #[1,1024]
        print('Gets final feats: ',feats_final.shape)



class basic_model(nn.Module):
    def __init__(self, config,tgt_spk_size, use_cuda ):
        super(basic_model, self).__init__()

        self.speech_model=SPEECH_MODEL_1(config,)
        # self.images_model=IMAGES_MODEL(config,)
        self.output_model=MERGE_MODEL(config, tgt_spk_size)
        self.use_cuda = use_cuda
        self.tgt_spk_size = tgt_spk_size
        self.config = config
        self.loss_for_ss= nn.MSELoss()
        self.log_softmax = nn.LogSoftmax()

        self.linear=nn.Linear(1024,tgt_spk_size)

    def forward(self,input_image,input_speech,):
        # Image:[t,1024,3,13],Speech:[4t,257,2]

        speech_hidden=self.speech_model(input_speech)
        feats_final=self.output_model(input_image,speech_hidden)
        predict_scores=self.linear(feats_final)

        return predict_scores

