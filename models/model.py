#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SPEECH_MODEL_v0(nn.Module):
    def __init__(self, config ):
        super(SPEECH_MODEL_v0, self).__init__()
        self.cnn1=nn.Conv2d(2,96,(1,7),stride=[1],padding=(0,3),dilation=(1,1))
        self.cnn2=nn.Conv2d(96,96,(7,1),stride=1,padding=(3,0),dilation=(1,1))
        self.cnn3=nn.Conv2d(96,96,(5,5),stride=1,padding=(2,2),dilation=(1,1))
        self.cnn4=nn.Conv2d(96,96,(5,5),stride=1,padding=(4,2),dilation=(2,1))
        self.cnn5=nn.Conv2d(96,96,(5,5),stride=1,padding=(8,2),dilation=(4,1))
        self.pool5=nn.MaxPool2d((2,2))

        self.cnn6=nn.Conv2d(96,96,(5,5),stride=1,padding=(16,2),dilation=(8,1))
        self.cnn7=nn.Conv2d(96,96,(5,5),stride=1,padding=(32,2),dilation=(16,1))
        self.cnn8=nn.Conv2d(96,96,(5,5),stride=1,padding=(64,2),dilation=(32,1))
        self.cnn9=nn.Conv2d(96,96,(5,5),stride=1,padding=(2,2),dilation=(1,1))
        self.cnn10=nn.Conv2d(96,96,(5,5),stride=1,padding=(4,4),dilation=(2,2))
        self.pool10=nn.MaxPool2d((2,1))

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
        # print(x.shape)
        # print('\nSpeech layer log:')
        x = x.contiguous()
        for idx in range(self.num_cnns):
            cnn_layer=eval('self.cnn{}'.format(idx+1))
            bn_layer=eval('self.bn{}'.format(idx+1))
            x=F.relu(cnn_layer(x))
            x=bn_layer(x)
            if idx in [4,9]:
                pool_layer=eval('self.pool{}'.format(idx+1))
                x=pool_layer(x)
            # print('speech shape after CNNs:',idx,'', x.size()) #最终应该是1,8,t,128
        x=x.transpose(2,3).contiguous().view(1,1024,-1) #　应该是　[1,1024,t]
        return x
    
class IMAGE_MODEL_2(nn.Module):
    def __init__(self,config):
        super(IMAGE_MODEL_2, self).__init__()
        self.cnn1=nn.Conv2d(3,64,(3,3),stride=1,padding=(1,1),dilation=(1,1))
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
        self.pool6=nn.MaxPool2d((2,2))

        self.cnn7=nn.Conv2d(256,512,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn7=nn.BatchNorm2d(512)
        self.cnn8=nn.Conv2d(512,512,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn8=nn.BatchNorm2d(512)
        self.pool8=nn.MaxPool2d((2,2))

        self.cnn9=nn.Conv2d(512,config.hidden_embs,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn9=nn.BatchNorm2d(config.hidden_embs)
        self.cnn10=nn.Conv2d(config.hidden_embs,config.hidden_embs,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn10=nn.BatchNorm2d(config.hidden_embs)
        # self.pool10=nn.MaxPool2d((2,2))

        self.num_cnns=10
        self.pool_list=[2,4,6,8]

    def forward(self, x):
        # IMAGE:[4t,3,416,416]
        print(x.shape)
        # print('\nImage layer log:')
        x = x.contiguous()
        for idx in range(self.num_cnns):
            cnn_layer=eval('self.cnn{}'.format(idx+1))
            bn_layer=eval('self.bn{}'.format(idx+1))
            x=F.relu(cnn_layer(x))
            x=bn_layer(x)
            if idx+1 in self.pool_list:
                pool_layer=eval('self.pool{}'.format(idx+1))
                x=pool_layer(x)
            # print('Image shape after CNNs:',idx,'', x.size())

        print('Image shape final:', x.size(),'\n')
        return x

class IMAGE_MODEL(nn.Module):
    def __init__(self,config):
        super(IMAGE_MODEL, self).__init__()
        self.cnn1=nn.Conv2d(3,64,(3,3),stride=1,padding=(1,1),dilation=(1,1))
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
        self.pool6=nn.MaxPool2d((2,2))

        self.cnn7=nn.Conv2d(256,512,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn7=nn.BatchNorm2d(512)
        self.cnn8=nn.Conv2d(512,512,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn8=nn.BatchNorm2d(512)
        self.pool8=nn.MaxPool2d((2,2))

        self.cnn9=nn.Conv2d(512,1024,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn9=nn.BatchNorm2d(1024)
        self.cnn10=nn.Conv2d(1024,1024,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn10=nn.BatchNorm2d(1024)
        # self.pool10=nn.MaxPool2d((2,2))

        self.num_cnns=10
        self.pool_list=[2,4,6,8]

    def forward(self, x):
        # IMAGE:[4t,3,416,416]
        print(x.shape)
        print('\nImage layer log:')
        x = x.contiguous()
        for idx in range(self.num_cnns):
            cnn_layer=eval('self.cnn{}'.format(idx+1))
            bn_layer=eval('self.bn{}'.format(idx+1))
            x=F.relu(cnn_layer(x))
            x=bn_layer(x)
            if idx+1 in self.pool_list:
                pool_layer=eval('self.pool{}'.format(idx+1))
                x=pool_layer(x)
            print('Image shape after CNNs:',idx,'', x.size())

        print('Image shape final:', x.size(),'\n')
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
        # print(x.shape) #[bs=1, 2, 152, 257]
        # print('\nSpeech layer log:')
        x = x.contiguous()
        for idx in range(self.num_cnns):
            cnn_layer=eval('self.cnn{}'.format(idx+1))
            bn_layer=eval('self.bn{}'.format(idx+1))
            x=F.relu(cnn_layer(x))
            x=bn_layer(x)
            if idx+1 in self.pool_list:
                pool_layer=eval('self.pool{}'.format(idx+1))
                x=pool_layer(x)
            # print('speech shape after CNNs:',idx,'', x.size())

        # print('speech shape final:', x.size(),'\n') # (t,8,1024)
        return x

class SPEECH_MODEL_2(nn.Module):
    def __init__(self, config ):
        super(SPEECH_MODEL_2, self).__init__()
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

        self.cnn9=nn.Conv2d(512,config.hidden_embs,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn9=nn.BatchNorm2d(config.hidden_embs)
        self.cnn10=nn.Conv2d(config.hidden_embs,config.hidden_embs,(3,3),stride=1,padding=(1,1),dilation=(1,1))
        self.bn10=nn.BatchNorm2d(config.hidden_embs)
        self.pool10=nn.MaxPool2d((1,2))

        self.num_cnns=10
        self.pool_list=[2,4,6,8,10]

    def forward(self, x):
        # Speech:[4t,257,2]
        x=torch.transpose(torch.transpose(x,0,2),1,2).unsqueeze(0)
        # print(x.shape) #[bs=1, 2, 152, 257]
        # print('\nSpeech layer log:')
        x = x.contiguous()
        for idx in range(self.num_cnns):
            cnn_layer=eval('self.cnn{}'.format(idx+1))
            bn_layer=eval('self.bn{}'.format(idx+1))
            x=F.relu(cnn_layer(x))
            x=bn_layer(x)
            if idx+1 in self.pool_list:
                pool_layer=eval('self.pool{}'.format(idx+1))
                x=pool_layer(x)
            # print('speech shape after CNNs:',idx,'', x.size())

        # print('speech shape final:', x.size(),'\n')
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
        # print('Gets speech final: ',speech_final.shape)
        speech_final=speech_final.squeeze() #[t,1024]
        speech_final=speech_final.unsqueeze(1).unsqueeze(1) #[t,1,1,1024]
        speech_final=speech_final.expand(-1,config.image_size[0],config.image_size[1],-1) #[t,1,1,1024]
        speech_final=speech_final.contiguous().view(-1,1024,1) #[t,1,1,1024]
        # print('Gets speech final: ',speech_final.shape)

        image_hidden=self.im_conv1(image_hidden)
        image_final=self.im_conv2(image_hidden) #[t,1024,13,13]
        # print('Gets image final: ',image_final.shape)
        image_final=torch.transpose(torch.transpose(image_final,1,3),1,2) #[t,13,13,1024]
        # print('Gets image final: ',image_final.shape)
        image_tmp=image_final.contiguous().view(-1,1,1024)

        mask=F.sigmoid(torch.bmm(image_tmp,speech_final).view(-1,config.image_size[0],config.image_size[1])).unsqueeze(1) #[t,1,13,13]
        mask=self.mask_conv(mask).squeeze().unsqueeze(-1) #[t,13,13,1]
        # print('Gets mask final: ',mask.shape)

        images_masked=image_final*mask #[t,13,13,1024)
        # print('Gets masked images: ',images_masked.shape)
        images_masked=torch.transpose(torch.transpose(images_masked,1,3),2,3) #[t,1024, 13,13]
        images_masked_aver=self.pool_over_size(images_masked).squeeze() #[t,1024]
        # print('Gets masked images aver: ',images_masked_aver.shape)

        feats_final=torch.mean(images_masked_aver,dim=0,keepdim=True) #[1,1024]
        # print('Gets final feats: ',feats_final.shape)

        return feats_final,mask

class MERGE_MODEL_v0(nn.Module):
    def __init__(self, config,tgt_spk_size):
        super(MERGE_MODEL_v0, self).__init__()
        self.tgt_spk_size = tgt_spk_size
        self.config = config

        self.sp_pool=nn.MaxPool2d((1,8))
        self.sp_fc1=nn.Linear(1024,1024)
        self.sp_fc2=nn.Linear(1024,1024)

        self.im_conv1=nn.Conv2d(1024,1024,(1,1),stride=1,padding=(0,0),dilation=(1,1))
        self.im_conv2=nn.Conv2d(1024,1024,(1,1),stride=1,padding=(0,0),dilation=(1,1))

        self.mask_conv=nn.Conv2d(1,1,(1,1),stride=1,padding=(0,0),dilation=(1,1),bias=False)
        # self.pool_over_size=nn.AvgPool2d((config.image_size[0],config.image_size[1]))
        self.pool_over_size=nn.MaxPool2d((config.image_size[0],config.image_size[1]))

        if config.image_time_conv:
            self.image_time_conv=nn.Conv1d(1024,1024,5,stride=1,padding=2,dilation=1,groups=1024)
        # self.pool_over_size=nn.AvgPool2d((config.image_size[0],config.image_size[1]))
        self.pool_over_size=nn.MaxPool2d((config.image_size[0],config.image_size[1]))

    def forward(self, image_hidden,speech_hidden):
        config=self.config
        #Image:[t,1024,13,13],Speech:[1,1024,t,8]
        # speech_hidden=self.sp_pool(speech_hidden).squeeze(-1) #[1,1024,t,1]-->[1,1024,t]
        speech_hidden=torch.transpose(speech_hidden,1,2) # [1,t,1024]
        speech_hidden=F.relu(self.sp_fc1(speech_hidden)) # [1,t,1024]
        speech_final=F.relu(self.sp_fc2(speech_hidden)) # [1,t,1024]
        # print('Gets speech final: ',speech_final.shape)
        speech_final=speech_final.squeeze() #[t,1024]
        speech_final=speech_final.unsqueeze(1).unsqueeze(1) #[t,1,1,1024]
        speech_final=speech_final.expand(-1,config.image_size[0],config.image_size[1],-1) #[t,1,1,1024]
        speech_final=speech_final.contiguous().view(-1,1024,1) #[t,1,1,1024]
        # print('Gets speech final: ',speech_final.shape)

        if config.image_time_conv: # 是否采用时间维度的conv
            image_hidden_tmp=image_hidden.view(-1,1024,config.image_size[0]*config.image_size[1]).transpose(0,2)  #[13*13,1024,t]
            image_hidden_tmp=self.image_time_conv(image_hidden_tmp)#[13*13,1024,t]
            image_hidden_tmp=image_hidden_tmp.transpose(0,2).view(-1,1024,config.image_size[0],config.image_size[1])
            image_hidden_tmp=F.relu(self.im_conv1(image_hidden_tmp))
        # elif config.images_recu:
        #     image_hidden_tmp=torch.zeros_like(image_hidden)
        # print('image original:')
        # print(image_hidden[0,10])
        # print(image_hidden[1,10])
        # for idx,frame in enumerate(image_hidden[:-1]):
        #     image_hidden_tmp[idx]=image_hidden[idx+1]-frame
        # print('image after:')
        # print(image_hidden_tmp[0,10])
        # print(image_hidden_tmp[1,10])
        else:
            image_hidden_tmp=F.relu(self.im_conv1(image_hidden))
        image_final=F.relu(self.im_conv2(image_hidden_tmp)) #[t,1024,13,13]
        if config.images_recu:
            image_final_tmp = torch.zeros_like(image_final)
            for idx, frame in enumerate(image_final[:-1]):
                image_final_tmp[idx] = image_final[idx + 1] - frame
            # print('Gets image final: ',image_final.shape)
            image_final_tmp = torch.transpose(torch.transpose(image_final_tmp, 1, 3), 1, 2)  # [t,13,13,1024]
            # print('Gets image final: ',image_final.shape)
            image_tmp = image_final_tmp.contiguous().view(-1, 1, 1024)
            image_final = torch.transpose(torch.transpose(image_final, 1, 3), 1, 2)  # [t,13,13,1024]
        else:
            # print('Gets image final: ',image_final.shape)
            image_final = torch.transpose(torch.transpose(image_final, 1, 3), 1, 2)  # [t,13,13,1024]
            # print('Gets image final: ',image_final.shape)
            image_tmp = image_final.contiguous().view(-1, 1, 1024)

        mask=torch.bmm(image_tmp,speech_final).view(-1,config.image_size[0],config.image_size[1]).unsqueeze(1) #[t,1,13,13]
        if config.mask_norm:
            # mask=F.normalize(mask,dim=0)

            mask=F.normalize(mask.view(-1,config.image_size[0]*config.image_size[1])).view(-1,1,config.image_size[0],config.image_size[1])
            # for idx,each_frame in enumerate(mask):
            #     mask[idx]=mask[idx]/torch.max(each_frame)
        if config.threshold:
            mask = F.threshold(mask, config.threshold, 0)
        if config.mask_topk:
            mask_plan = mask.squeeze().view(-1,config.image_size[0]*config.image_size[1]) #t,13*13
            kth_value = torch.topk(mask_plan,config.mask_topk,dim=1)[0][:,-1].unsqueeze(-1) #t,1
            kth_value_map = kth_value.expand(-1,config.image_size[0]*config.image_size[1]) # t,169
            mask = ((mask_plan>=kth_value_map).float()*mask_plan).view(-1,config.image_size[0],config.image_size[1])

        if config.mask_softmax:
            mask=F.relu(self.mask_conv(mask).squeeze()) #[t,13,13]
            mask=mask.view(-1,config.image_size[0]*config.image_size[1]) #[t,13*13]
            mask=F.softmax(mask,dim=1) #[t,13*13]
            mask=mask.view(-1,config.image_size[0],config.image_size[1],1)#[t,13,13,1]
        else:
            # mask=F.sigmoid(self.mask_conv(mask).squeeze().unsqueeze(-1)) #[t,13,13,1]
            # 用Tanh的好处在于，之前算出来的score是0的，就他妈是0,别让sigmoid再注册个0.5的值进去捣乱了。
            # mask=F.tanh(F.relu(self.mask_conv(mask).squeeze().unsqueeze(-1))) #[t,13,13,1]
            mask=mask.squeeze().unsqueeze(-1) #[t,13,13,1]
        # print('Gets mask final: ',mask.shape)

        if not config.mask_over_init:
            images_masked=image_final*mask #[t,13,13,1024)
        else:
            image_hidden=torch.transpose(torch.transpose(image_hidden,1,3),1,2) #[t,13,13,1024]
            images_masked=image_hidden*mask #[t,13,13,1024)

        # print('Gets masked images: ',images_masked.shape)
        images_masked=torch.transpose(torch.transpose(images_masked,1,3),2,3) #[t,1024, 13,13]
        if not config.size_sum:
            images_masked_aver=self.pool_over_size(images_masked).squeeze() #[t,1024]
        else:
            images_masked_aver=images_masked.view(-1,1024,config.image_size[0]*config.image_size[1]).sum(2)#[t,1024]
        # print('Gets masked images aver: ',images_masked_aver.shape)

        if not config.class_frame:
            feats_final=torch.mean(images_masked_aver,dim=0,keepdim=True) #[1,1024]
        else:
            feats_final=images_masked_aver
        # print('Gets final feats: ',feats_final.shape)

        return feats_final,mask

class image_only(nn.Module):
    def __init__(self, config,tgt_spk_size):
        super(image_only, self).__init__()
        self.tgt_spk_size = tgt_spk_size
        self.config = config

        self.sp_pool=nn.MaxPool2d((1,8))
        self.sp_fc1=nn.Linear(1024,1024)
        self.sp_fc2=nn.Linear(1024,1024)

        self.im_conv1=nn.Conv2d(1024,1024,(1,1),stride=1,padding=(0,0),dilation=(1,1))
        self.im_conv2=nn.Conv2d(1024,1024,(1,1),stride=1,padding=(0,0),dilation=(1,1))

        self.mask_conv=nn.Conv2d(1,1,(1,1),stride=1,padding=(0,0),dilation=(1,1),bias=False)
        # self.pool_over_size=nn.AvgPool2d((config.image_size[0],config.image_size[1]))
        self.pool_over_size=nn.MaxPool2d((config.image_size[0],config.image_size[1]))

        if config.image_time_conv:
            self.image_time_conv=nn.Conv1d(1024,1024,5,stride=1,padding=2,dilation=1,groups=1024)

        if config.image_time_rnn:
            self.image_time_rnn=nn.LSTM(1024,512,2,batch_first=True,bidirectional=True)
        # self.pool_over_size=nn.AvgPool2d((config.image_size[0],config.image_size[1]))
        self.pool_over_size=nn.MaxPool2d((config.image_size[0],config.image_size[1]))

    def forward(self, image_hidden):
        config=self.config
        if config.image_time_conv: # 是否采用时间维度的conv
            image_hidden_tmp=image_hidden.view(-1,1024,config.image_size[0]*config.image_size[1]).transpose(0,2)  #[13*13,1024,t]
            image_hidden_tmp=self.image_time_conv(image_hidden_tmp)#[13*13,1024,t]
            image_hidden_tmp=image_hidden_tmp.transpose(0,2).view(-1,1024,config.image_size[0],config.image_size[1])
            image_hidden_tmp=F.relu(self.im_conv1(image_hidden_tmp))

        elif config.image_time_rnn: # 是否采用时间维度的conv
            image_hidden_tmp=image_hidden.view(-1,1024,config.image_size[0]*config.image_size[1]).transpose(0,2).transpose(1,2)  #[13*13,t,1024]
            image_hidden_tmp=self.image_time_rnn(image_hidden_tmp)[0]#[13*13,t,1024]
            image_hidden_tmp=image_hidden_tmp.transpose(0,2).transpose(0,1).contiguous().view(-1,1024,config.image_size[0],config.image_size[1])
            image_hidden_tmp=F.relu(self.im_conv1(image_hidden_tmp))
        # elif config.images_recu:
        #     image_hidden_tmp=torch.zeros_like(image_hidden)
        # print('image original:')
        # print(image_hidden[0,10])
        # print(image_hidden[1,10])
        # for idx,frame in enumerate(image_hidden[:-1]):
        #     image_hidden_tmp[idx]=image_hidden[idx+1]-frame
        # print('image after:')
        # print(image_hidden_tmp[0,10])
        # print(image_hidden_tmp[1,10])
        else:
            image_hidden_tmp=F.relu(self.im_conv1(image_hidden))
        image_final=F.relu(self.im_conv2(image_hidden_tmp)) #[t,1024,13,13]
        image_final = torch.transpose(torch.transpose(image_final, 1, 3), 1, 2)  # [t,13,13,1024]
        images_masked=image_final

        # print('Gets masked images: ',images_masked.shape)
        images_masked=torch.transpose(torch.transpose(images_masked,1,3),2,3) #[t,1024, 13,13]
        if not config.size_sum:
            images_masked_aver=self.pool_over_size(images_masked).squeeze() #[t,1024]
        else:
            images_masked_aver=images_masked.view(-1,1024,config.image_size[0]*config.image_size[1]).sum(2)#[t,1024]
        # print('Gets masked images aver: ',images_masked_aver.shape)

        if not config.class_frame:
            feats_final=torch.mean(images_masked_aver,dim=0,keepdim=True) #[1,1024]
        else:
            feats_final=images_masked_aver
        # print('Gets final feats: ',feats_final.shape)

        return feats_final, None

class audio_only(nn.Module):
    def __init__(self, config,tgt_spk_size):
        super(audio_only, self).__init__()
        self.tgt_spk_size = tgt_spk_size
        self.config = config

        self.sp_pool=nn.MaxPool2d((1,8))
        self.sp_fc1=nn.Linear(1024,1024)
        self.sp_fc2=nn.Linear(1024,1024)

        self.im_conv1=nn.Conv2d(1024,1024,(1,1),stride=1,padding=(0,0),dilation=(1,1))
        self.im_conv2=nn.Conv2d(1024,1024,(1,1),stride=1,padding=(0,0),dilation=(1,1))

        self.mask_conv=nn.Conv2d(1,1,(1,1),stride=1,padding=(0,0),dilation=(1,1),bias=False)
        # self.pool_over_size=nn.AvgPool2d((config.image_size[0],config.image_size[1]))
        self.pool_over_size=nn.MaxPool2d((config.image_size[0],config.image_size[1]))

        if config.image_time_conv:
            self.image_time_conv=nn.Conv1d(1024,1024,5,stride=1,padding=2,dilation=1,groups=1024)

        if config.image_time_rnn:
            self.image_time_rnn=nn.LSTM(1024,512,2,batch_first=True,bidirectional=True)
        # self.pool_over_size=nn.AvgPool2d((config.image_size[0],config.image_size[1]))
        self.pool_over_size=nn.MaxPool2d((config.image_size[0],config.image_size[1]))

    def forward(self,speech_hidden):
        config=self.config
        #Image:[t,1024,13,13],Speech:[1,1024,t,8]
        speech_hidden=self.sp_pool(speech_hidden).squeeze(-1) #[1,1024,t,1]-->[1,1024,t]
        speech_hidden=torch.transpose(speech_hidden,1,2) # [1,t,1024]
        speech_hidden=F.relu(self.sp_fc1(speech_hidden)) # [1,t,1024]
        speech_final=F.relu(self.sp_fc2(speech_hidden)) # [1,t,1024]
        # print('Gets speech final: ',speech_final.shape)
        speech_final=speech_final.squeeze() #[t,1024]
        # print('Gets speech final: ',speech_final.shape)

        if not config.class_frame:
            feats_final=torch.mean(speech_final,dim=0,keepdim=True) #[1,1024]
        else:
            feats_final=speech_final
        # print('Gets final feats: ',feats_final.shape)

        return feats_final,None

class MERGE_MODEL_1(nn.Module):
    def __init__(self, config,tgt_spk_size):
        super(MERGE_MODEL_1, self).__init__()
        self.tgt_spk_size = tgt_spk_size
        self.config = config

        self.sp_pool=nn.MaxPool2d((1,8))
        self.sp_fc1=nn.Linear(1024,1024)
        self.sp_fc2=nn.Linear(1024,1024)

        self.im_conv1=nn.Conv2d(1024,1024,(1,1),stride=1,padding=(0,0),dilation=(1,1))
        self.im_conv2=nn.Conv2d(1024,1024,(1,1),stride=1,padding=(0,0),dilation=(1,1))

        self.mask_conv=nn.Conv2d(1,1,(1,1),stride=1,padding=(0,0),dilation=(1,1),bias=False)
        # self.pool_over_size=nn.AvgPool2d((config.image_size[0],config.image_size[1]))
        self.pool_over_size=nn.MaxPool2d((config.image_size[0],config.image_size[1]))

        if config.image_time_conv:
            self.image_time_conv=nn.Conv1d(1024,1024,5,stride=1,padding=2,dilation=1,groups=1024)

        if config.image_time_rnn:
            self.image_time_rnn=nn.LSTM(1024,512,2,batch_first=True,bidirectional=True)
        # self.pool_over_size=nn.AvgPool2d((config.image_size[0],config.image_size[1]))
        self.pool_over_size=nn.MaxPool2d((config.image_size[0],config.image_size[1]))

    def forward(self, image_hidden,speech_hidden):
        config=self.config
        #Image:[t,1024,13,13],Speech:[1,1024,t,8]
        speech_hidden=self.sp_pool(speech_hidden).squeeze(-1) #[1,1024,t,1]-->[1,1024,t]
        speech_hidden=torch.transpose(speech_hidden,1,2) # [1,t,1024]
        speech_hidden=F.relu(self.sp_fc1(speech_hidden)) # [1,t,1024]
        speech_final=F.relu(self.sp_fc2(speech_hidden)) # [1,t,1024]
        # print('Gets speech final: ',speech_final.shape)
        speech_final=speech_final.squeeze() #[t,1024]
        speech_final=speech_final.unsqueeze(1).unsqueeze(1) #[t,1,1,1024]
        speech_final=speech_final.expand(-1,config.image_size[0],config.image_size[1],-1) #[t,1,1,1024]
        speech_final=speech_final.contiguous().view(-1,1024,1) #[t,1,1,1024]
        # print('Gets speech final: ',speech_final.shape)

        if config.image_time_conv: # 是否采用时间维度的conv
            image_hidden_tmp=image_hidden.view(-1,1024,config.image_size[0]*config.image_size[1]).transpose(0,2)  #[13*13,1024,t]
            image_hidden_tmp=self.image_time_conv(image_hidden_tmp)#[13*13,1024,t]
            image_hidden_tmp=image_hidden_tmp.transpose(0,2).view(-1,1024,config.image_size[0],config.image_size[1])
            image_hidden_tmp=F.relu(self.im_conv1(image_hidden_tmp))

        elif config.image_time_rnn: # 是否采用时间维度的conv
            image_hidden_tmp=image_hidden.view(-1,1024,config.image_size[0]*config.image_size[1]).transpose(0,2).transpose(1,2)  #[13*13,t,1024]
            image_hidden_tmp=self.image_time_rnn(image_hidden_tmp)[0]#[13*13,t,1024]
            image_hidden_tmp=image_hidden_tmp.transpose(0,2).transpose(0,1).contiguous().view(-1,1024,config.image_size[0],config.image_size[1])
            image_hidden_tmp=F.relu(self.im_conv1(image_hidden_tmp))
        # elif config.images_recu:
        #     image_hidden_tmp=torch.zeros_like(image_hidden)
            # print('image original:')
            # print(image_hidden[0,10])
            # print(image_hidden[1,10])
            # for idx,frame in enumerate(image_hidden[:-1]):
            #     image_hidden_tmp[idx]=image_hidden[idx+1]-frame
            # print('image after:')
            # print(image_hidden_tmp[0,10])
            # print(image_hidden_tmp[1,10])
        else:
            image_hidden_tmp=F.relu(self.im_conv1(image_hidden))
        image_final=F.relu(self.im_conv2(image_hidden_tmp)) #[t,1024,13,13]
        if config.images_recu:
            image_final_tmp = torch.zeros_like(image_final)
            for idx, frame in enumerate(image_final[:-1]):
                image_final_tmp[idx] = image_final[idx + 1] - frame
            # print('Gets image final: ',image_final.shape)
            image_final_tmp = torch.transpose(torch.transpose(image_final_tmp, 1, 3), 1, 2)  # [t,13,13,1024]
            # print('Gets image final: ',image_final.shape)
            image_tmp = image_final_tmp.contiguous().view(-1, 1, 1024)
            image_final = torch.transpose(torch.transpose(image_final, 1, 3), 1, 2)  # [t,13,13,1024]
        else:
            # print('Gets image final: ',image_final.shape)
            image_final = torch.transpose(torch.transpose(image_final, 1, 3), 1, 2)  # [t,13,13,1024]
            # print('Gets image final: ',image_final.shape)
            image_tmp = image_final.contiguous().view(-1, 1, 1024)

        mask=torch.bmm(image_tmp,speech_final).view(-1,config.image_size[0],config.image_size[1]).unsqueeze(1) #[t,1,13,13]
        if config.mask_norm:
            # mask=F.normalize(mask,dim=0)

            mask=F.normalize(mask.view(-1,config.image_size[0]*config.image_size[1])).view(-1,1,config.image_size[0],config.image_size[1])
            # for idx,each_frame in enumerate(mask):
            #     mask[idx]=mask[idx]/torch.max(each_frame)
        if config.threshold:
            mask = F.threshold(mask, config.threshold, 0)
        if config.mask_topk:
            mask_plan = mask.squeeze().view(-1,config.image_size[0]*config.image_size[1]) #t,13*13
            kth_value = torch.topk(mask_plan,config.mask_topk,dim=1)[0][:,-1].unsqueeze(-1) #t,1
            kth_value_map = kth_value.expand(-1,config.image_size[0]*config.image_size[1]) # t,169
            mask = ((mask_plan>=kth_value_map).float()*mask_plan).view(-1,config.image_size[0],config.image_size[1])

        if config.mask_softmax:
            mask=F.relu(self.mask_conv(mask).squeeze()) #[t,13,13]
            mask=mask.view(-1,config.image_size[0]*config.image_size[1]) #[t,13*13]
            mask=F.softmax(mask,dim=1) #[t,13*13]
            mask=mask.view(-1,config.image_size[0],config.image_size[1],1)#[t,13,13,1]
        else:
            # mask=F.sigmoid(self.mask_conv(mask).squeeze().unsqueeze(-1)) #[t,13,13,1]
            # 用Tanh的好处在于，之前算出来的score是0的，就他妈是0,别让sigmoid再注册个0.5的值进去捣乱了。
            # mask=F.tanh(F.relu(self.mask_conv(mask).squeeze().unsqueeze(-1))) #[t,13,13,1]
            mask=mask.squeeze().unsqueeze(-1) #[t,13,13,1]
        # print('Gets mask final: ',mask.shape)

        if not config.mask_over_init:
            images_masked=image_final*mask #[t,13,13,1024)
        else:
            image_hidden=torch.transpose(torch.transpose(image_hidden,1,3),1,2) #[t,13,13,1024]
            images_masked=image_hidden*mask #[t,13,13,1024)

        # print('Gets masked images: ',images_masked.shape)
        images_masked=torch.transpose(torch.transpose(images_masked,1,3),2,3) #[t,1024, 13,13]
        if not config.size_sum:
            images_masked_aver=self.pool_over_size(images_masked).squeeze() #[t,1024]
        else:
            images_masked_aver=images_masked.view(-1,1024,config.image_size[0]*config.image_size[1]).sum(2)#[t,1024]
        # print('Gets masked images aver: ',images_masked_aver.shape)

        if not config.class_frame:
            feats_final=torch.mean(images_masked_aver,dim=0,keepdim=True) #[1,1024]
        else:
            feats_final=images_masked_aver
        # print('Gets final feats: ',feats_final.shape)

        return feats_final,mask

class MERGE_MODEL_both(nn.Module): # modified from Merge_1
    def __init__(self, config,tgt_spk_size):
        super(MERGE_MODEL_both, self).__init__()
        self.tgt_spk_size = tgt_spk_size
        self.config = config

        self.sp_pool=nn.MaxPool2d((1,8))
        self.sp_fc1=nn.Linear(1024,1024)
        self.sp_fc2=nn.Linear(1024,1024)

        self.im_conv1=nn.Conv2d(1024,1024,(1,1),stride=1,padding=(0,0),dilation=(1,1))
        self.im_conv2=nn.Conv2d(1024,1024,(1,1),stride=1,padding=(0,0),dilation=(1,1))

        self.mask_conv=nn.Conv2d(1,1,(1,1),stride=1,padding=(0,0),dilation=(1,1),bias=False)
        # self.pool_over_size=nn.AvgPool2d((config.image_size[0],config.image_size[1]))
        self.pool_over_size=nn.MaxPool2d((config.image_size[0],config.image_size[1]))

        if config.image_time_conv:
            self.image_time_conv=nn.Conv1d(1024,1024,5,stride=1,padding=2,dilation=1,groups=1024)

        if config.image_time_rnn:
            self.image_time_rnn=nn.LSTM(1024,512,2,batch_first=True,bidirectional=True)
        # self.pool_over_size=nn.AvgPool2d((config.image_size[0],config.image_size[1]))
        self.pool_over_size=nn.MaxPool2d((config.image_size[0],config.image_size[1]))

    def forward(self, image_hidden,speech_hidden):
        config=self.config
        #Image:[t,1024,13,13],Speech:[1,1024,t,8]
        speech_hidden=self.sp_pool(speech_hidden).squeeze(-1) #[1,1024,t,1]-->[1,1024,t]
        speech_hidden=torch.transpose(speech_hidden,1,2) # [1,t,1024]
        speech_hidden=F.relu(self.sp_fc1(speech_hidden)) # [1,t,1024]
        speech_final=F.relu(self.sp_fc2(speech_hidden)) # [1,t,1024]
        # print('Gets speech final: ',speech_final.shape)
        speech_final=speech_final.squeeze() #[t,1024]
        # print('Gets speech final: ',speech_final.shape)

        if config.image_time_conv: # 是否采用时间维度的conv
            image_hidden_tmp=image_hidden.view(-1,1024,config.image_size[0]*config.image_size[1]).transpose(0,2)  #[13*13,1024,t]
            image_hidden_tmp=self.image_time_conv(image_hidden_tmp)#[13*13,1024,t]
            image_hidden_tmp=image_hidden_tmp.transpose(0,2).view(-1,1024,config.image_size[0],config.image_size[1])
            image_hidden_tmp=F.relu(self.im_conv1(image_hidden_tmp))

        elif config.image_time_rnn: # 是否采用时间维度的conv
            image_hidden_tmp=image_hidden.view(-1,1024,config.image_size[0]*config.image_size[1]).transpose(0,2).transpose(1,2)  #[13*13,t,1024]
            image_hidden_tmp=self.image_time_rnn(image_hidden_tmp)[0]#[13*13,t,1024]
            image_hidden_tmp=image_hidden_tmp.transpose(0,2).transpose(0,1).contiguous().view(-1,1024,config.image_size[0],config.image_size[1])
            image_hidden_tmp=F.relu(self.im_conv1(image_hidden_tmp))
        else:
            image_hidden_tmp=F.relu(self.im_conv1(image_hidden))
        image_final=F.relu(self.im_conv2(image_hidden_tmp)) #[t,1024,13,13]
        # print('Gets image final: ',image_final.shape)
        image_final = torch.transpose(torch.transpose(image_final, 1, 3), 1, 2)  # [t,13,13,1024]
        # print('Gets image final: ',image_final.shape)
        '''
        mask=torch.bmm(image_tmp,speech_final).view(-1,config.image_size[0],config.image_size[1]).unsqueeze(1) #[t,1,13,13]
        if config.mask_norm:
            # mask=F.normalize(mask,dim=0)

            mask=F.normalize(mask.view(-1,config.image_size[0]*config.image_size[1])).view(-1,1,config.image_size[0],config.image_size[1])
            # for idx,each_frame in enumerate(mask):
            #     mask[idx]=mask[idx]/torch.max(each_frame)
        if config.threshold:
            mask = F.threshold(mask, config.threshold, 0)
        if config.mask_topk:
            mask_plan = mask.squeeze().view(-1,config.image_size[0]*config.image_size[1]) #t,13*13
            kth_value = torch.topk(mask_plan,config.mask_topk,dim=1)[0][:,-1].unsqueeze(-1) #t,1
            kth_value_map = kth_value.expand(-1,config.image_size[0]*config.image_size[1]) # t,169
            mask = ((mask_plan>=kth_value_map).float()*mask_plan).view(-1,config.image_size[0],config.image_size[1])

        if config.mask_softmax:
            mask=F.relu(self.mask_conv(mask).squeeze()) #[t,13,13]
            mask=mask.view(-1,config.image_size[0]*config.image_size[1]) #[t,13*13]
            mask=F.softmax(mask,dim=1) #[t,13*13]
            mask=mask.view(-1,config.image_size[0],config.image_size[1],1)#[t,13,13,1]
        else:
            # mask=F.sigmoid(self.mask_conv(mask).squeeze().unsqueeze(-1)) #[t,13,13,1]
            # 用Tanh的好处在于，之前算出来的score是0的，就他妈是0,别让sigmoid再注册个0.5的值进去捣乱了。
            # mask=F.tanh(F.relu(self.mask_conv(mask).squeeze().unsqueeze(-1))) #[t,13,13,1]
            mask=mask.squeeze().unsqueeze(-1) #[t,13,13,1]
        # print('Gets mask final: ',mask.shape)

        if not config.mask_over_init:
            images_masked=image_final*mask #[t,13,13,1024)
        else:
            image_hidden=torch.transpose(torch.transpose(image_hidden,1,3),1,2) #[t,13,13,1024]
            images_masked=image_hidden*mask #[t,13,13,1024)
        '''

        images_masked=image_final # use the un-masked images feats directly here.
        # print('Gets masked images: ',images_masked.shape)
        images_masked=torch.transpose(torch.transpose(images_masked,1,3),2,3) #[t,1024, 13,13]
        if not config.size_sum:
            images_masked_aver=self.pool_over_size(images_masked).squeeze() #[t,1024]
        else:
            images_masked_aver=images_masked.view(-1,1024,config.image_size[0]*config.image_size[1]).sum(2)#[t,1024]
        # print('Gets masked images aver: ',images_masked_aver.shape)

        feats_final=torch.cat((images_masked_aver,speech_final.squeeze()),1)
        if not config.class_frame:
            feats_final=torch.mean(feats_final,dim=0,keepdim=True) #[1,1024]
        else:
            feats_final=feats_final
        # print('Gets final feats: ',feats_final.shape)

        return feats_final, None

class MERGE_MODEL_2(nn.Module):
    def __init__(self, config,tgt_spk_size):
        super(MERGE_MODEL_2, self).__init__()
        self.tgt_spk_size = tgt_spk_size
        self.config = config

        self.sp_pool=nn.MaxPool2d((1,8))
        self.sp_fc1=nn.Linear(config.hidden_embs,config.hidden_embs)
        self.sp_fc2=nn.Linear(config.hidden_embs,config.hidden_embs)

        self.im_conv1=nn.Conv2d(config.hidden_embs,config.hidden_embs,(1,1),stride=1,padding=(0,0),dilation=(1,1))
        self.im_conv2=nn.Conv2d(config.hidden_embs,config.hidden_embs,(1,1),stride=1,padding=(0,0),dilation=(1,1))

        self.mask_conv=nn.Conv2d(1,1,(1,1),stride=1,padding=(0,0),dilation=(1,1),bias=False)
        # self.pool_over_size=nn.AvgPool2d((config.image_size[0],config.image_size[1]))
        self.pool_over_size=nn.MaxPool2d((config.image_size[0],config.image_size[1]))

        if config.image_time_conv:
            self.image_time_conv=nn.Conv1d(config.hidden_embs,config.hidden_embs,5,stride=1,padding=2,dilation=1,groups=config.hidden_embs)
        # self.pool_over_size=nn.AvgPool2d((config.image_size[0],config.image_size[1]))
        self.pool_over_size=nn.MaxPool2d((config.image_size[0],config.image_size[1]))

    def forward(self, image_hidden,speech_hidden):
        config=self.config
        #Image:[t,config.hidden_embs,13,13],Speech:[1,config.hidden_embs,t,8]
        speech_hidden=self.sp_pool(speech_hidden).squeeze(-1) #[1,config.hidden_embs,t,1]-->[1,config.hidden_embs,t]
        speech_hidden=torch.transpose(speech_hidden,1,2) # [1,t,config.hidden_embs]
        speech_hidden=F.relu(self.sp_fc1(speech_hidden)) # [1,t,config.hidden_embs]
        speech_final=F.relu(self.sp_fc2(speech_hidden)) # [1,t,config.hidden_embs]
        # print('Gets speech final: ',speech_final.shape)
        speech_final=speech_final.squeeze() #[t,config.hidden_embs]
        speech_final=speech_final.unsqueeze(1).unsqueeze(1) #[t,1,1,config.hidden_embs]
        speech_final=speech_final.expand(-1,config.image_size[0],config.image_size[1],-1) #[t,1,1,config.hidden_embs]
        speech_final=speech_final.contiguous().view(-1,config.hidden_embs,1) #[t,1,1,config.hidden_embs]
        # print('Gets speech final: ',speech_final.shape)

        if config.image_time_conv: # 是否采用时间维度的conv
            image_hidden_tmp=image_hidden.view(-1,config.hidden_embs,config.image_size[0]*config.image_size[1]).transpose(0,2)  #[13*13,config.hidden_embs,t]
            image_hidden_tmp=self.image_time_conv(image_hidden_tmp)#[13*13,config.hidden_embs,t]
            image_hidden_tmp=image_hidden_tmp.transpose(0,2).view(-1,config.hidden_embs,config.image_size[0],config.image_size[1])
            image_hidden_tmp=F.relu(self.im_conv1(image_hidden_tmp))
        # elif config.images_recu:
        #     image_hidden_tmp=torch.zeros_like(image_hidden)
        # print('image original:')
        # print(image_hidden[0,10])
        # print(image_hidden[1,10])
        # for idx,frame in enumerate(image_hidden[:-1]):
        #     image_hidden_tmp[idx]=image_hidden[idx+1]-frame
        # print('image after:')
        # print(image_hidden_tmp[0,10])
        # print(image_hidden_tmp[1,10])
        else:
            image_hidden_tmp=F.relu(self.im_conv1(image_hidden))
        image_final=F.relu(self.im_conv2(image_hidden_tmp)) #[t,config.hidden_embs,13,13]
        if config.images_recu:
            image_final_tmp = torch.zeros_like(image_final)
            for idx, frame in enumerate(image_final[:-1]):
                image_final_tmp[idx] = image_final[idx + 1] - frame
            # print('Gets image final: ',image_final.shape)
            image_final_tmp = torch.transpose(torch.transpose(image_final_tmp, 1, 3), 1, 2)  # [t,13,13,config.hidden_embs]
            # print('Gets image final: ',image_final.shape)
            image_tmp = image_final_tmp.contiguous().view(-1, 1, config.hidden_embs)
            image_final = torch.transpose(torch.transpose(image_final, 1, 3), 1, 2)  # [t,13,13,config.hidden_embs]
        else:
            # print('Gets image final: ',image_final.shape)
            image_final = torch.transpose(torch.transpose(image_final, 1, 3), 1, 2)  # [t,13,13,config.hidden_embs]
            # print('Gets image final: ',image_final.shape)
            image_tmp = image_final.contiguous().view(-1, 1, config.hidden_embs)

        mask=torch.bmm(image_tmp,speech_final).view(-1,config.image_size[0],config.image_size[1]).unsqueeze(1) #[t,1,13,13]
        if config.mask_norm:
            # mask=F.normalize(mask,dim=0)

            mask=F.normalize(mask.view(-1,config.image_size[0]*config.image_size[1])).view(-1,1,config.image_size[0],config.image_size[1])
            # for idx,each_frame in enumerate(mask):
            #     mask[idx]=mask[idx]/torch.max(each_frame)
        if config.threshold:
            mask = F.threshold(mask, config.threshold, 0)
        if config.mask_topk:
            mask_plan = mask.squeeze().view(-1,config.image_size[0]*config.image_size[1]) #t,13*13
            kth_value = torch.topk(mask_plan,config.mask_topk,dim=1)[0][:,-1].unsqueeze(-1) #t,1
            kth_value_map = kth_value.expand(-1,config.image_size[0]*config.image_size[1]) # t,169
            mask = ((mask_plan>=kth_value_map).float()*mask_plan).view(-1,config.image_size[0],config.image_size[1])

        if config.mask_softmax:
            mask=F.relu(self.mask_conv(mask).squeeze()) #[t,13,13]
            mask=mask.view(-1,config.image_size[0]*config.image_size[1]) #[t,13*13]
            mask=F.softmax(mask,dim=1) #[t,13*13]
            mask=mask.view(-1,config.image_size[0],config.image_size[1],1)#[t,13,13,1]
        else:
            # mask=F.sigmoid(self.mask_conv(mask).squeeze().unsqueeze(-1)) #[t,13,13,1]
            # 用Tanh的好处在于，之前算出来的score是0的，就他妈是0,别让sigmoid再注册个0.5的值进去捣乱了。
            # mask=F.tanh(F.relu(self.mask_conv(mask).squeeze().unsqueeze(-1))) #[t,13,13,1]
            mask=mask.squeeze().unsqueeze(-1) #[t,13,13,1]
        # print('Gets mask final: ',mask.shape)

        if not config.mask_over_init:
            images_masked=image_final*mask #[t,13,13,config.hidden_embs)
        else:
            image_hidden=torch.transpose(torch.transpose(image_hidden,1,3),1,2) #[t,13,13,config.hidden_embs]
            images_masked=image_hidden*mask #[t,13,13,config.hidden_embs)

        # print('Gets masked images: ',images_masked.shape)
        images_masked=torch.transpose(torch.transpose(images_masked,1,3),2,3) #[t,config.hidden_embs, 13,13]
        if not config.size_sum:
            images_masked_aver=self.pool_over_size(images_masked).squeeze() #[t,config.hidden_embs]
        else:
            images_masked_aver=images_masked.view(-1,config.hidden_embs,config.image_size[0]*config.image_size[1]).sum(2)#[t,config.hidden_embs]
        # print('Gets masked images aver: ',images_masked_aver.shape)

        if not config.class_frame:
            feats_final=torch.mean(images_masked_aver,dim=0,keepdim=True) #[1,config.hidden_embs]
        else:
            feats_final=images_masked_aver
        # print('Gets final feats: ',feats_final.shape)

        return feats_final,mask

class basic_model(nn.Module):
    def __init__(self, config, use_cuda,tgt_spk_size):
        super(basic_model, self).__init__()

        self.speech_model=SPEECH_MODEL_1(config,)
        # self.images_model=IMAGES_MODEL(config,)
        # self.output_model=MERGE_MODEL(config, tgt_spk_size)
        self.output_model=MERGE_MODEL_1(config, tgt_spk_size)

        # self.speech_model=SPEECH_MODEL_v0(config,)
        # self.output_model=MERGE_MODEL_v0(config, tgt_spk_size)
        self.use_cuda = use_cuda
        self.tgt_spk_size = tgt_spk_size
        self.config = config
        self.loss_for_ss= nn.MSELoss()
        self.log_softmax = nn.LogSoftmax()

        self.linear=nn.Linear(1024,tgt_spk_size)

    def forward(self,input_image,input_speech,):
        # Image:[t,1024,3,13],Speech:[4t,257,2]

        speech_hidden=self.speech_model(input_speech)
        feats_final,masks=self.output_model(input_image,speech_hidden)
        predict_scores=self.linear(feats_final)

        return predict_scores,masks

class basic_image_model(nn.Module):
    def __init__(self, config, use_cuda,tgt_spk_size):
        super(basic_image_model, self).__init__()

        self.output_model=image_only(config, tgt_spk_size)

        self.use_cuda = use_cuda
        self.tgt_spk_size = tgt_spk_size
        self.config = config
        self.loss_for_ss= nn.MSELoss()
        self.log_softmax = nn.LogSoftmax()

        self.linear=nn.Linear(1024,tgt_spk_size)

    def forward(self,input_image,input_speech,):
        # Image:[t,1024,3,13],Speech:[4t,257,2]

        feats_final,masks=self.output_model(input_image)
        predict_scores=self.linear(feats_final)
        return predict_scores,masks

class basic_audio_model(nn.Module):
    def __init__(self, config, use_cuda,tgt_spk_size):
        super(basic_audio_model, self).__init__()

        self.speech_model=SPEECH_MODEL_1(config,)
        self.output_model=audio_only(config, tgt_spk_size)

        self.use_cuda = use_cuda
        self.tgt_spk_size = tgt_spk_size
        self.config = config
        self.loss_for_ss= nn.MSELoss()
        self.log_softmax = nn.LogSoftmax()

        self.linear=nn.Linear(1024,tgt_spk_size)

    def forward(self,input_image,input_speech,):
        # Image:[t,1024,3,13],Speech:[4t,257,2]
        speech_hidden=self.speech_model(input_speech)
        feats_final,masks=self.output_model(speech_hidden)
        predict_scores=self.linear(feats_final)
        return predict_scores,masks

class basic_both_model(nn.Module):
    def __init__(self, config, use_cuda,tgt_spk_size):
        super(basic_both_model, self).__init__()

        self.speech_model=SPEECH_MODEL_1(config,)
        self.output_model=MERGE_MODEL_both(config, tgt_spk_size)

        self.use_cuda = use_cuda
        self.tgt_spk_size = tgt_spk_size
        self.config = config
        self.loss_for_ss= nn.MSELoss()
        self.log_softmax = nn.LogSoftmax()

        self.linear=nn.Linear(2*1024,tgt_spk_size) # cont merge function

    def forward(self,input_image,input_speech,):
        # Image:[t,1024,3,13],Speech:[4t,257,2]

        speech_hidden=self.speech_model(input_speech) # [t,1024]
        feats_final,masks=self.output_model(input_image,speech_hidden)
        predict_scores=self.linear(feats_final)

        return predict_scores,masks

class raw_model(nn.Module):
    def __init__(self, config, use_cuda,tgt_spk_size):
        super(raw_model, self).__init__()

        self.speech_model=SPEECH_MODEL_2(config,)
        self.images_model=IMAGE_MODEL_2(config,)
        # self.output_model=MERGE_MODEL(config, tgt_spk_size)
        self.output_model=MERGE_MODEL_2(config, tgt_spk_size)
        self.use_cuda = use_cuda
        self.tgt_spk_size = tgt_spk_size
        self.config = config
        self.loss_for_ss= nn.MSELoss()
        self.log_softmax = nn.LogSoftmax()

        # self.linear=nn.Linear(1024,tgt_spk_size)
        self.linear=nn.Linear(config.hidden_embs,tgt_spk_size)

    def forward(self,input_image,input_speech,):
        # Image:[t,3,416,416],Speech:[4t,257,2]

        image_hidden=self.images_model(input_image)
        speech_hidden=self.speech_model(input_speech)
        feats_final,masks=self.output_model(image_hidden,speech_hidden)
        predict_scores=self.linear(feats_final)

        return predict_scores,masks
