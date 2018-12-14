#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class basic_model(nn.Module):
    def __init__(self, config, tgt_spk_size, use_cuda ):
        super(basic_model, self).__init__()

        # self.speech_model=SPEECH_MODEL(config,)
        # self.images_model=IMAGES_MODEL(config,)
        # self.output_model=MERGE_MODEL(config,)
        self.use_cuda = use_cuda
        self.tgt_spk_size = tgt_spk_size
        self.config = config
        self.loss_for_ss= nn.MSELoss()
        self.log_softmax = nn.LogSoftmax()

    def forward(self, *input):
        return None

