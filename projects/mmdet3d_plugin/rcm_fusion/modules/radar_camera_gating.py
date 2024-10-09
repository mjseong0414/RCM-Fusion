import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.runner import force_fp32, auto_fp16

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import numpy as np
import math
import logging

class RadarCameraGating(nn.Module):
    def __init__(self,
                 in_channels=256):
        super(RadarCameraGating, self).__init__()
        self.in_channels = in_channels

        self.cam_atten_weight = nn.Sequential(
            nn.Conv1d(in_channels,in_channels,kernel_size=7,padding=3),
        )
        self.rad_atten_weight = nn.Sequential(
            nn.Conv1d(in_channels,in_channels,kernel_size=7,padding=3),
        )
        
    @auto_fp16(apply_to=('query_c', 'query_r'))
    def forward(self, query_c, query_r):
        query_rc = (query_c + query_r).permute(0,2,1).contiguous()
        cam_weight = self.cam_atten_weight(query_rc).sigmoid().permute(0,2,1).contiguous()
        rad_weight = self.rad_atten_weight(query_rc).sigmoid().permute(0,2,1).contiguous()
        query_rc = query_c * cam_weight + query_r * rad_weight
        
        return query_rc
