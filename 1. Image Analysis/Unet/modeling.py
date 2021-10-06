# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:29:13 2021

@author: USER
"""
import torch
import torch.nn as nn
import components as c
import parameters

class UnetGenerator(nn.Module):
    def __init__(self):
        
        super(UnetGenerator, self).__init__()
        params = parameters.params()
                
        self.in_dim = params['in_dim']
        self.out_dim = params['out_dim']
        self.n_filter = params['n_filter']
        
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        self.down1 = c.conv_block2(self.in_dim, self.n_filter, act_fn)
        self.pool1 = c.maxpool()
        self.down2 = c.conv_block2(self.n_filter*1, self.n_filter*2, act_fn)
        self.pool2 = c.maxpool()
        self.down3 = c.conv_block2(self.n_filter*2, self.n_filter*4, act_fn)
        self.pool3 = c.maxpool()
        self.down4 = c.conv_block2(self.n_filter*4, self.n_filter*8, act_fn)
        self.pool4 = c.maxpool()
        
        self.bridge = c.conv_block2(self.n_filter*8, self.n_filter*16, act_fn)
        
        self.trans1 = c.conv_block2(self.n_filter*16, self.n_filter*8, act_fn)
        self.up1 = c.conv_block2(self.n_filter*16, self.n_filter*8, act_fn)
        self.trans2 = c.conv_block2(self.n_filter*8, self.n_filter*4, act_fn)
        self.up2 = c.conv_block2(self.n_filter*8, self.n_filter*4, act_fn)
        self.trans3 = c.conv_block2(self.n_filter*4, self.n_filter*2, act_fn)
        self.up3 = c.conv_block2(self.n_filter*4, self.n_filter*2, act_fn)
        self.trans4 = c.conv_block2(self.n_filter*2, self.n_filter*1, act_fn)
        self.up4 = c.conv_block2(self.n_filter*2, self.n_filter*1, act_fn)
        
        self.out = nn.Sequential(
            nn.Conv2d(self.n_filter, self.out_dim, 3, 1, 1)
            )

    def forward(self, input):
        down1 = self.down1(input)
        pool1 = self.poo11(down1)
        down2 = self.down2(pool1)
        pool2 = self.poo12(down2)
        down3 = self.down2(pool2)
        pool3 = self.poo12(down3)
        down4 = self.down2(pool3)
        pool4 = self.poo12(down4)
        
        bridge = self.bridge(pool4)
        
        trans1 = self.trans1(bridge)
        concat1 = torch.cat([trans1, down4], dim=1)
        up1 = self.up1(concat1)
        trans2 = self.trans1(up1)
        concat2 = torch.cat([trans2, down3], dim=1)
        up2 = self.up1(concat2)
        trans3 = self.trans1(up2)
        concat3 = torch.cat([trans3, down2], dim=1)
        up3 = self.up1(concat3)
        trans4 = self.trans1(up3)
        concat4 = torch.cat([trans4, down1], dim=1)
        up4 = self.up1(concat4)
        
        logits = self.out(up4)
        
        return logits