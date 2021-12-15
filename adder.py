'''
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of BSD 3-Clause License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3-Clause License for more details.
'''
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import math
import adder_cuda
# try:
    
#     import unoptimized_cuda
# except:
#     print("Unable to import CUDA unoptimized kernels")

def adder2d_function(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
    X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)
    W_col = W.view(n_filters, -1)
    
    out = adder.apply(W_col,X_col)
    
    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()
    
    return out

class adder(Function):
    @staticmethod
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col,X_col) 
        Co = W_col.size(0)
        HoWoN = X_col.size(1)

        output = torch.zeros((Co, HoWoN),device="cuda:0")
        adder_cuda.ADDER_CONV(X_col, W_col, output)

        ###############test code################
        # ground_truth = -(W_col.unsqueeze(2)-X_col.unsqueeze(0)).abs().sum(1)
        # sub = output - ground_truth
        # print("check result:", torch.sum(sub), torch.var(sub), torch.max(sub), torch.min(sub))
        return output

    @staticmethod
    def backward(ctx,grad_output):
        W_col,X_col = ctx.saved_tensors

        grad_W_col = torch.zeros_like(W_col)
        grad_X_col = torch.zeros_like(X_col)
        adder_cuda.ADDER_BACKWARD(grad_output, X_col, W_col, grad_W_col, grad_X_col)

        ###############test code################
        # gt_w = ((X_col.unsqueeze(0)-W_col.unsqueeze(2))*grad_output.unsqueeze(1)).sum(2)
        # sub = grad_W_col - gt_w
        # print("check grad_W_col result:", torch.sum(sub), torch.var(sub), torch.max(sub), torch.min(sub))

        # gt_x = (-(X_col.unsqueeze(0)-W_col.unsqueeze(2)).clamp(-1,1)*grad_output.unsqueeze(1)).sum(0)
        # sub = grad_X_col - gt_x
        # print("check grad_X_col result:", torch.sum(sub), torch.var(sub), torch.max(sub), torch.min(sub))
        ###############test end#################
        
        grad_W_col = grad_W_col/grad_W_col.norm(p=2).clamp(min=1e-12)*math.sqrt(W_col.size(1)*W_col.size(0))/5
        
        return grad_W_col, grad_X_col


class adder2d(nn.Module):

    def __init__(self,input_channel,output_channel,kernel_size, stride=1, padding=0, bias = False, use_cuda = False):
        super(adder2d, self).__init__()
        self.use_cuda = use_cuda
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.adder = torch.nn.Parameter(nn.init.normal_(torch.randn(output_channel,input_channel,kernel_size,kernel_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = adder2d_function(x,self.adder, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        return output
    
    
