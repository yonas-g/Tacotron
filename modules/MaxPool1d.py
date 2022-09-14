import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPool1d(nn.Module):
    '''
    inputs: [N, T, C]
    outputs: [N, T // stride, C]
    '''
    def __init__(self, kernel_size, stride=1, padding='same'):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
    
    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)  # [N, C, T]
        if self.padding == 'same':
            left = (self.kernel_size - 1) // 2
            right = (self.kernel_size - 1) - left
            pad = (left, right)
        else:
            pad = (0, 0)

        inputs = F.pad(inputs, pad)
        outputs = self.max_pool(inputs)
        outputs = outputs.transpose(1, 2)  # [N, T, C]

        return outputs