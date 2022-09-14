import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.Conv1d import Conv1d
from modules.BatchNorm1d import BatchNorm1d

class Conv1dBank(nn.Module):
    '''
        inputs: [N, T, C_in]
        outputs: [N, T, C_out * K]  # same padding
    Args:
        in_channels: C_in: E//2
        out_channels: C_out: E//2
    '''
    def __init__(self, K, in_channels, out_channels):
        super().__init__()

        banks = [Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k+1
        ) for k in range(K)]

        self.banks = nn.ModuleList(banks)
        self.bn = BatchNorm1d(out_channels * K)

    def forward(self, inputs):
        
        outputs = self.bank[0](inputs)

        for k in range(1, len(self.bank)):
            output = self.bank[k](inputs)
            outputs = torch.cat([outputs, output], dim=2)

        outputs = self.bn(outputs)  # [N, T, C_out * K]
        outputs = F.relu(outputs)

        return outputs