import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        '''
        inputs: [N, T, C_in]
        outputs: [N, T, C_out]
        '''
        super().__init__()
        if padding == 'same':
            left = (kernel_size - 1) // 2
            right = (kernel_size - 1) - left
            self.pad = (left, right)
            # pad = kernel_size // 2
        else:
            self.pad = (0, 0)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)

    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1, 2)  # [N, C_in, T]
        inputs = F.pad(inputs, self.pad)
        out = self.conv1d(inputs)  # [N, C_out, T]
        out = torch.transpose(out, 1, 2)  # [N, T, C_out]
        return out

if __name__ == "__main__":
    model = Conv1d(256, 128, 3)
    inputs = torch.autograd.Variable(torch.randn(10, 22, 256))
    out = model(inputs)
    print(out.shape)