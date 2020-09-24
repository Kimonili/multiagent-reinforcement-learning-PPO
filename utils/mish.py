from torch import nn

# import activation functions
import functional as Func


class Mish(nn.Module):
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return Func.mish(input)
