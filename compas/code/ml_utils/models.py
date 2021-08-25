import numpy as np
import torch
import torch.nn.functional as F

from enum import Enum


class ModelType(Enum):
    NN = 0
    SVM = 1
    DT = 2
    RF = 3


class Custom_Net(torch.nn.Module):

    def __init__(self, inputwidth_list, n_layers, n_hidden, enc, outwidth, bias=False, scalewidth='singular'):
        super(Custom_Net, self).__init__()

        if enc > n_layers - 1:
            raise ValueError('Not enough hidden layers to include encoding-decoding')

        self.n_hidden = n_hidden
        self.nout = outwidth
        # make layers for different input scales
        self.input_layers = []
        self.input_widths = inputwidth_list
        for width in inputwidth_list:
            if scalewidth != 'singular':
                owidth = width
            else:
                owidth = 1

            self.input_layers.append(torch.nn.Linear(width, owidth, bias=bias))

        # make hidden layers
        if scalewidth != 'singular':
            iwidth = int(np.sum(inputwidth_list))
        else:
            iwidth = len(inputwidth_list)

        self.layers = [torch.nn.Linear(iwidth, self.n_hidden, bias=bias)]
        for i in range(0, n_layers):
            inp = self.n_hidden
            op = self.n_hidden
            # add encoder & decoder
            if enc:
                if i == enc:
                    op = int(np.floor(self.n_hidden / 2))
                elif i == enc + 1:
                    inp = int(np.floor(self.n_hidden / 2))

            self.layers += [torch.nn.Linear(inp, op, bias=bias)]
        self.out = torch.nn.Linear(op, self.nout)

    def forward(self, x):
        # make input slices
        inputcalc = []
        beg = 0
        dim = len(x.shape) - 1
        for i in range(0, len(self.input_widths)):
            slic = torch.narrow(x, dim, beg, self.input_widths[i])
            beg = int(np.sum(self.input_widths[0:i + 1]))
            tl = self.input_layers[i]
            inputcalc.append(F.relu(tl(slic)))

        x = torch.cat(tuple(inputcalc), dim)
        for i in range(0, len(self.layers)):
            x = F.relu(self.layers[i](x))
        x = self.out(x)
        return x