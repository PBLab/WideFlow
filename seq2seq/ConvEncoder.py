import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, hid_dim, n_layers, kernel_size, embedding_kernel_size, p_drop=0.2, device='cpu'):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.embedding_kernel_size = embedding_kernel_size
        self.p_drop = p_drop
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.embedding_conv = nn.Conv2d(in_channels=1,
                                        out_channels=hid_dim,
                                        kernel_size=embedding_kernel_size,
                                        padding=(embedding_kernel_size - 1) // 2)

        self.convs = nn.ModuleList([nn.Conv3d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2)
                                    for _ in range(n_layers)])

        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, src):
        # src = [batch_size, #channels, H, W, T] #channels = 1

        embedded = self.embedding_conv(src)
        conv_input = embedded

        # begin convolutional blocks...
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)

            # apply residual connection
            conved = (conved + conv_input) * self.scale

            # set conv_input to conved for next loop iteration
            conv_input = conved

        # permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))

        # elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale

        return conved, combined
