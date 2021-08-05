import torch
import torch.nn as nn
import torch.nn.functional as F


# class VidAutoEncoder(nn.Module):
#     def __init__(self):
#         super(VidAutoEncoder, self).__init__()
#
#         # Encoder
#         self.e_conv1 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=[3, 3, 3], stride=[1, 1, 1])
#         self.e_conv2 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=[3, 1, 1], stride=[1, 1, 1])
#         self.e_pool1 = nn.MaxPool3d(kernel_size=[3, 3, 3])
#         self.e_pool2 = nn.MaxPool3d(kernel_size=[1, 1, 3])
#
#         # Decoder
#         self.t_conv1 = nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=[3, 3, 3], stride=[1, 1, 1])
#         self.t_conv2 = nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=[3, 1, 1], stride=[1, 1, 1])
#
#
#     def forward(self, x):
#         # encode
#         x = F.relu(self.e_conv1(x))
#         x = F.relu(self.e_conv1(x))
#         # x = self.e_pool1(x)
#
#         x = F.relu(self.e_conv2(x))
#         x = F.relu(self.e_conv2(x))
#         # x = self.e_pool2(x)
#
#         # decode
#         # x = F.relu(self.t_conv2(x))
#         # x = F.relu(self.t_conv2(x))
#         x = F.relu(self.t_conv2(x))
#         x = F.relu(self.t_conv2(x))
#         x = F.relu(self.t_conv1(x))
#         x = F.sigmoid(self.t_conv1(x))
#
#         return x

class VidAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(VidAutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Enc(nn.Module):
    def __init__(self):
        super(Enc, self).__init__()
        self.e_conv1 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=[3, 3, 3], stride=[1, 1, 1])
        self.e_conv2 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=[3, 1, 1], stride=[1, 1, 1])
        self.e_pool1 = nn.MaxPool3d(kernel_size=[3, 3, 3])
        self.e_pool2 = nn.MaxPool3d(kernel_size=[1, 1, 3])

    def forward(self, x):
        x = F.relu(self.e_conv1(x))
        x = F.relu(self.e_conv1(x))
        x = F.relu(self.e_conv2(x))
        x = F.relu(self.e_conv2(x))
        return x


class Dec(nn.Module):
    def __init__(self):
        super(Dec, self).__init__()
        self.t_conv1 = nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=[3, 3, 3], stride=[1, 1, 1])
        self.t_conv2 = nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=[3, 1, 1], stride=[1, 1, 1])

    def forward(self, x):
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv1(x))
        return x