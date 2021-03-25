# source at: https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb
import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        # verify encoder, decoder shapes are equivalent

    def forward(self, src, trg):

        # TODO: change to self.encoder.forward(src), and self.decoder.forward(trg, encoder_conved, encoder_combined)??????
        encoder_conved, encoder_combined = self.encoder(src)
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)

        return output, attention