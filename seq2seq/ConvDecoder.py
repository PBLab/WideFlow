import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self,  output_dim, emb_dim, hid_dim, n_layers, kernel_size, embedding_kernel_size, p_drop=0.2, device='cpu'):
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
                                              kernel_size=kernel_size)
                                    for _ in range(n_layers)])

        self.dropout = nn.Dropout(p=p_drop)

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        # embedded = [batch size, trg len, emb dim]
        # conved = [batch size, hid dim, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]

        # permute and convert back to emb dim
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))

        # conved_emb = [batch size, trg len, emb dim]

        combined = (conved_emb + embedded) * self.scale

        # combined = [batch size, trg len, emb dim]

        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))

        # energy = [batch size, trg len, src len]

        attention = F.softmax(energy, dim=2)

        # attention = [batch size, trg len, src len]

        attended_encoding = torch.matmul(attention, encoder_combined)

        # attended_encoding = [batch size, trg len, emd dim]

        # convert from emb dim -> hid dim
        attended_encoding = self.attn_emb2hid(attended_encoding)

        # attended_encoding = [batch size, trg len, hid dim]

        # apply residual connection
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale

        # attended_combined = [batch size, hid dim, trg len]

        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):
        # src = [batch_size, #channels, H, W, T] #channels = 1
        padding_shape = trg.shape
        padding_shape[-1] = self.kernel_size // 2 + 1

        embedded = self.embedding_conv(trg)
        conv_input = embedded

        padding = torch.zeros(padding_shape).fill_(self.trg_pad_idx).to(self.device)
        # begin convolutional blocks...
        for i, conv in enumerate(self.convs):
            conv_input = self.dropout(conv_input)

            # need to pad so decoder can't "cheat"
            padded_conv_input = torch.cat((padding, conv_input), dim=2)
            # padded_conv_input = [batch_size, hid_dim, H, W, T + kernel_size/2 + 1]

            # pass through convolutional layer
            conved = conv(padded_conv_input)
            conved = F.glu(conved, dim=1)

            # calculate attention
            attention, conved = self.calculate_attention(embedded, conved, encoder_conved, encoder_combined)

            # apply residual connection
            conved = (conved + conv_input) * self.scale

            # set conv_input to conved for next loop iteration
            conv_input = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))

        output = self.fc_out(self.dropout(conved))

        return output, attention