# https://analyticsindiamag.com/how-to-implement-convolutional-autoencoder-in-pytorch-with-cuda/

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from seq2seq.vid_embedding.vid_autoencoder import VidAutoEncoder, Enc, Dec
from seq2seq.vid_embedding.autoencoder_dataset import *
import os
import pathlib


#Converting data to torch.FloatTensor
batch_size = 1
n_frames = 30
transform = transforms.Compose([Rescale((256, 256)),
                               ToTensor()])

train_data_dirs = [str(pathlib.Path('C:/') / 'Users' / 'motar' / 'PycharmProjects' / 'WideFlow' / 'data' / 'widefield_fast_acq'), ]
train_data = AutoEncoderDataSet(train_data_dirs, n_frames, transform)

test_data_dirs = [str(pathlib.Path('C:/') / 'Users' / 'motar' / 'PycharmProjects' / 'WideFlow' / 'data' / 'widefield_fast_acq'), ]
test_data = AutoEncoderDataSet(test_data_dirs, n_frames, transform)

#Prepare data loaders
train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size)

#Instantiate the model
enc = Enc()
dec = Dec()
model = VidAutoEncoder(enc, dec)
print(model)

#Loss function
criterion = nn.BCELoss()

#Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = 'cpu'
model.to(device)

# Epochs
n_epochs = 10

for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

    # Training
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


# show results
z = train_data.__getitem__(0)
z = z.unsqueeze(0)
output = model(z)
output = output.view(batch_size, 1, 30, 256, 256)
output = output.detach().numpy()

plt.imshow(z[0, 0, 15, :, :])
plt.show()
plt.figure()
plt.imshow(output[0, 0, 15, :, :])
plt.show()

z_enc = model.encoder(z)
zs = z_enc.shape
z_enc = z_enc.view(batch_size, 1, zs[2], zs[3], zs[4])
z_enc = z_enc.detach().numpy()
plt.figure()
plt.imshow(z[0, 0, 15, :, :])
plt.show()
