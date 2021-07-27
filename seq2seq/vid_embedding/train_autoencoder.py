# https://analyticsindiamag.com/how-to-implement-convolutional-autoencoder-in-pytorch-with-cuda/

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
from seq2seq.vid_embedding.vid_autoencoder import VidAutoEncoder, ConvNetGenerator
from seq2seq.vid_embedding.autoencoder_dataset import *
from seq2seq.vid_embedding.files_list import files_list
import os


model_config_path = os.path.abspath(os.path.join(os.path.pardir, 'vid_embedding', 'models_configs', 'v1', 'autoencoder_configs.json'))
model_path = os.path.abspath(os.path.join(os.path.pardir, 'vid_embedding', 'embedding_results', 'models', 'model_test', 'model.pt'))
checkpoints_path = os.path.abspath(os.path.join(os.path.pardir, 'models', 'model_test', 'checkpoints'))

# Converting data to torch.FloatTensor
batch_size = 1
n_frames = 30
transform = transforms.Compose([Rescale((256, 256)),
                                GrayScale(),
                                ToTensor()])

# Prepare data loaders
train_data = AutoEncoderDataSet(files_list, n_frames, transform)
test_data = AutoEncoderDataSet(files_list, n_frames, transform)

train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size)

# Optimizer

# Instantiate the model
if os.path.exists(model_path):
    model = torch.load(model_path)
    model.eval()  # TODO: is this necessary
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    init_epoch = 1
    if os.path.exists(checkpoints_path):
        checkpoints_list = os.listdir(os.path.abspath(checkpoints_path))
        checkpoints_list = [int(name.split('_')[1].split('.')[0]) for name in checkpoints_list if isinstance(name, str)]
        last_checkpoint = 'epoch_' + str(max(checkpoints_list)) + '.pt'
        checkpoint = torch.load(os.path.join(checkpoints_path, last_checkpoint))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
else:

    model = VidAutoEncoder.get_class_from_config(model_config_path)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # model_config_path = os.path.abspath(os.path.join(os.path.pardir, 'vid_embedding', 'models_configs', 'v1', 'encoder_configs.json'))
    # model = ConvNetGenerator.get_class_from_config(model_config_path)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    init_epoch = 1

print(model)

# Loss function
criterion = nn.BCELoss()

# prepare for training
device = 'cpu'
model.to(device)

# Epochs
n_epochs = 200 + init_epoch
saving_epoch = 5

for epoch in range(init_epoch, n_epochs + 1):
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
    if not epoch % saving_epoch:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, os.path.join(checkpoints_path, f'epoch_{epoch}.pt'))

# save model
torch.save(model, model_path)

# show results
z = train_data.__getitem__(0)
z = z.unsqueeze(0)
output = model(z)
output = output.view(batch_size, 1, 30, 256, 256)
output = output.detach().numpy()

plt.imshow(z[0, 0, 15, :, :], vmin=0.9, vmax=0.99)
plt.show()
plt.figure()
plt.imshow(output[0, 0, 15, :, :])
plt.show()

z_enc = model.encoder(z)
zs = z_enc.shape
z_enc = z_enc.view(batch_size, 1, zs[2], zs[3], zs[4])
z_enc = z_enc.detach().numpy()
plt.figure()
plt.imshow(z[0, 0, 15, :, :], vmin=0.9, vmax=0.99)
plt.show()
