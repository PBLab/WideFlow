import torch
import torch.nn as nn
import torch.nn.functional as F
import json


class VidAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, name="GeneralAutoEncoder"):
        super(VidAutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.name = name


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @classmethod
    def get_class_from_config(cls, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

        name = config['name']
        encoder_path = config['encoder_config_path']
        decoder_path = config['decoder_config_path']
        encoder = ConvNetGenerator.get_class_from_config(encoder_path)
        decoder = ConvNetGenerator.get_class_from_config(decoder_path)

        return cls(encoder, decoder, name)


class ConvNetGenerator(nn.Module):
    def __init__(self, model_list, inception_list=None, name='GeneralConvNet'):
        super(ConvNetGenerator, self).__init__()
        self.model_list = model_list
        self.inception_list = inception_list
        self.name = name
        self.network_module_list = self.create_network_module_list()
        z=3

    def forward(self, x):
        for module in self.network_module_list:
            if module["type"] == "inception":
                x = self.inception_forward(module["layer"], x)
            elif module["type"] == "layer":
                x = module["layer"](x)

            if module["skip_connection"] == 'add_skip':  # add residual block
                x.add(x_skip)

            elif module["skip_connection"] == 'skip':  # save tensor for residual connection
                x_skip = x

        return x

    @staticmethod
    def inception_forward(module_list, x):
        """

        Args:
            module_list: list of nn.ModuleList
            x: input Tensor

        Returns: output Tensor

        """
        n_routes = len(module_list)
        routes_outputs = [None] * n_routes
        for i, route in enumerate(module_list):
            y = x
            for layer in route:
                y = layer(y)
            routes_outputs[i] = y
        inception_out = torch.squeeze(torch.cat(routes_outputs, dim=2))
        # TODO: Add inception output batchnormaliztions option?

        return inception_out

    def create_network_module_list(self):
        """

        Returns: a list in which each element is a list defining a module in the network
                    [module type: 'inception' or 'layer',
                    if 'inception': torch ModuleList, if 'layer': torch layer,
                    skip connection: 'skip' or 'add_skip' or None
                    batch normaliztion: True or False

        """
        network_module_list = []
        for module in self.model_list:
            if module["layer"] == 'inception':
                network_module_list.append({
                    "type": "inception",
                    "layer": self.create_inception_module_list(),
                    "skip_connection": module["skip_connection"]}
                )
            elif type(module["layer"]) is dict:
                network_module_list.append({
                    "type": "layer",
                    "layer": self.create_layer(module["layer"]),
                    "skip_connection": module["skip_connection"]}
                )
            else:
                self.raise_model_construction_error(module["layer"], "layer")

        return network_module_list

    def create_inception_module_list(self):
        module_list = []
        for route_list in self.inception_list["routes"]:
            route_module = nn.ModuleList()
            for layer in route_list:
                route_module.append(self.create_layer(layer)[:])
            module_list.append(route_module)

        return module_list

    def create_layer(self, layer):

        layer_module = nn.ModuleList()
        if layer['type'] == 'convolution':
            if layer['dim'] == '2d':
                layer_module.append(nn.Conv2d(in_channels=layer['in_channels'], out_channels=layer['out_channels'],
                                             kernel_size=layer['kernel_size'], stride=layer['stride'],
                                             padding=layer['padding'], dilation=layer['dilation']))
            elif layer['dim'] == '3d':
                layer_module.append(nn.Conv3d(in_channels=layer['in_channels'], out_channels=layer['out_channels'],
                                             kernel_size=layer['kernel_size'], stride=layer['stride'],
                                             padding=layer['padding'], dilation=layer['dilation']))
            else:
                self.raise_model_construction_error(layer['dim'], 'dim')

        elif layer['type'] == 'convolution transpose':
            if layer['dim'] == '2d':
                layer_module.append(nn.ConvTranspose2d(in_channels=layer['in_channels'], out_channels=layer['out_channels'],
                                          kernel_size=layer['kernel_size'], stride=layer['stride'],
                                          padding=layer['padding'], dilation=layer['dilation']))
            elif layer['dim'] == '3d':
                layer_module.append(nn.ConvTranspose3d(in_channels=layer['in_channels'], out_channels=layer['out_channels'],
                                          kernel_size=layer['kernel_size'], stride=layer['stride'],
                                          padding=layer['padding'], dilation=layer['dilation']))
            else:
                self.raise_model_construction_error(layer['dim'], 'dim')

        elif layer['type'] == 'pooling':
            if layer['dim'] == '2d':
                if layer['pool'] == 'max':
                    layer_module.append(nn.MaxPool2d(kernel_size=layer['kernel_size'], stride=layer['stride'],
                                                    padding=layer['padding'], dilation=layer['dilation']))
                elif layer['pool'] == 'average':
                    layer_module.append(nn.AvgPool2d(kernel_size=layer['kernel_size'], stride=layer['stride'],
                                                    padding=layer['padding'], dilation=layer['dilation']))
                else:
                    self.raise_model_construction_error(layer['pool'], 'pool')
            elif layer['dim'] == '3d':
                if layer['pool'] == 'max':
                    layer_module.append(nn.MaxPool3d(kernel_size=layer['kernel_size'], stride=layer['stride'],
                                                    padding=layer['padding'], dilation=layer['dilation']))
                elif layer['pool'] == 'average':
                    layer_module.append(nn.AvgPool3d(kernel_size=layer['kernel_size'], stride=layer['stride'],
                                                    padding=layer['padding'], dilation=layer['dilation']))
                else:
                    self.raise_model_construction_error(layer['pool'], 'pool')
            else:
                self.raise_model_construction_error(layer['dim'], 'dim')

        elif layer['type'] == 'dense':
            pass

        else:
            self.raise_model_construction_error(layer['type'], 'type')

        # perform batch normalization
        if layer['batch_normalization']:
            if layer['dim'] == '2d':
                layer_module.append(nn.BatchNorm2d())
            if layer['dim'] == '3d':
                layer_module.append(nn.BatchNorm3d())

        # add activation function
        if layer['activation'] == 'relu':
            layer_module.append(nn.ReLU())
        elif layer['activation'] == 'tanh':
            layer_module.append(nn.Tanh())
        elif layer['activation'] == 'glu':
            layer_module.append(nn.GLU())
        else:
            self.raise_model_construction_error(layer['activation'], 'activation')

        return layer_module

    @staticmethod
    def raise_model_construction_error(key, value):
        raise ValueError('Model construction error:\n'
                         'Invalid value {value} for key "{}"'.format(value=value, key=key))

    @classmethod
    def get_class_from_config(cls, config_path):

        with open(config_path, 'r') as f:
            config = json.load(f)

        name = config['name']
        model_list = config['model']
        inception_list = config['inception'] if 'inception' in config else None

        return cls(model_list, inception_list, name)

# class Enc(nn.Module):
#     def __init__(self):
#         super(Enc, self).__init__()
#         self.batch_norm = nn.BatchNorm3d(1)
#         self.conv1 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=[3, 3, 3], stride=[1, 1, 1])
#         self.conv2 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=[3, 1, 1], stride=[1, 1, 1])
#         self.pool1 = nn.MaxPool3d(kernel_size=[3, 3, 3])
#         self.pool2 = nn.MaxPool3d(kernel_size=[1, 1, 3])
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv2(x))
#         return x
#
#
# class Dec(nn.Module):
#     def __init__(self):
#         super(Dec, self).__init__()
#         self.batch_norm = nn.BatchNorm3d(1)
#         self.conv1 = nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=[3, 3, 3], stride=[1, 1, 1])
#         self.conv2 = nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=[3, 1, 1], stride=[1, 1, 1])
#
#     def forward(self, x):
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv1(x))
#         x = F.sigmoid(self.conv1(x))
#         return x
