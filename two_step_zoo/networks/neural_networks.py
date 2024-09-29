import math
import numpy as np
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, conv1x1


class BaseNetworkClass(nn.Module):
    def __init__(self, output_split_sizes):
        super().__init__()

        self.output_split_sizes = output_split_sizes

    def _get_correct_nn_output_format(self, nn_output, split_dim):
        if self.output_split_sizes is not None:
            return torch.split(nn_output, self.output_split_sizes, dim=split_dim)
        else:
            return nn_output

    def _apply_spectral_norm(self):
        for module in self.modules():
            if "weight" in module._parameters:
                nn.utils.spectral_norm(module)

    def forward(self, x, y):
        raise NotImplementedError("Define in child classes.")


class MLP(BaseNetworkClass):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            output_dim,
            activation,
            output_split_sizes=None,
            spectral_norm=False
    ):
        super().__init__(output_split_sizes)

        layers = []
        prev_layer_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features=prev_layer_dim, out_features=hidden_dim))
            layers.append(activation())
            prev_layer_dim = hidden_dim

        layers.append(nn.Linear(in_features=hidden_dims[-1], out_features=output_dim))
        self.net = nn.Sequential(*layers)

        if spectral_norm: self._apply_spectral_norm()

    def forward(self, x, y=None):
        return self._get_correct_nn_output_format(self.net(x), split_dim=-1)
    
    
class ParallelMLP(BaseNetworkClass):
    def __init__(
            self,
            input_x_dim,
            input_y_dim,
            hidden_dims,
            output_dim,
            activation,
            output_split_sizes=None,
            spectral_norm=False
    ):
        super().__init__(output_split_sizes)

        # NOTE: First y_layer brings us into x dimension
        #       Each subsequent layer mirrors the current x dimension
        #       Each layer consists of a linear layer and an activation
        y_layers = []
        prev_layer_dim = input_y_dim
        for hidden_dim in [input_x_dim, *hidden_dims]:
            y_layer = []
            y_layer.append(nn.Linear(in_features=prev_layer_dim, out_features=hidden_dim))
            y_layer.append(activation())
            y_layers.append(nn.Sequential(*y_layer))
            prev_layer_dim = hidden_dim

        # NOTE: Each x_layer now takes an input of double the size (from the y network)
        #       As above, each layer consists of a linear layer and an activation (besides the output layer)
        x_layers = []
        prev_layer_dim = input_x_dim
        for hidden_dim in hidden_dims:
            x_layer = []
            x_layer.append(nn.Linear(in_features=2*prev_layer_dim, out_features=hidden_dim))
            x_layer.append(activation())
            x_layers.append(nn.Sequential(*x_layer))
            prev_layer_dim = hidden_dim

        x_layers.append(nn.Linear(in_features=2*hidden_dims[-1], out_features=output_dim))

        self.x_layers = nn.ModuleList(x_layers)
        self.y_layers = nn.ModuleList(y_layers)

        if spectral_norm: self._apply_spectral_norm()

    def forward(self, x, y):
        for x_layer, y_layer in zip(self.x_layers, self.y_layers):
            y = y_layer(y)
            xy = torch.cat((x, y), dim=1)
            x = x_layer(xy)

        return self._get_correct_nn_output_format(x, split_dim=-1)


class BaseCNNClass(BaseNetworkClass):
    def __init__(
            self,
            hidden_channels_list,
            stride,
            kernel_size,
            output_split_sizes=None
    ):
        super().__init__(output_split_sizes)
        self.stride = self._get_stride_or_kernel(stride, hidden_channels_list)
        self.kernel_size = self._get_stride_or_kernel(kernel_size, hidden_channels_list)

    def _get_stride_or_kernel(self, s_or_k, hidden_channels_list):
        if type(s_or_k) not in [list, tuple]:
            return [s_or_k for _ in hidden_channels_list]
        else:
            assert len(s_or_k) == len(hidden_channels_list), \
                "Mismatch between stride/kernels provided and number of hidden channels."
            return s_or_k


class CNN(BaseCNNClass):
    def __init__(
            self,
            input_channels,
            hidden_channels_list,
            output_dim,
            kernel_size,
            stride,
            image_height,
            activation,
            output_split_sizes=None,
            spectral_norm=False,
            noise_dim=0
    ):
        super().__init__(hidden_channels_list, stride, kernel_size, output_split_sizes)

        cnn_layers = []
        prev_channels = input_channels
        for hidden_channels, k, s in zip(hidden_channels_list, self.kernel_size, self.stride):
            cnn_layers.append(nn.Conv2d(prev_channels, hidden_channels, k, s))
            cnn_layers.append(activation())
            prev_channels = hidden_channels

            # NOTE: Assumes square image
            image_height = self._get_new_image_height(image_height, k, s)
        self.cnn_layers = nn.ModuleList(cnn_layers)

        self.fc_layer = nn.Linear(
            in_features=prev_channels*image_height**2+noise_dim,
            out_features=output_dim
        )

        if spectral_norm: self._apply_spectral_norm()

    def forward(self, x, eps=None):
        for layer in self.cnn_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)

        net_in = torch.cat((x, eps), dim=1) if eps is not None else x
        net_out = self.fc_layer(net_in)

        return self._get_correct_nn_output_format(net_out, split_dim=1)

    def _get_new_image_height(self, height, kernel, stride):
        # cf. https://pytorch.org/docs/1.9.1/generated/torch.nn.Conv2d.html
        # Assume dilation = 1, padding = 0
        return math.floor((height - kernel)/stride + 1)


class T_CNN(BaseCNNClass):
    def __init__(
            self,
            input_dim,
            hidden_channels_list,
            output_channels,
            kernel_size,
            stride,
            image_height,
            activation,
            output_split_sizes=None,
            spectral_norm=False,
            single_sigma=False,
    ):
        super().__init__(hidden_channels_list, stride, kernel_size, output_split_sizes)

        self.single_sigma = single_sigma

        if self.single_sigma:
            # NOTE: In the MLP above, the single_sigma case can be handled by correctly
            #       specifying output_split_sizes. However, here the first output of the
            #       network will be of image shape, which more strongly contrasts with the
            #       desired sigma output of shape (batch_size, 1). We need the additional
            #       linear layer to project the image-like output to a scalar.
            self.sigma_output_layer = nn.Linear(output_split_sizes[-1]*image_height**2, 1)

        output_paddings = []
        for _, k, s in zip(hidden_channels_list, self.kernel_size[::-1], self.stride[::-1]):
            # First need to infer the appropriate number of outputs for the first linear layer
            image_height, output_padding = self._get_new_image_height_and_output_padding(
                image_height, k, s
            )
            output_paddings.append(output_padding)
        output_paddings = output_paddings[::-1]

        self.fc_layer = nn.Linear(input_dim, hidden_channels_list[0]*image_height**2)
        self.post_fc_shape = (hidden_channels_list[0], image_height, image_height)

        t_cnn_layers = [activation()]
        prev_channels = hidden_channels_list[0]
        for hidden_channels, k, s, op in zip(
            hidden_channels_list[1:], self.kernel_size[:-1], self.stride[:-1], output_paddings[:-1]
        ):
            t_cnn_layers.append(
                nn.ConvTranspose2d(prev_channels, hidden_channels, k, s, output_padding=op)
            )
            t_cnn_layers.append(activation())

            prev_channels = hidden_channels

        t_cnn_layers.append(
            nn.ConvTranspose2d(
                prev_channels, output_channels, self.kernel_size[-1], self.stride[-1],
                output_padding=output_paddings[-1]
            )
        )
        self.t_cnn_layers = nn.ModuleList(t_cnn_layers)

        if spectral_norm: self._apply_spectral_norm()

    def forward(self, x):
        x = self.fc_layer(x).reshape(-1, *self.post_fc_shape)

        for layer in self.t_cnn_layers:
            x = layer(x)
        net_output = self._get_correct_nn_output_format(x, split_dim=1)

        if self.single_sigma:
            mu, log_sigma_unprocessed = net_output
            log_sigma = self.sigma_output_layer(log_sigma_unprocessed.flatten(start_dim=1))
            return mu, log_sigma.view(-1, 1, 1, 1)
        else:
            return net_output

    def _get_new_image_height_and_output_padding(self, height, kernel, stride):
        # cf. https://pytorch.org/docs/1.9.1/generated/torch.nn.ConvTranspose2d.html
        # Assume dilation = 1, padding = 0
        output_padding = (height - kernel) % stride
        height = (height - kernel - output_padding) // stride + 1

        return height, output_padding


class GaussianMixtureLSTM(BaseNetworkClass):
    def __init__(self, input_size,  hidden_size, num_layers, k_mixture):
        output_split_sizes = [k_mixture, k_mixture * input_size, k_mixture * input_size]
        super().__init__(output_split_sizes)
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=k_mixture*(2*input_size+1))
        self.hidden_size = hidden_size
        self.k = k_mixture

    def forward(self, x, y=None, return_h_c=False, h_c=None, not_sampling=True):
        if not_sampling:
            x = torch.flatten(x, start_dim=2)
            if y is not None:
                x = torch.cat((torch.unsqueeze(y, 2), x), 2)
            x = torch.permute(x, (0, 2, 1))  # move "channels" to last axis
            x = x[:, :-1]  # last coordinate is never used as input
        if h_c is None:
            out, h_c = self.rnn(x)
        else:
            out, h_c = self.rnn(x, h_c)
        if not_sampling:
            out = torch.cat((torch.zeros((out.shape[0], 1, self.hidden_size)).to(out.device), out), dim=1)
        out = self.linear(out)
        weights, mus, sigmas = self.split_transform_and_reshape(out)  # NOTE: weights contains logits
        if not_sampling and y is not None:
            weights = weights[:, 1:]
            mus = mus[:, 1:]
            sigmas = sigmas[:, 1:]
        if return_h_c:
            return weights, mus, sigmas, h_c
        else:
            return weights, mus, sigmas

    def split_transform_and_reshape(self, out):
        weights, mus, sigmas = self._get_correct_nn_output_format(out, split_dim=-1)
        mus = torch.reshape(mus, (mus.shape[0], mus.shape[1], self.k, -1))
        sigmas = torch.reshape(torch.exp(sigmas), (sigmas.shape[0], sigmas.shape[1], self.k, -1))
        return weights, mus, sigmas


class ResidualStack(nn.Module):
    def __init__(
        self,
        block,
        layer_channels,
        blocks_per_layer,
        norm,
        input_channel_dim,
        layer_strides=None,
        upsample_layer=False
    ):
        super().__init__()
        self.norm = norm
        self.input_channel_dim = input_channel_dim

        if layer_strides == None:
            layer_strides = [2 for _ in range(len(layer_channels))]

        layers = []
        assert len(layer_channels) == len(blocks_per_layer)
        for channel_dim,num_blocks,stride in zip(layer_channels, blocks_per_layer, layer_strides):

            if upsample_layer:
                layers.append(
                    nn.Upsample(
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=False
                    )
                )

            layers.append(
                self._make_layer(
                    block=block,
                    channel_dim=channel_dim,
                    num_blocks=num_blocks,
                    stride=stride
                )
            )
        self.layers = nn.Sequential(*layers)

    def _make_layer(
        self,
        block,
        channel_dim,
        num_blocks,
        stride=2
    ):
        norm_layer = self.norm

        downsample = nn.Sequential(
            conv1x1(self.input_channel_dim, channel_dim * block.expansion, stride),
            norm_layer(channel_dim * block.expansion),
        )

        blocks = []
        blocks.append(
            block(self.input_channel_dim, channel_dim, stride=stride, downsample=downsample, norm_layer=self.norm)
        )
        self.input_channel_dim = channel_dim
        for _ in range(1, num_blocks):
            blocks.append(
                block(
                    channel_dim,
                    channel_dim,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*blocks)

    def forward(self, x, y=None):
        return self.layers(x)


class ParallelResidualStack(nn.Module):
    def __init__(
        self,
        block,
        layer_channels,
        blocks_per_layer,
        norm,
        input_channel_dim,
        layer_strides=None,
        upsample_layer=False
    ):
        super().__init__()
        self.norm = norm
        self.input_channel_dim = input_channel_dim

        if layer_strides == None:
            layer_strides = [2 for _ in range(len(layer_channels))]

        x_layers = []
        y_layers = []
        assert len(layer_channels) == len(blocks_per_layer)
        for channel_dim,num_blocks,stride in zip(layer_channels, blocks_per_layer, layer_strides):

            x_layer = []
            y_layer = []
            if upsample_layer:
                x_layer.append(
                    nn.Upsample(
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=False
                    )
                )
                y_layer.append(
                    nn.Upsample(
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=False
                    )
                )

            x_block, y_block = self._make_layers(
                block=block,
                channel_dim=channel_dim,
                num_blocks=num_blocks,
                stride=stride
            )
            x_layer.append(x_block)
            y_layer.append(y_block)

            x_layers.append(nn.Sequential(*x_layer))
            y_layers.append(nn.Sequential(*y_layer))

        self.x_layers = nn.ModuleList(x_layers)
        self.y_layers = nn.ModuleList(y_layers)

    def _make_layers(
        self,
        block,
        channel_dim,
        num_blocks,
        stride=2
    ):
        norm_layer = self.norm

        x_downsample = nn.Sequential(
            conv1x1(2*self.input_channel_dim, channel_dim * block.expansion, stride),
            norm_layer(channel_dim * block.expansion),
        )
        y_downsample = nn.Sequential(
            conv1x1(self.input_channel_dim, channel_dim * block.expansion, stride),
            norm_layer(channel_dim * block.expansion),
        )

        x_blocks = []
        y_blocks = []
        x_blocks.append(
            block(2*self.input_channel_dim, channel_dim, stride=stride, downsample=x_downsample, norm_layer=self.norm)
        )
        y_blocks.append(
            block(self.input_channel_dim, channel_dim, stride=stride, downsample=y_downsample, norm_layer=self.norm)
        )
        self.input_channel_dim = channel_dim
        for _ in range(1, num_blocks):
            x_blocks.append(
                block(
                    channel_dim,
                    channel_dim,
                    norm_layer=norm_layer,
                )
            )
            y_blocks.append(
                block(
                    channel_dim,
                    channel_dim,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*x_blocks), nn.Sequential(*y_blocks)

    def forward(self, x, y):
        for x_layer, y_layer in zip(self.x_layers, self.y_layers):
            xy = torch.cat((x, y), dim=1)
            x = x_layer(xy)
            y = y_layer(y)

        return x, y


class ResidualEncoder(BaseNetworkClass):
    def __init__(
        self,
        layer_channels,
        blocks_per_layer,
        output_dim,
        input_channels,
        output_split_sizes=None,
        input_channel_dim=64,
        norm=nn.BatchNorm2d,
        block=BasicBlock,
        noise_dim=0,
        **kwargs
    ):
        super().__init__(output_split_sizes)
        self.norm = norm
        self.input_channel_dim = input_channel_dim

        self.conv1 = nn.Conv2d(input_channels, self.input_channel_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm(self.input_channel_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = ResidualStack(block, layer_channels, blocks_per_layer, norm=self.norm, input_channel_dim=self.input_channel_dim)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(layer_channels[-1] * block.expansion + noise_dim, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y=None, eps=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if eps is not None:
            x = torch.cat((x, eps), dim=1)
        else:
            x = x

        net_output = self.fc(x)

        return self._get_correct_nn_output_format(net_output, split_dim=1)


class ResidualDecoder(BaseNetworkClass):
    def __init__(
        self,
        input_dim,
        layer_channels,
        blocks_per_layer,
        output_channels,
        image_height,
        image_width,
        output_split_sizes=None,
        single_sigma=False,
        norm=nn.BatchNorm2d,
        block=BasicBlock
    ):
        super().__init__(output_split_sizes)
        self.single_sigma = single_sigma
        self.norm = norm
        self.layer_channels = layer_channels + [output_channels]
        self.num_blocks = len(self.layer_channels)

        if single_sigma:
            self.sigma_output_layer = nn.Linear(output_split_sizes[-1]*image_height*image_width, 1)

        self.layers = ResidualStack(
            block=block,
            layer_channels=self.layer_channels,
            blocks_per_layer=blocks_per_layer,
            layer_strides=[1 for _ in range(len(self.layer_channels))],
            upsample_layer=True,
            norm=self.norm,
            input_channel_dim=layer_channels[0]
        )
        self.layers.layers[-1][-1].relu = nn.Identity() # Last layer should not have ReLU

        height_ratio = math.ceil(image_height / (2**self.num_blocks))
        width_ratio = math.ceil(image_width / (2**self.num_blocks))
        self.fc_layer = nn.Linear(input_dim, layer_channels[0] * height_ratio * width_ratio)
        self.post_fc_shape = (layer_channels[0], image_height, image_width)

    def forward(self, x, y=None):

        x = self.fc_layer(x).reshape(
            -1,
            self.post_fc_shape[0],
            math.ceil(self.post_fc_shape[1] / (2**self.num_blocks)),
            math.ceil(self.post_fc_shape[2] / (2**self.num_blocks))
        )

        net_output = self.layers(x)
        net_output = net_output[:, :, :self.post_fc_shape[1], :self.post_fc_shape[2]]

        net_output = self._get_correct_nn_output_format(net_output, split_dim=1)

        if self.single_sigma:
            mu, log_sigma_unprocessed = net_output
            log_sigma = self.sigma_output_layer(log_sigma_unprocessed.flatten(start_dim=1))
            return mu, log_sigma.view(-1, 1, 1, 1)
        else:
            return net_output


class ParallelResidualEncoder(BaseNetworkClass):
    def __init__(
        self,
        layer_channels,
        blocks_per_layer,
        output_dim,
        input_channels,
        input_shape,
        label_dim=None,
        output_split_sizes=None,
        input_channel_dim=64,
        norm=nn.BatchNorm2d,
        block=BasicBlock,
        noise_dim=0,
    ):
        super().__init__(output_split_sizes)
        self.norm = norm
        self.input_channel_dim = input_channel_dim
        self.input_shape = input_shape

        # NOTE: Force y representations to be same shape as x
        if label_dim is None:
            raise ValueError("ParallelResidualEncoder needs the y label dimension")
        self.y_proj_layer = nn.Linear(label_dim, np.prod(input_shape))

        # NOTE: Above note assumes x net will have 2x the channel inputs
        x_layer_1 = []
        x_layer_1.append(nn.Conv2d(2*input_channels, self.input_channel_dim, kernel_size=7, stride=2, padding=3, bias=False))
        x_layer_1.append(norm(self.input_channel_dim))
        x_layer_1.append(nn.ReLU(inplace=True))
        x_layer_1.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.x_layer_1 = nn.Sequential(*x_layer_1)

        y_layer_1 = []
        y_layer_1.append(nn.Conv2d(input_channels, self.input_channel_dim, kernel_size=7, stride=2, padding=3, bias=False))
        y_layer_1.append(norm(self.input_channel_dim))
        y_layer_1.append(nn.ReLU(inplace=True))
        y_layer_1.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.y_layer_1 = nn.Sequential(*y_layer_1)

        self.parallel_stack = ParallelResidualStack(
            block,
            layer_channels,
            blocks_per_layer,
            norm=self.norm,
            input_channel_dim=self.input_channel_dim
        )

        self.x_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.y_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(2*layer_channels[-1] * block.expansion + noise_dim, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y, eps=None):
        y = self.y_proj_layer(y).reshape(-1, *self.input_shape)

        xy = torch.cat((x, y), dim=1)
        x = self.x_layer_1(xy)
        y = self.y_layer_1(y)

        x, y = self.parallel_stack(x, y)

        x = self.x_avgpool(x)
        y = self.y_avgpool(y)

        xy = torch.cat((x, y), dim=1)
        xy = torch.flatten(xy, 1)

        if eps is not None:
            xy = torch.cat((xy, eps), dim=1)
        else:
            xy = xy

        net_output = self.fc(xy)

        return self._get_correct_nn_output_format(net_output, split_dim=1)


class ParallelResidualDecoder(BaseNetworkClass):
    def __init__(
        self,
        input_dim,
        layer_channels,
        blocks_per_layer,
        output_channels,
        image_height,
        image_width,
        output_split_sizes=None,
        single_sigma=False,
        norm=nn.BatchNorm2d,
        block=BasicBlock
    ):
        super().__init__(output_split_sizes)
        self.single_sigma = single_sigma
        self.norm = norm
        self.layer_channels = layer_channels + [output_channels]
        self.num_blocks = len(self.layer_channels)

        if single_sigma:
            self.sigma_output_layer = nn.Linear(2*output_split_sizes[-1]*image_height*image_width, 1)

        self.parallel_stack = ParallelResidualStack(
            block=block,
            layer_channels=self.layer_channels,
            blocks_per_layer=blocks_per_layer,
            layer_strides=[1 for _ in range(len(self.layer_channels))],
            upsample_layer=True,
            norm=self.norm,
            input_channel_dim=layer_channels[0]
        )
        self.parallel_stack.x_layers[-1][-1][-1].relu = nn.Identity()

        height_ratio = math.ceil(image_height / (2**self.num_blocks))
        width_ratio = math.ceil(image_width / (2**self.num_blocks))
        self.y_proj_layer = nn.Linear(1, input_dim)
        self.x_fc_layer = nn.Linear(2*input_dim, layer_channels[0] * height_ratio * width_ratio)
        self.y_fc_layer = nn.Linear(input_dim, layer_channels[0] * height_ratio * width_ratio)
        self.post_fc_shape = (layer_channels[0], image_height, image_width)

    def forward(self, x, y):
        y = self.y_proj_layer(y)

        xy = torch.cat((x, y), dim=1)
        x = self.x_fc_layer(xy).reshape(
            -1,
            self.post_fc_shape[0],
            math.ceil(self.post_fc_shape[1] / (2**self.num_blocks)),
            math.ceil(self.post_fc_shape[2] / (2**self.num_blocks))
        )
        y = self.y_fc_layer(y).reshape(
            -1,
            self.post_fc_shape[0],
            math.ceil(self.post_fc_shape[1] / (2**self.num_blocks)),
            math.ceil(self.post_fc_shape[2] / (2**self.num_blocks))
        )

        x, y = self.parallel_stack(x, y)

        x = x[:, :, :self.post_fc_shape[1], :self.post_fc_shape[2]]
        x_output = self._get_correct_nn_output_format(x, split_dim=1)

        if self.single_sigma:
            y = y[:, :, :self.post_fc_shape[1], :self.post_fc_shape[2]]
            y_output = self._get_correct_nn_output_format(y, split_dim=1)

            mu, log_sigma_unprocessed = x_output[0], torch.cat((x_output[1], y_output[1]), dim=1)
            log_sigma = self.sigma_output_layer(log_sigma_unprocessed.flatten(start_dim=1))
            return mu, log_sigma.view(-1, 1, 1, 1)
        else:
            return x_output
