# Modification of hypnettorch file
# https://hypnettorch.readthedocs.io/en/latest/_modules/hypnettorch/mnets/resnet_imgnet.html#ResNetIN
# The structure of ResNet18 is based on https://github.com/grypesc/AdaGauss/blob/b719dec738f0a248d0aa80a09ce15b3219733d68/src/approach/models/resnet18.py

from hypnettorch.mnets.classifier_interface import Classifier
from hypnettorch.mnets.mlp import MLP
from hypnettorch.mnets.wide_resnet import WRN

import torch
import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet18(nn.Module):

    def __init__(self, block, layers, num_features=64, is_224=False):
        super(ResNet18, self).__init__()
        self.inplanes = 64
        self.num_features = num_features
        self.is_224 = is_224
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if is_224:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bottleneck = nn.Conv2d(512, num_features, 1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.is_224:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bottleneck(x)
        x = self.avgpool(x).squeeze(2).squeeze(2)
        return x


def resnet18(num_features, is_224, **kwargs):
    model = ResNet18(BasicBlock, [2, 2, 2, 2], num_features, is_224, **kwargs)
    return model


class PretrainedResNet18(Classifier):
    """
    ResNet-18 with weigths pretrained on ImageNet dataset. The weights are applied to a feature extractor part.
    However, a hypernetwork generates weights to a classification linear head.
    Right now, only images with input shape (224, 224, 3) are applicable.
    """

    def __init__(
        self,
        in_shape=(224, 224, 3),
        num_classes=10,
        no_weights=True,
        verbose=True,
        num_features=32,
        **kwargs
    ):
        super(PretrainedResNet18, self).__init__(num_classes, verbose)

        assert no_weights, "Weights should be entirely generated by a hypernetwork!"
        assert in_shape == (224,224,3), "Please reshape your data!"

        self.feature_extractor = resnet18(num_features=num_features, is_224=True)

        # wget https://download.pytorch.org/models/resnet18-f37072fd.pth
        state_dict = torch.load("/home/patrykkrukowski/Projects/Hyper_IBP_CL/Hyper_IBP_CL/SavedModels/CUB200/resnet18-f37072fd.pth")
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
        self.feature_extractor.load_state_dict(state_dict, strict=False)

        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad_(False)

        self._in_shape = (in_shape[2], in_shape[0], in_shape[1])

        self.linear_head = MLP(
            n_in=num_features,
            n_out=num_classes,
            hidden_layers=(),
            no_weights=no_weights,
            dropout_rate=-1,
            bn_track_stats=False,
            verbose=False
        )

        self._param_shapes = self.linear_head.param_shapes

        if verbose:
            print(f"Creating ResNet-18 model with weights pretrained on ImageNet.")


    def forward(self, x, weights=None, distilled_params=None, condition=None):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Parameters:
        -----------
            (....): See docstring of method
                :meth:`mnets.resnet.ResNet.forward`. We provide some more
                specific information below.
            x: torch.Tensor
                Based on the constructor argument ``chw_input_format``, either a flattened image batch with
                encoding ``HWC`` or an unflattened image batch with encoding
                ``CHW`` is expected.

        Returns:
        --------
            (torch.Tensor): The output of the network.
        """

        x = x.reshape((-1, *self._in_shape))
        
        # Forward pass through feature extractor
        x = self.feature_extractor(x)
        x = x.flatten(start_dim=1)

        # Forward pass through linear head
        x = self.linear_head.forward(x=x, weights=weights)

        return x
    

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.

        This method will return the current batch statistics of all batch
        normalization layers if ``distill_bn_stats`` and ``use_batch_norm``
        were set to ``True`` in the constructor.

        Returns:
        --------
            The target tensors corresponding to the shapes specified in
            attribute :attr:`hyper_shapes_distilled`.
        """
        if self.hyper_shapes_distilled is None:
            return None

        ret = []
        for bn_layer in self._batchnorm_layers:
            ret.extend(bn_layer.get_stats())

        return ret

    def _compute_layer_out_sizes(self):
        """Compute the output shapes of all layers in this network excluding
        skip connection layers.

        This method will compute the output shape of each layer in this network,
        including the output layer, which just corresponds to the number of
        classes.

        Returns:
        ---------
            (list): A list of shapes (lists of integers). The first entry will
            correspond to the shape of the output of the first convolutional
            layer. The last entry will correspond to the output shape.

            .. note:
                Output shapes of convolutional layers will adhere PyTorch
                convention, i.e., ``[C, H, W]``, where ``C`` denotes the channel
                dimension.
        """
        in_shape = self._in_shape
        fs = self._filter_sizes
        init_ks = self._init_kernel_size
        stride_init = self._init_stride
        pd_init = self._init_padding

        # Note, `in_shape` is in Tensorflow layout.
        assert len(in_shape) == 3
        in_shape = [in_shape[2], *in_shape[:2]]

        ret = []

        C, H, W = in_shape

        # Recall the formular for convolutional layers:
        # W_new = (W - K + 2P) // S + 1

        # First conv layer.
        C = fs[0]
        H = (H - init_ks[0] + 2 * pd_init) // stride_init + 1
        W = (W - init_ks[1] + 2 * pd_init) // stride_init + 1
        ret.append([C, H, W])

        def add_block(H, W, C, stride):
            if self._bottleneck_blocks:
                H = (H - 1 + 2 * 0) // stride + 1
                W = (W - 1 + 2 * 0) // stride + 1
                ret.append([C, H, W])

                H = (H - 3 + 2 * 1) // 1 + 1
                W = (W - 3 + 2 * 1) // 1 + 1
                ret.append([C, H, W])

                C = 4 * C
                H = (H - 1 + 2 * 0) // 1 + 1
                W = (W - 1 + 2 * 0) // 1 + 1
                ret.append([C, H, W])

            else:
                H = (H - 3 + 2 * 1) // stride + 1
                W = (W - 3 + 2 * 1) // stride + 1
                ret.append([C, H, W])

                H = (H - 3 + 2 * 1) // 1 + 1
                W = (W - 3 + 2 * 1) // 1 + 1
                ret.append([C, H, W])

            return H, W, C

        # Group conv2_x
        if not self._cutout_mod:  # Max-pooling layer.
            H = (H - 3 + 2 * 1) // 2 + 1
            W = (W - 3 + 2 * 1) // 2 + 1

        for b in range(self._num_blocks[0]):
            H, W, C = add_block(H, W, fs[1], 1)

        # Group conv3_x
        for b in range(self._num_blocks[1]):
            H, W, C = add_block(H, W, fs[2], 2 if b == 0 else 1)

        # Group conv4_x
        for b in range(self._num_blocks[2]):
            H, W, C = add_block(H, W, fs[3], 2 if b == 0 else 1)

        # Group conv5_x
        for b in range(self._num_blocks[3]):
            H, W, C = add_block(H, W, fs[4], 2 if b == 0 else 1)

        # Final fully-connected layer (after avg pooling), i.e., output size.
        ret.append([self._num_classes])

        return ret

    def get_output_weight_mask(self, out_inds=None, device=None):
        """Create a mask for selecting weights connected solely to certain
        output units.

        See docstring of overwritten super method
        :meth:`mnets.mnet_interface.MainNetInterface.get_output_weight_mask`.
        """
        return WRN.get_output_weight_mask(self, out_inds=out_inds, device=device)

if __name__ == "__main__":
    pass