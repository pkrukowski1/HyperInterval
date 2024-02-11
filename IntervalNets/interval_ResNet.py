#!/usr/bin/env python3
# Copyright 2019 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :mnets/resnet.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :09/20/2019
# @version        :1.0
# @python_version :3.6.8
"""
ResNet
------

This module implements the class of Resnet networks described in section 4.2 of
the following paper:

    "Deep Residual Learning for Image Recognition", He et al., 2015
    https://arxiv.org/abs/1512.03385
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hypnettorch.mnets.classifier_interface import Classifier
from IntervalNets.interval_modules import (IntervalBatchNorm2d,
                              IntervalConv2d,
                              IntervalLinear,
                              IntervalAvgPool2d)
from hypnettorch.utils.torch_utils import init_params
from typing import cast
from torch import Tensor

class IntervalResNet(Classifier):
   
    def __init__(self, in_shape=(32, 32, 3), num_classes=10, use_bias=True,
                 num_feature_maps=(16, 16, 32, 64), verbose=True, n=5, k=1,
                 no_weights=False, init_weights=None, use_batch_norm=True,
                 bn_track_stats=True, distill_bn_stats=False, **kwargs):
        super(IntervalResNet, self).__init__(num_classes, verbose)

        self._in_shape = in_shape
        self._n = n

        assert init_weights is None, "Weight initialization is deprecated!"

        assert(init_weights is None or not no_weights)
        self._no_weights = no_weights

        assert(not use_batch_norm or (not distill_bn_stats or bn_track_stats))

        self._use_batch_norm = use_batch_norm
        self._bn_track_stats = bn_track_stats
        self._distill_bn_stats = distill_bn_stats and use_batch_norm

        self._kernel_size = [3, 3]
        if len(num_feature_maps) != 4:
            raise ValueError('Option "num_feature_maps" must be a list of 4 ' +
                             'integers.')
        self._filter_sizes = list(num_feature_maps)
        for i in range(1, 4):
            if k != 1:
                self._filter_sizes[i] = k * num_feature_maps[i]
            if num_feature_maps[i] < num_feature_maps[i-1]:
                raise ValueError('We currently require the number of ' +
                                 'channels to stay constant or to increase, ' +
                                 'in which case we apply zero-padding to the ' +
                                 'shortcut connections.')

        self._has_bias = use_bias
        self._has_fc_out = True
        # We need to make sure that the last 2 entries of `weights` correspond
        # to the weight matrix and bias vector of the last layer.
        self._mask_fc_out = True
        # We don't use any output non-linearity.
        self._has_linear_out = True

        self._param_shapes = []
        self._param_shapes_meta = []
        self._internal_lower_params = None if no_weights else nn.ParameterList()
        self._internal_middle_params = None if no_weights else nn.ParameterList()
        self._internal_upper_params = None if no_weights else nn.ParameterList()
        self._hyper_shapes_learned = None \
            if not no_weights else []
        self._hyper_shapes_learned_ref = None if self._hyper_shapes_learned \
            is None else []


        ################################################
        ### Define and initialize batch norm weights ###
        ################################################
        self._batchnorm_layers = nn.ModuleList() if use_batch_norm else None

        if use_batch_norm:
            if distill_bn_stats:
                self._hyper_shapes_distilled = []

            for i, s in enumerate(self._filter_sizes):
                if i == 0:
                    num = 1
                else:
                    num = 2*n

                for j in range(num):
                    bn_layer = IntervalBatchNorm2d(s, affine=not no_weights,
                    interval_statistics=bn_track_stats)

                    self._batchnorm_layers.append(bn_layer)

                    if distill_bn_stats:
                        self._hyper_shapes_distilled.extend( \
                            [list(p.shape) for p in bn_layer.get_stats(0)])

        # Note, method `_compute_hyper_shapes` doesn't take context-mod into
        # consideration.
        internal_weight_shapes = self._compute_hyper_shapes(no_weights=True)
        self._param_shapes.extend(internal_weight_shapes)
        # It's a bit hacky, as it was post-hoc integrated.
        # `internal_weight_shapes` contains first all batchnorm shapes, then
        # all conv layer shapes and finally the weights of the output layer.
        ii = 0
        if use_batch_norm:
            while True:
                if len(internal_weight_shapes[ii]) == 1:
                    self._param_shapes_meta.append({
                        'name': 'bn_scale' if ii % 2 == 0 else 'bn_shift',
                        'index': -1 if no_weights else \
                            len(self._internal_middle_params)+ii,
                        'layer': -1 # TODO implement
                    })
                    ii += 1
                else:
                    break
        assert len(internal_weight_shapes[ii]) == 4
        while True:
            assert len(internal_weight_shapes[ii]) in [4, 2]
            self._param_shapes_meta.append({
                'name': 'weight',
                'index': -1 if no_weights else len(self._internal_middle_params)+ii,
                'layer': -1 # TODO implement
            })
            if use_bias:
                self._param_shapes_meta.append({
                    'name': 'bias',
                    'index': -1 if no_weights else \
                        len(self._internal_middle_params)+ii+1,
                    'layer': -1 # TODO implement
                })
            if len(internal_weight_shapes[ii]) == 2:
                break
            if use_bias:
                ii += 2
            else:
                ii += 1
        assert len(self._param_shapes) == len(self._param_shapes_meta)

        self._layer_upper_weight_tensors  = nn.ParameterList()
        self._layer_middle_weight_tensors = nn.ParameterList()
        self._layer_lower_weight_tensors  = nn.ParameterList()

        self._layer_upper_bias_vectors  = nn.ParameterList()
        self._layer_middle_bias_vectors = nn.ParameterList()
        self._layer_lower_bias_vectors  = nn.ParameterList()

        ###########################
        ### Print infos to user ###
        ###########################
        # Compute the total number of weights in this network and display
        # them to the user.
        # Note, this complicated calculation is not necessary as we can simply
        # count the number of weights afterwards. But it's an additional sanity
        # check for us.
        fs = self._filter_sizes
        num_weights = np.prod(self._kernel_size) * \
            (in_shape[2] * fs[0] + np.sum([fs[i] * fs[i+1] + \
                (2*n-1) * fs[i+1]**2 for i in range(3)])) + \
            ((fs[0] + 2*n * np.sum([fs[i] for i in range(1, 4)])) \
             if self.has_bias else 0) + \
            fs[-1] * num_classes + (num_classes if self.has_bias else 0)

        if use_batch_norm:
            # The gamma and beta parameters of a batch norm layer are
            # learned as well.
            num_weights += 2 * (fs[0] + \
                                2*n*np.sum([fs[i] for i in range(1, 4)]))

        assert num_weights == self.num_params

        if verbose:
            print('A ResNet with %d layers and %d weights is created' \
                  % (6*n+2, num_weights)
                  + (' The network uses batchnorm.' if use_batch_norm  else ''))

        if use_batch_norm:
            for bn_layer in self._batchnorm_layers:
                self._internal_lower_params.extend([bn_layer.lower_gamma, bn_layer.lower_beta])
                self._internal_middle_params.extend([bn_layer.middle_gamma, bn_layer.middle_beta])
                self._internal_upper_params.extend([bn_layer.upper_gamma, bn_layer.upper_beta])

        ############################################
        ### Define and initialize layer weights ###
        ###########################################
        ### Does not include context-mod or batchnorm weights.
        # First layer.
        self._layer_upper_weight_tensors.append(nn.Parameter(
            torch.Tensor(self._filter_sizes[0], self._in_shape[2],
                *self._kernel_size),
            requires_grad=True))
        
        self._layer_middle_weight_tensors.append(nn.Parameter(
            torch.Tensor(self._filter_sizes[0], self._in_shape[2],
                *self._kernel_size),
            requires_grad=True))
        
        self._layer_lower_weight_tensors.append(nn.Parameter(
            torch.Tensor(self._filter_sizes[0], self._in_shape[2],
                *self._kernel_size),
            requires_grad=True))
        
        if self.has_bias:
            self._layer_upper_bias_vectors.append(nn.Parameter(
                torch.Tensor(self._filter_sizes[0]), requires_grad=True))
            
            self._layer_middle_bias_vectors.append(nn.Parameter(
                torch.Tensor(self._filter_sizes[0]), requires_grad=True))
            
            self._layer_lower_bias_vectors.append(nn.Parameter(
                torch.Tensor(self._filter_sizes[0]), requires_grad=True))

        # Each block consists of 2n layers.
        for i in range(1, len(self._filter_sizes)):
            in_filters = self._filter_sizes[i-1]
            out_filters = self._filter_sizes[i]

            for _ in range(2*n):
                self._layer_upper_weight_tensors.append(nn.Parameter(
                    torch.Tensor(out_filters, in_filters, *self._kernel_size),
                    requires_grad=True))
                
                self._layer_middle_weight_tensors.append(nn.Parameter(
                    torch.Tensor(out_filters, in_filters, *self._kernel_size),
                    requires_grad=True))
                
                self._layer_lower_weight_tensors.append(nn.Parameter(
                    torch.Tensor(out_filters, in_filters, *self._kernel_size),
                    requires_grad=True))
                
                if self.has_bias:
                    self._layer_upper_bias_vectors.append(nn.Parameter(
                        torch.Tensor(out_filters), requires_grad=True))
                    
                    self._layer_middle_bias_vectors.append(nn.Parameter(
                        torch.Tensor(out_filters), requires_grad=True))
                    
                    self._layer_lower_bias_vectors.append(nn.Parameter(
                        torch.Tensor(out_filters), requires_grad=True))
                    
                # Note, that the first layer in this block has potentially a
                # different number of input filters.
                in_filters = out_filters

        # After the average pooling, there is one more dense layer.
        self._layer_upper_weight_tensors.append(nn.Parameter(
            torch.Tensor(num_classes, self._filter_sizes[-1]),
            requires_grad=True))
        
        self._layer_middle_weight_tensors.append(nn.Parameter(
            torch.Tensor(num_classes, self._filter_sizes[-1]),
            requires_grad=True))
        
        self._layer_lower_weight_tensors.append(nn.Parameter(
            torch.Tensor(num_classes, self._filter_sizes[-1]),
            requires_grad=True))
        
        if self.has_bias:
            self._layer_upper_bias_vectors.append(nn.Parameter( \
                torch.Tensor(num_classes), requires_grad=True))
            
            self._layer_middle_bias_vectors.append(nn.Parameter( \
                torch.Tensor(num_classes), requires_grad=True))
            
            self._layer_lower_bias_vectors.append(nn.Parameter( \
                torch.Tensor(num_classes), requires_grad=True))
            

        # We add the weights interleaved, such that there are always consecutive
        # weight tensor and bias vector per layer. This fulfils the requirements
        # of attribute `mask_fc_out`.
        for i in range(len(self._layer_middle_weight_tensors)):
            self._internal_upper_params.append(self._layer_upper_weight_tensors[i])
            self._internal_middle_params.append(self._layer_middle_weight_tensors[i])
            self._internal_lower_params.append(self._layer_lower_weight_tensors[i])
            if self.has_bias:
                self._internal_upper_params.append(self._layer_upper_bias_vectors[i])
                self._internal_middle_params.append(self._layer_middle_bias_vectors[i])
                self._internal_lower_params.append(self._layer_lower_bias_vectors[i])

        ### Initialize weights.
        for i in range(len(self._layer_middle_weight_tensors)):
            init_params(self._layer_lower_weight_tensors[i],
                self._layer_lower_bias_vectors[i] if self.has_bias else None)
            
            init_params(self._layer_middle_weight_tensors[i],
                self._layer_middle_bias_vectors[i] if self.has_bias else None)
            
            init_params(self._layer_upper_weight_tensors[i],
                self._layer_upper_bias_vectors[i] if self.has_bias else None)


    def forward(self, x, upper_weights=None, middle_weights=None, lower_weights=None):
        
        if (self._no_weights or \
                self._no_weights) and \
                middle_weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        ############################################
        ### Extract which weights should be used ###
        ############################################
        # I.e., are we using internally maintained weights or externally given
        # ones or are we even mixing between these groups.
        # FIXME code mostly copied from MLP forward method.
        n_cm = self._num_context_mod_shapes()

        int_upper_weights = None
        int_middle_weights = None
        int_lower_weights = None

        if isinstance(upper_weights, dict) and isinstance(middle_weights, dict) and isinstance(lower_weights, dict):
            assert('internal_weights' in upper_weights.keys() or \
                    'mod_weights' in upper_weights.keys())
            
            assert('internal_weights' in middle_weights.keys() or \
                    'mod_weights' in middle_weights.keys())
            
            assert('internal_weights' in lower_weights.keys() or \
                    'mod_weights' in lower_weights.keys())
            
            if 'internal_weights' in upper_weights.keys() and \
                'internal_weights' in middle_weights.keys() and \
                    'internal_weights' in lower_weights.keys():
                
                int_upper_weights = upper_weights['internal_weights']
                int_middle_weights = middle_weights['internal_weights']
                int_lower_weights = lower_weights['internal_weights']
                
            if 'mod_weights' in upper_weights.keys() and \
                'mod_weights' in middle_weights.keys() and \
                'mod_weights' in lower_weights.keys():

                raise Exception("Deprecated!")
        else:
           
            assert(len(upper_weights) == len(self.param_shapes))
            assert(len(middle_weights) == len(self.param_shapes))
            assert(len(lower_weights) == len(self.param_shapes))

            
            int_upper_weights = upper_weights
            int_middle_weights = middle_weights
            int_lower_weights = lower_weights

        if int_upper_weights is None and \
            int_middle_weights is None and \
            int_lower_weights is None:

            if self._no_weights:
                raise Exception('Network was generated without internal ' +
                    'weights. Hence, they must be passed via the ' +
                    '"weights" option.')
            if self._context_mod_no_weights:
                int_upper_weights = self._upper_weights
                int_middle_weights = self._middle_weights
                int_lower_weights = self._lower_weights
            else:
                int_upper_weights = self._upper_weights[n_cm:]
                int_middle_weights = self._middle_weights[n_cm:]
                int_lower_weights = self._lower_weights[n_cm:]

        # Note, context-mod weights might have different shapes, as they
        # may be parametrized on a per-sample basis.
        
        int_shapes = self.param_shapes[n_cm:]

        assert(len(int_upper_weights) == len(int_shapes))
        assert(len(int_middle_weights) == len(int_shapes))
        assert(len(int_lower_weights) == len(int_shapes))

        for i, s in enumerate(int_shapes):
            assert(np.all(np.equal(s, list(int_upper_weights[i].shape))))
            assert(np.all(np.equal(s, list(int_middle_weights[i].shape))))
            assert(np.all(np.equal(s, list(int_lower_weights[i].shape))))


        ######################################
        ### Select batchnorm running stats ###
        ######################################
        if self._use_batch_norm:
            # There are 6*n+1 layers that use batch normalization.
            lbw = 2 * (6 * self._n + 1)

            bn_upper_weights = int_upper_weights[:lbw]
            bn_middle_weights = int_middle_weights[:lbw]
            bn_lower_weights = int_lower_weights[:lbw]

            layer_upper_weights = int_upper_weights[lbw:]
            layer_middle_weights = int_middle_weights[lbw:]
            layer_lower_weights = int_lower_weights[lbw:]
        else:
            layer_upper_weights = int_upper_weights
            layer_middle_weights = int_middle_weights
            layer_lower_weights = int_lower_weights


        ###############################################
        ### Extract weight tensors and bias vectors ###
        ###############################################
        w_upper_weights = []
        b_upper_weights = []

        w_middle_weights = []
        b_middle_weights = []

        w_lower_weights = []
        b_lower_weights = []

        for i, (p_upper, p_middle, p_lower) in enumerate(zip(
                                                            layer_upper_weights, 
                                                            layer_middle_weights, 
                                                            layer_lower_weights)):
            if self.has_bias and i % 2 == 1:
                b_upper_weights.append(p_upper)
                b_middle_weights.append(p_middle)
                b_lower_weights.append(p_lower)
            else:
                w_upper_weights.append(p_upper)
                w_middle_weights.append(p_middle)
                w_lower_weights.append(p_lower)

        ###########################
        ### Forward Computation ###
        ###########################
        cm_ind = 0
        bn_ind = 0
        layer_ind = 0

        ### Helper function to process convolutional layers.
        def conv_layer(h, stride, upper_shortcut=None, 
                               middle_shortcut=None, lower_shortcut=None):
            """Compute the output of a resnet conv layer including batchnorm,
            context-mod, non-linearity and shortcut.

            The order if the following:

            conv-layer -> batch-norm -> shortcut -> non-linearity

            This method increments the indices ``layer_ind``, ``cm_ind`` and
            ``bn_ind``.

            Args:
                h: Input activity.
                stride: Stride of conv. layer (padding is set to 1).
                shortcut: If set, this tensor will be added to the activation
                    before the non-linearity is applied.

            Returns:
                Output of layer.
            """
            nonlocal layer_ind, cm_ind, bn_ind

            assert ((upper_shortcut is not None and middle_shortcut is not None and lower_shortcut is not None) or \
                     (upper_shortcut is None and middle_shortcut is None and lower_shortcut is None))

            h = IntervalConv2d.apply_conv2d(h, 
                                            upper_weights=w_upper_weights[layer_ind],
                                            middle_weights=w_middle_weights[layer_ind],
                                            lower_weights=w_lower_weights[layer_ind],
                                            upper_bias=b_upper_weights[layer_ind],
                                            middle_bias=b_middle_weights[layer_ind],
                                            lower_bias=b_lower_weights[layer_ind],
                                            stride=stride, padding=1)
            layer_ind += 1

            # Batch-norm
            if self._use_batch_norm:
                h = self._batchnorm_layers[bn_ind].forward(h,
                        upper_gamma=bn_upper_weights[2*bn_ind],
                        middle_gamma=bn_middle_weights[2*bn_ind],
                        lower_gamma=bn_lower_weights[2*bn_ind],
                        upper_beta=bn_upper_weights[2*bn_ind+1],
                        middle_beta=bn_middle_weights[2*bn_ind+1],
                        lower_beta=bn_lower_weights[2*bn_ind+1])
                bn_ind += 1


            # Note, as can be seen in figure 5 of the original paper, the
            # shortcut is performed before the ReLU is applied.
            if upper_shortcut is not None and \
                middle_shortcut is not None and \
                    lower_shortcut is not None:

                h = h.refine_names("N", "bounds", "C", "H", "W")
                h_lower, h_middle, h_upper = map(lambda x_: cast(Tensor, x_.rename(None)), h.unbind("bounds"))  # type: ignore
            
                h_upper = h_upper + upper_shortcut
                h_lower = h_lower + lower_shortcut
                h_middle = h_middle + middle_shortcut

                h = torch.stack([h_lower, h_middle, h_upper], dim=1).refine_names("N", "bounds", "C", "H", "W")  # type: ignore

            # Non-linearity
            h = F.relu(h)

            return h

        x = x.view(-1, *self._in_shape)
        x = x.permute(0, 3, 1, 2)
        h = torch.stack([x, x, x], dim=1)

        ### Initial convolutional layer.
        h = conv_layer(h, 1, lower_shortcut=None, middle_shortcut=None, upper_shortcut=None)

        ### Three groups, each containing n resnet blocks.
        for i in range(3):
            # Only the first layer in a group may be a strided convolution.
            if i == 0:
                stride = 1
            else:
                stride = 2

            fs = self._filter_sizes[i+1]
            # For each resnet block. A resnet block consists of 2 convolutional
            # layers.
            for j in range(self._n):
                h = h.refine_names("N", "bounds", "C", "H", "W")
                h_lower, h_middle, h_upper = map(lambda x_: cast(Tensor, x_.rename(None)), h.unbind("bounds"))  # type: ignore

                shortcut_upper_h = h_upper
                shortcut_middle_h = h_middle
                shortcut_lower_h = h_lower

                if j == 0 and fs != self._filter_sizes[i]:
                    # The original paper uses zero padding for added output
                    # feature dimensions. Since we apply a strided conv, we
                    # additionally have to subsample the input.
                    # This implementation is motivated by
                    #    https://git.io/fhcfk
                    # FIXME I guess it is a nicer solution to use 1x1
                    # convolutions to increase/decrease the number of channels.
                    # Note, this would add more layers (and trainable weights)
                    # to the network. Hence, the statement, that this networks
                    # has `6n+2` layers might be invalid.
                    fs_prev = self._filter_sizes[i]
                    pad_left = (fs - fs_prev) // 2
                    pad_right = int(np.ceil((fs - fs_prev) / 2))
                    if stride == 2:
                        shortcut_upper_h = h_upper[:, :, ::2, ::2]
                        shortcut_middle_h = h_middle[:, :, ::2, ::2]
                        shortcut_lower_h = h_lower[:, :, ::2, ::2]

                    shortcut_upper_h = F.pad(shortcut_upper_h,
                        (0, 0, 0, 0, pad_left, pad_right), "constant", 0)
                    shortcut_middle_h = F.pad(shortcut_middle_h,
                        (0, 0, 0, 0, pad_left, pad_right), "constant", 0)
                    shortcut_lower_h = F.pad(shortcut_lower_h,
                        (0, 0, 0, 0, pad_left, pad_right), "constant", 0)

                h = torch.stack([h_lower, h_middle, h_upper], dim=1).refine_names("N", "bounds", "C", "H", "W")  # type: ignore
                h = conv_layer(h, stride, upper_shortcut=None, 
                               middle_shortcut=None, lower_shortcut=None)

                stride = 1

                h = conv_layer(h, stride, 
                               upper_shortcut=shortcut_upper_h,
                               middle_shortcut=shortcut_middle_h,
                               lower_shortcut=shortcut_lower_h)

        ### Average pool all activities within a feature map.
        h = IntervalAvgPool2d.apply_avg_pool2d(h, [h.size()[3], h.size()[4]])
        h = h.rename(None)
        h = h.view(h.size(0), 3, -1)

        ### Apply final fully-connected layer and compute outputs.
        h = IntervalLinear.apply_linear(h,upper_weights=w_upper_weights[layer_ind],
                                            middle_weights=w_middle_weights[layer_ind],
                                            lower_weights=w_lower_weights[layer_ind],
                                            upper_bias=b_upper_weights[layer_ind],
                                            middle_bias=b_middle_weights[layer_ind],
                                            lower_bias=b_lower_weights[layer_ind])

        return h

    def _compute_hyper_shapes(self, no_weights=None):
        r"""Helper function to compute weight shapes of this network for
        externally maintained weights.

        Returns a list of lists of integers denoting the shape of every
        weight tensor that is not a trainable parameter of this network (i.e.,
        those weight tensors whose shapes are specified in
        :attr:`mnets.mnet_interface.MainNetInterface.hyper_shapes_distilled`).

        If batchnorm layers are used, then the first :math:`2 * (6n+1)` lists
        will denote the shapes of the batchnorm weights
        :math:`[\gamma_1, \beta_1, \gamma_2, ..., \beta_{6n+1}]`.

        The remaining :math:`2 * (6n+2)` entries are weight tensors and bias
        vectors of each convolutional or fully-connected (last two entries)
        layer in this network.

        Args:
            no_weights (optional): If specified, it will overwrite the private
                member :code:`self._no_weights`.

                If set to ``True``, then all weight shapes of the network
                are computed independent of whether they are maintained
                internally or externally.

        Returns:
            A list of lists of integers.
        """
        if no_weights is None:
            no_weights = self._no_weights

        ret = []
        if no_weights is False:
            return ret

        fs = self._filter_sizes
        ks = self._kernel_size
        n = self._n

        if self._use_batch_norm:
            for i, s in enumerate(fs):
                if i == 0:
                    num = 1
                else:
                    num = 2*n

                for _ in range(2*num):
                    ret.append([s])

        f_in = self._in_shape[-1]
        for i, s in enumerate(fs):
            f_out = s
            if i == 0:
                num = 1
            else:
                num = 2*n

            for _ in range(num):
                ret.append([f_out, f_in, *ks])
                if self.has_bias:
                    ret.append([f_out])
                f_in = f_out
        ret.append([self._num_classes, fs[-1]])
        if self.has_bias:
            ret.append([self._num_classes])

        return ret

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.

        This method will return the current batch statistics of all batch
        normalization layers if ``distill_bn_stats`` and ``use_batch_norm``
        were set to ``True`` in the constructor.

        Returns:
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
        """Compute the output shapes of all layers in this network.

        This method will compute the output shape of each layer in this network,
        including the output layer, which just corresponds to the number of
        classes.

        Returns:
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
        ks = self._kernel_size
        pd = 1 # all paddings are 1.
        assert len(ks) == 2
        assert len(fs) == 4
        n = self._n

        # Note, `in_shape` is in Tensorflow layout.
        assert(len(in_shape) == 3)
        in_shape = [in_shape[2], *in_shape[:2]]

        ret = []

        C, H, W = in_shape

        # Recall the formular for convolutional layers:
        # W_new = (W - K + 2P) // S + 1

        # First conv layer (stride 1).
        C = fs[0]
        H = (H - ks[0] + 2*pd) // 1 + 1
        W = (W - ks[1] + 2*pd) // 1 + 1
        ret.append([C, H, W])

        # First block (no strides).
        C = fs[1]
        H = (H - ks[0] + 2*pd) // 1 + 1
        W = (W - ks[1] + 2*pd) // 1 + 1
        ret.extend([[C, H, W]] * (2*n))

        # Second block (first layer has stride 2).
        C = fs[2]
        H = (H - ks[0] + 2*pd) // 2 + 1
        W = (W - ks[1] + 2*pd) // 2 + 1
        ret.extend([[C, H, W]] * (2*n))

        # Third block (first layer has stride 2).
        C = fs[3]
        H = (H - ks[0] + 2*pd) // 2 + 1
        W = (W - ks[1] + 2*pd) // 2 + 1
        ret.extend([[C, H, W]] * (2*n))

        # Final fully-connected layer (after avg pooling), i.e., output size.
        ret.append([self._num_classes])

        assert len(ret) == 6*n + 2

        return ret

if __name__ == '__main__':
    pass
