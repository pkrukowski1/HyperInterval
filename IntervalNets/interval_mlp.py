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
# @title          :mnets/mlp.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :10/21/2019
# @version        :1.0
# @python_version :3.6.8
"""
Multi-Layer Perceptron
----------------------

Implementation of a fully-connected neural network.

An example usage is as a main model, that doesn't include any trainable weights.
Instead, weights are received as additional inputs. For instance, using an
auxilliary network, a so called hypernetwork, see

    Ha et al., "HyperNetworks", arXiv, 2016,
    https://arxiv.org/abs/1609.09106
"""
import torch
import torch.nn as nn
import numpy as np
from warnings import warn

from hypnettorch.mnets.mnet_interface import MainNetInterface
from hypnettorch.utils.torch_utils import init_params
from hypnettorch.mnets import MLP

from IntervalNets.interval_modules import (IntervalDropout, 
                              IntervalBatchNorm2d,
                              IntervalLinear)

class IntervalMLP(MLP, MainNetInterface):
    """Implementation of a Multi-Layer Perceptron (MLP) which works on intervals

    This is a simple fully-connected network, that receives input vector
    :math:`\mathbf{x}` and outputs a vector :math:`\mathbf{y}` of real values.

    The output mapping does not include a non-linearity by default, as we wanna
    map to the whole real line (but see argument ``out_fn``).

    Args:
        n_in (int): Number of inputs.
        n_out (int): Number of outputs.
        hidden_layers (list or tuple): A list of integers, each number denoting
            the size of a hidden layer.
        activation_fn: The nonlinearity used in hidden layers. If ``None``, no
            nonlinearity will be applied.
        use_bias (bool): Whether layers may have bias terms.
        no_weights (bool): If set to ``True``, no trainable parameters will be
            constructed, i.e., weights are assumed to be produced ad-hoc
            by a hypernetwork and passed to the :meth:`forward` method.
        init_weights (optional): This option is for convinience reasons.
            The option expects a list of parameter values that are used to
            initialize the network weights. As such, it provides a
            convinient way of initializing a network with a weight draw
            produced by the hypernetwork.

            Note, internal weights (see 
            :attr:`mnets.mnet_interface.MainNetInterface.weights`) will be
            affected by this argument only.
        dropout_rate: If ``-1``, no dropout will be applied. Otherwise a number
            between 0 and 1 is expected, denoting the dropout rate of hidden
            layers.
        use_spectral_norm: Use spectral normalization for training.
        use_batch_norm (bool): Whether batch normalization should be used. Will
            be applied before the activation function in all hidden layers.
        bn_track_stats (bool): If batch normalization is used, then this option
            determines whether running statistics are tracked in these
            layers or not (see argument ``track_running_stats`` of class
            :class:`utils.batchnorm_layer.BatchNormLayer`).

            If ``False``, then batch statistics are utilized even during
            evaluation. If ``True``, then running stats are tracked. When
            using this network in a continual learning scenario with
            different tasks then the running statistics are expected to be
            maintained externally. The argument ``stats_id`` of the method
            :meth:`utils.batchnorm_layer.BatchNormLayer.forward` can be
            provided using the argument ``condition`` of method :meth:`forward`.

            Example:
                To maintain the running stats, one can simply iterate over
                all batch norm layers and checkpoint the current running
                stats (e.g., after learning a task when applying a Continual
                learning scenario).

                .. code:: python

                    for bn_layer in net.batchnorm_layers:
                        bn_layer.checkpoint_stats()
        distill_bn_stats (bool): If ``True``, then the shapes of the batchnorm
            statistics will be added to the attribute
            :attr:`mnets.mnet_interface.MainNetInterface.\
hyper_shapes_distilled` and the current statistics will be returned by the
            method :meth:`distillation_targets`.

            Note, this attribute may only be ``True`` if ``bn_track_stats``
            is ``True``.
        use_context_mod (bool): Add context-dependent modulation layers
            :class:`utils.context_mod_layer.ContextModLayer` after the linear
            computation of each layer.
        context_mod_inputs (bool): Whether context-dependent modulation should
            also be applied to network intpus directly. I.e., assume
            :math:`\mathbf{x}` is the input to the network. Then the first
            network operation would be to modify the input via
            :math:`\mathbf{x} \cdot \mathbf{g} + \mathbf{s}` using context-
            dependent gain and shift parameters.

            Note:
                Argument applies only if ``use_context_mod`` is ``True``.
        no_last_layer_context_mod (bool): If ``True``, context-dependent
            modulation will not be applied to the output layer.

            Note:
                Argument applies only if ``use_context_mod`` is ``True``.
        context_mod_no_weights (bool): The weights of the context-mod layers
            (:class:`utils.context_mod_layer.ContextModLayer`) are treated
            independently of the option ``no_weights``.
            This argument can be used to decide whether the context-mod
            parameters (gains and shifts) are maintained internally or
            externally.

            Note:
                Check out argument ``weights`` of the :meth:`forward` method
                on how to correctly pass weights to the network that are
                externally maintained.
        context_mod_post_activation (bool): Apply context-mod layers after the
            activation function (``activation_fn``) in hidden layer rather than
            before, which is the default behavior.

            Note:
                This option only applies if ``use_context_mod`` is ``True``.

            Note:
                This option does not affect argument ``context_mod_inputs``.

            Note:
                This option does not affect argument
                ``no_last_layer_context_mod``. Hence, if a output-nonlinearity
                is applied through argument ``out_fn``, then context-modulation
                would be applied before this non-linearity.
        context_mod_gain_offset (bool): Activates option ``apply_gain_offset``
            of class :class:`utils.context_mod_layer.ContextModLayer` for all
            context-mod layers that will be instantiated.
        context_mod_gain_softplus (bool): Activates option
            ``apply_gain_softplus`` of class
            :class:`utils.context_mod_layer.ContextModLayer` for all
            context-mod layers that will be instantiated.
        out_fn (optional): If provided, this function will be applied to the
            output neurons of the network.

            Warning:
                This changes the interpretation of the output of the
                :meth:`forward` method.
        verbose (bool): Whether to print information (e.g., the number of
            weights) during the construction of the network.
    """
    def __init__(self, n_in=1, n_out=1, hidden_layers=(10, 10),
                 activation_fn=torch.nn.ReLU(), use_bias=True, no_weights=False,
                 init_weights=None, dropout_rate=-1, use_spectral_norm=False,
                 use_batch_norm=False, bn_track_stats=True,
                 distill_bn_stats=False, use_context_mod=False,
                 context_mod_inputs=False, no_last_layer_context_mod=False,
                 context_mod_no_weights=False,
                 context_mod_post_activation=False,
                 context_mod_gain_offset=False, context_mod_gain_softplus=False,
                 out_fn=None, verbose=True):
        
        MainNetInterface.__init__(self)
        MLP.__init__(self, n_in=n_in, n_out=n_out, hidden_layers=hidden_layers,
                 activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights,
                 init_weights=init_weights, dropout_rate=dropout_rate, use_spectral_norm=use_spectral_norm,
                 use_batch_norm=use_batch_norm, bn_track_stats=bn_track_stats,
                 distill_bn_stats=distill_bn_stats, use_context_mod=use_context_mod,
                 context_mod_inputs=context_mod_inputs, no_last_layer_context_mod=no_last_layer_context_mod,
                 context_mod_no_weights=context_mod_no_weights,
                 context_mod_post_activation=context_mod_post_activation,
                 context_mod_gain_offset=context_mod_gain_offset, context_mod_gain_softplus=context_mod_gain_softplus,
                 out_fn=out_fn, verbose=verbose)
        
        assert init_weights is None, "`init_weights` option is deprecated"
        assert use_context_mod is False, "`use_context_mod` is deprecated"
        assert use_spectral_norm is False, "`use_spectral_norm` is deprecated"

        # Tuple are not mutable.
        hidden_layers = list(hidden_layers)

        self._a_fun = activation_fn
        assert(init_weights is None or \
               (not no_weights or not context_mod_no_weights))
        self._no_weights = no_weights
        self._dropout_rate = dropout_rate
        self._use_batch_norm = use_batch_norm
        self._bn_track_stats = bn_track_stats
        self._out_fn = out_fn

        self._has_bias = use_bias
        self._has_fc_out = True

        # We need to make sure that the last 2 entries of `weights` correspond
        # to the weight matrix and bias vector of the last layer.
        self._mask_fc_out = True
        self._has_linear_out = True if out_fn is None else False

        self._param_shapes = []
        self._param_shapes_meta = []

        # Initialize lower, middle and upper weights
        if no_weights and context_mod_no_weights:
            self._lower_weights  = None
            self._middle_weights = None
            self._upper_weights  = None
        else:
            self._lower_weights  = nn.ParameterList()
            self._middle_weights = nn.ParameterList()
            self._upper_weights  = nn.ParameterList()
        
        self._hyper_shapes_learned = None \
            if not no_weights and not context_mod_no_weights else []
        self._hyper_shapes_learned_ref = None if self._hyper_shapes_learned \
            is None else []

        if dropout_rate != -1:
            assert(dropout_rate >= 0. and dropout_rate <= 1.)
            self._dropout = IntervalDropout(p=dropout_rate)

        ### Define and initialize batch norm weights.
        self._batchnorm_layers = nn.ModuleList() if use_batch_norm else None

        if use_batch_norm:

            bn_ind = 0
            for i, n in enumerate(hidden_layers):
                bn_layer = IntervalBatchNorm2d(n, affine=not no_weights,
                    interval_statistics=bn_track_stats)
                self._batchnorm_layers.append(bn_layer)

                self._param_shapes.extend(bn_layer.param_shapes)
                assert len(bn_layer.param_shapes) == 2
                self._param_shapes_meta.extend([
                    {'name': 'bn_scale',
                     'index': -1 if no_weights else len(self._middle_weights),
                     'layer': -1}, # 'layer' is set later.
                    {'name': 'bn_shift',
                     'index': -1 if no_weights else len(self._middle_weights)+1,
                     'layer': -1}, # 'layer' is set later.
                ])

                if no_weights:
                    self._hyper_shapes_learned.extend(bn_layer.param_shapes)
                else:
                    self._upper_weights.extend([bn_layer.upper_gamma, bn_layer.upper_beta])
                    self._middle_weights.extend([bn_layer.middle_gamma, bn_layer.middle_beta])
                    self._lower_weights.extend([bn_layer.lower_gamma, bn_layer.lower_beta])

        ### Compute shapes of linear layers.
        linear_shapes = MLP.weight_shapes(n_in=n_in, n_out=n_out,
            hidden_layers=hidden_layers, use_bias=use_bias)
        self._param_shapes.extend(linear_shapes)

        for i, s in enumerate(linear_shapes):
            self._param_shapes_meta.append({
                'name': 'weight' if len(s) != 1 else 'bias',
                'index': -1 if no_weights else len(self._middle_weights) + i,
                'layer': -1 # 'layer' is set later.
            })

        num_weights = MainNetInterface.shapes_to_num_weights(self._param_shapes)

        ### Set missing meta information of param_shapes.
        offset = 1 if use_context_mod and context_mod_inputs else 0
        shift = 1
        if use_batch_norm:
            shift += 1
        if use_context_mod:
            shift += 1

        cm_offset = 2 if context_mod_post_activation else 1
        bn_offset = 1 if context_mod_post_activation else 2

        cm_ind = 0
        bn_ind = 0
        layer_ind = 0

        for i, dd in enumerate(self._param_shapes_meta):
            if dd['name'].startswith('cm'):
                if offset == 1 and i in [0, 1]:
                    dd['layer'] = 0
                else:
                    if cm_ind < len(hidden_layers):
                        dd['layer'] = offset + cm_ind * shift + cm_offset
                    else:
                        assert cm_ind == len(hidden_layers) and \
                            not no_last_layer_context_mod
                        # No batchnorm in output layer.
                        dd['layer'] = offset + cm_ind * shift + 1

                    if dd['name'] == 'cm_shift':
                        cm_ind += 1

            elif dd['name'].startswith('bn'):
                dd['layer'] = offset + bn_ind * shift + bn_offset
                if dd['name'] == 'bn_shift':
                        bn_ind += 1

            else:
                dd['layer'] = offset + layer_ind * shift
                if not use_bias or dd['name'] == 'bias':
                    layer_ind += 1

        ### Uer information
        if verbose:
            print('Creating an MLP with %d weights' % num_weights
                  + '.'
                  + (' The network uses dropout.' if dropout_rate != -1 else '')
                  + (' The network uses batchnorm.' if use_batch_norm  else ''))

        self._layer_upper_weight_tensors = nn.ParameterList()
        self._layer_middle_weight_tensors = nn.ParameterList()
        self._layer_lower_weight_tensors = nn.ParameterList()

        self._layer_upper_bias_vectors = nn.ParameterList()
        self._layer_middle_bias_vectors = nn.ParameterList()
        self._layer_lower_bias_vectors = nn.ParameterList()

        if no_weights:
            self._hyper_shapes_learned.extend(linear_shapes)

            
            self._is_properly_setup()
            return

        ### Define and initialize linear weights.
        for i, dims in enumerate(linear_shapes):
            self._upper_weights.append(nn.Parameter(torch.Tensor(*dims),
                                              requires_grad=True))
            self._middle_weights.append(nn.Parameter(torch.Tensor(*dims),
                                              requires_grad=True))
            self._lower_weights.append(nn.Parameter(torch.Tensor(*dims),
                                              requires_grad=True))
            
            if len(dims) == 1:
                self._layer_upper_bias_vectors.append(self._upper_weights[-1])
                self._layer_middle_bias_vectors.append(self._middle_weights[-1])
                self._layer_lower_bias_vectors.append(self._lower_weights[-1])
            else:
                self._layer_upper_weight_tensors.append(self._upper_weights[-1])
                self._layer_middle_weight_tensors.append(self._middle_weights[-1])
                self._layer_lower_weight_tensors.append(self._lower_weights[-1])

        
        for i in range(len(self._layer_weight_tensors)):
            if use_bias:
                init_params(self._layer_upper_weight_tensors[i],
                            self._layer_upper_bias_vectors[i])
                
                init_params(self._layer_middle_weight_tensors[i],
                            self._layer_middle_bias_vectors[i])
                
                init_params(self._layer_lower_weight_tensors[i],
                            self._layer_lower_bias_vectors[i])
            else:
                init_params(self._layer_upper_weight_tensors[i])
                init_params(self._layer_middle_weight_tensors[i])
                init_params(self._layer_lower_weight_tensors[i])

        self._is_properly_setup()

    def forward(self, x, upper_weights, middle_weights, lower_weights):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            (....): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`. We
                provide some more specific information below.
            weights (list or dict): If a list of parameter tensors is given and
                context modulation is used (see argument ``use_context_mod`` in
                constructor), then these parameters are interpreted as context-
                modulation parameters if the length of ``weights`` equals
                :code:`2*len(net.context_mod_layers)`. Otherwise, the length is
                expected to be equal to the length of the attribute
                :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`.

                Alternatively, a dictionary can be passed with the possible
                keywords ``internal_weights`` and ``mod_weights``. Each keyword
                is expected to map onto a list of tensors.
                The keyword ``internal_weights`` refers to all weights of this
                network except for the weights of the context-modulation layers.
                The keyword ``mod_weights``, on the other hand, refers
                specifically to the weights of the context-modulation layers.
                It is not necessary to specify both keywords.

        Returns:
            (tuple): Tuple containing:

            - **y**: The output of the network.
            - **h_y** (optional): If ``out_fn`` was specified in the
              constructor, then this value will be returned. It is the last
              hidden activation (before the ``out_fn`` has been applied).
        """
        if ((not self._use_context_mod and self._no_weights) or \
                (self._no_weights or self._context_mod_no_weights)) and \
                middle_weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        ############################################
        ### Extract which weights should be used ###
        ############################################
        # I.e., are we using internally maintained weights or externally given
        # ones or are we even mixing between these groups.
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

        bn_ind = 0

        if self._use_batch_norm:
            n_bn = 2 * len(self.batchnorm_layers)

            bn_upper_weights = int_upper_weights[:n_bn]
            bn_middle_weights = int_middle_weights[:n_bn]
            bn_lower_weights = int_lower_weights[:n_bn]
            
            layer_upper_weights = int_upper_weights[n_bn:]
            layer_middle_weights = int_middle_weights[n_bn:]
            layer_lower_weights = int_lower_weights[n_bn:]
        else:
            layer_upper_weights = int_upper_weights
            layer_middle_weights = int_middle_weights
            layer_lower_weights = int_lower_weights

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
        hidden = torch.stack([x, x, x], dim=1)

        for l in range(len(w_middle_weights)):
            w_upper = w_upper_weights[l]
            w_middle = w_middle_weights[l]
            w_lower = w_lower_weights[l]

            if self.has_bias:
                b_upper = b_upper_weights[l]
                b_middle = b_middle_weights[l]
                b_lower = b_lower_weights[l]
            else:
                b_upper = None
                b_middle = None
                b_lower = None

            # Linear layer.
            hidden = IntervalLinear.apply_linear(hidden,
                                                upper_weights=w_upper,
                                                middle_weights=w_middle,
                                                lower_weights=w_lower,
                                                upper_bias=b_upper,
                                                middle_bias=b_middle,
                                                lower_bias=b_lower)

            # Only for hidden layers.
            if l < len(w_middle_weights) - 1:

                # Batch norm
                if self._use_batch_norm:
                
                    hidden = self._batchnorm_layers[bn_ind].forward(hidden,
                        upper_gamma=bn_upper_weights[2*bn_ind],
                        middle_gamma=bn_middle_weights[2*bn_ind],
                        lower_gamma=bn_lower_weights[2*bn_ind],
                        upper_beta=bn_upper_weights[2*bn_ind+1],
                        middle_beta=bn_middle_weights[2*bn_ind+1],
                        lower_beta=bn_lower_weights[2*bn_ind+1])
                    bn_ind += 1

                # Dropout
                if self._dropout_rate != -1:
                    hidden = self._dropout(hidden)

                # Non-linearity
                if self._a_fun is not None:
                    hidden = self._a_fun(hidden)

        if self._out_fn is not None:
            return self._out_fn(hidden), hidden

        return hidden

    @staticmethod
    def weight_shapes(n_in=1, n_out=1, hidden_layers=[10, 10], use_bias=True):
        """Compute the tensor shapes of all parameters in a fully-connected
        network.

        Args:
            n_in: Number of inputs.
            n_out: Number of output units.
            hidden_layers: A list of ints, each number denoting the size of a
                hidden layer.
            use_bias: Whether the FC layers should have biases.

        Returns:
            A list of list of integers, denoting the shapes of the individual
            parameter tensors.
        """
        shapes = []

        prev_dim = n_in
        layer_out_sizes = hidden_layers + [n_out]
        for i, size in enumerate(layer_out_sizes):
            shapes.append([size, prev_dim])
            if use_bias:
                shapes.append([size])
            prev_dim = size

        return shapes
    
    def _is_properly_setup(self, check_has_bias=True):
        """This method can be used by classes that implement this interface to
        check whether all required properties have been set."""
        assert(self._param_shapes is not None or self._all_shapes is not None)
        if self._param_shapes is None:
            warn('Private member "_param_shapes" should be specified in each ' +
                 'sublcass that implements this interface, since private ' +
                 'member "_all_shapes" is deprecated.', DeprecationWarning)
            self._param_shapes = self._all_shapes

        if self._hyper_shapes is not None or \
                self._hyper_shapes_learned is not None:
            if self._hyper_shapes_learned is None:
                warn('Private member "_hyper_shapes_learned" should be ' +
                     'specified in each sublcass that implements this ' +
                     'interface, since private member "_hyper_shapes" is ' +
                     'deprecated.', DeprecationWarning)
                self._hyper_shapes_learned = self._hyper_shapes
            # FIXME we should actually assert equality if
            # `_hyper_shapes_learned` was not None.
            self._hyper_shapes = self._hyper_shapes_learned

        if self._weights is not None and self._internal_params is None:
            # Note, in the future we might throw a deprecation warning here,
            # once "weights" becomes deprecated.
            self._internal_params = self._weights

        assert self._internal_params is not None or \
               self._hyper_shapes_learned is not None

        if self._hyper_shapes_learned is None and \
                self.hyper_shapes_distilled is None:
            # Note, `internal_params` should only contain trainable weights and
            # not other things like running statistics. Thus, things that are
            # passed to an optimizer.
            assert len(self._internal_params) == len(self._param_shapes)

        if self._param_shapes_meta is None:
            # Note, this attribute was inserted post-hoc.
            # FIXME Warning is annoying, programmers will notice when they use
            # this functionality.
            #warn('Attribute "param_shapes_meta" has not been implemented!')
            pass
        else:
            assert(len(self._param_shapes_meta) == len(self._param_shapes))
            for dd in self._param_shapes_meta:
                assert isinstance(dd, dict)
                assert 'name' in dd.keys() and 'index' in dd.keys() and \
                    'layer' in dd.keys()
                assert dd['name'] is None or \
                       dd['name'] in ['weight', 'bias', 'bn_scale', 'bn_shift',
                                      'cm_scale', 'cm_shift', 'embedding']

                assert isinstance(dd['index'], int)
                if self._internal_params is None:
                    assert dd['index'] == -1
                else:
                    assert dd['index'] == -1 or \
                        0 <= dd['index'] < len(self._internal_params)

                assert isinstance(dd['layer'], int)
                assert dd['layer'] == -1 or dd['layer'] >= 0

        if self._hyper_shapes_learned is not None:
            if self._hyper_shapes_learned_ref is None:
                # Note, this attribute was inserted post-hoc.
                # FIXME Warning is annoying, programmers will notice when they
                # use this functionality.
                #warn('Attribute "hyper_shapes_learned_ref" has not been ' +
                #     'implemented!')
                pass
            else:
                assert isinstance(self._hyper_shapes_learned_ref, list)
                for ii in self._hyper_shapes_learned_ref:
                    assert isinstance(ii, int)
                    assert ii == -1 or 0 <= ii < len(self._param_shapes)

        assert(isinstance(self._has_fc_out, bool))
        assert(isinstance(self._mask_fc_out, bool))
        assert(isinstance(self._has_linear_out, bool))

        assert(self._layer_weight_tensors is not None)
        assert(self._layer_bias_vectors is not None)

        # Note, you should overwrite the `has_bias` attribute if you do not
        # follow this requirement.
        if check_has_bias:
            assert isinstance(self._has_bias, bool)
            if self._has_bias:
                assert len(self._layer_weight_tensors) == \
                       len(self._layer_bias_vectors)

if __name__ == '__main__':
    pass



