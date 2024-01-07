"""
This file implements neccessary logic for applying interval bound propagation
over tasks' embeddings being inputs to an MLP hypernetwork
"""

from hypnettorch.hnets import HMLP
from hypnettorch.hnets.hnet_interface import HyperNetInterface

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

class HMLP_IBP(HMLP, HyperNetInterface):

    """
    Implementation of a `full hypernet` with interval bound propagation mechanism around tasks' embeddings.

    The network will consist of several hidden layers and a final linear output
    layer that produces all weight matrices/bias-vectors the network has to
    produce.

    The network allows to maintain a set of embeddings internally that can be
    used as conditional input.
    """

    def __init__(self, 
                target_shapes, 
                cond_in_size, 
                num_cond_embs=1,
                *args, 
                **kwargs): 
        
        super(HMLP_IBP, self).__init__(target_shapes=target_shapes,
                        cond_in_size=cond_in_size,
                        activation_fn=nn.ReLU(),       # For now only ReLU is supported
                        num_cond_embs=num_cond_embs) 
        
        self.trained_radii = torch.zeros(cond_in_size)    # Initialize an empty tensor for intervals
                                                          # around embeddings in the weight space
        self.tasks_embeddings = torch.zeros(cond_in_size) # This variable stores the learned embedding
        self.perturbated_eps = kwargs["perturbated_eps"]

    def forward(self, uncond_input=None, cond_input=None, cond_id=None,
                weights=None, distilled_params=None, condition=None,
                ret_format='squeezed', return_extended_output = False,
                perturbated_eps = None):
        """Compute the weights of a target network.

        Args:
            (....): See docstring of method
                :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
            condition (int, optional): This argument will be passed as argument
                ``stats_id`` to the method
                :meth:`utils.batchnorm_layer.BatchNormLayer.forward` if batch
                normalization is used.
            return_extended_output (bool): if true, then the function returns target weights,
                                            lower target weight, upper target weights and predicted radii 
                                            of intervals 

        Returns:
            (list or torch.Tensor): See docstring of method
            :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
        """
        uncond_input, cond_input, uncond_weights, _ = \
            self._preprocess_forward_args(uncond_input=uncond_input,
                cond_input=cond_input, cond_id=cond_id, weights=weights,
                distilled_params=distilled_params, condition=condition,
                ret_format=ret_format)
        
        perturbated_eps = self.perturbated_eps if perturbated_eps is None else perturbated_eps

        ### Prepare hypernet input ###
        assert self._uncond_in_size == 0 or uncond_input is not None
        assert self._cond_in_size == 0 or cond_input is not None
        if uncond_input is not None:
            assert len(uncond_input.shape) == 2 and \
                   uncond_input.shape[1] == self._uncond_in_size
            h = uncond_input
        if cond_input is not None:
            assert len(cond_input.shape) == 2 and \
                   cond_input.shape[1] == self._cond_in_size
            h = cond_input
        if uncond_input is not None and cond_input is not None:
            h = torch.cat([uncond_input, cond_input], dim=1)

        ### Extract layer weights ###
        bn_scales = []
        bn_shifts = []
        fc_weights = []
        fc_biases = []

        assert len(uncond_weights) == len(self.unconditional_param_shapes_ref)
        for i, idx in enumerate(self.unconditional_param_shapes_ref):
            meta = self.param_shapes_meta[idx]

            if meta['name'] == 'bn_scale':
                bn_scales.append(uncond_weights[i])
            elif meta['name'] == 'bn_shift':
                bn_shifts.append(uncond_weights[i])
            elif meta['name'] == 'weight':
                fc_weights.append(uncond_weights[i])
            else:
                assert meta['name'] == 'bias'
                fc_biases.append(uncond_weights[i])

        if not self.has_bias:
            assert len(fc_biases) == 0
            fc_biases = [None] * len(fc_weights)

        if self._use_batch_norm:
            assert len(bn_scales) == len(fc_weights) - 1

        ### Process inputs through the network ###
            
        # Normalization step - we give to the neural net a chance to
        # decide about length of interval around each dimension of
        # embedding
        eps = perturbated_eps*F.softmax(h, dim=1)

        # Store the trained radii
        self.trained_radii = deepcopy(eps.detach())

        for i in range(len(fc_weights)):
            last_layer = i == (len(fc_weights) - 1)

            h = F.linear(h, fc_weights[i], bias=fc_biases[i])

            # Calculate adaptive weights
            W   = torch.abs(fc_weights[i])
            eps = F.linear(eps, W, bias=torch.zeros_like(fc_biases[i]))

            if not last_layer:
                # Batch-norm
                if self._use_batch_norm:
                    h = self.batchnorm_layers[i].forward(h, running_mean=None,
                        running_var=None, weight=bn_scales[i],
                        bias=bn_shifts[i], stats_id=condition)
                
                # Dropout
                if self._dropout_rate != -1:
                    h = self._dropout(h)

                # Non-linearity
                if self._act_fn is not None:
                    z_l, z_u = h - eps, h + eps
                    z_l, z_u = self._act_fn(z_l), self._act_fn(z_u)
                    h, eps   = (z_u + z_l) / 2, (z_u - z_l) / 2

        z_l, z_u = h-eps, h+eps
        
        # Store the embedding
        self.tasks_embeddings = deepcopy(h.detach())

        ### Split output into target shapes ###
        ret = self._flat_to_ret_format(h, ret_format)
        if return_extended_output:
            ret_zl = self._flat_to_ret_format(z_l, ret_format)
            ret_zu = self._flat_to_ret_format(z_u, ret_format)

            return ret, ret_zl, ret_zu, eps
        else:
            return ret
        
       