"""
This file implements neccessary logic for applying interval bound propagation
over tasks' embeddings being inputs to an MLP hypernetwork
"""

from hypnettorch.hnets import HMLP
from hypnettorch.hnets.hnet_interface import HyperNetInterface

import torch
import torch.nn as nn
import torch.nn.functional as F

class HMLP_IBP(HMLP, HyperNetInterface):

    """
    Implementation of a `full hypernet` with interval bound propagation mechanism around tasks' embeddings.

    The network will consist of several hidden layers and a final linear output
    layer that produces all weight matrices/bias-vectors the network has to
    produce.

    The network allows to maintain a set of embeddings internally that can be
    used as conditional input.
    """

    def __init__(self, target_shapes, uncond_in_size=0, cond_in_size=8,
                 layers=(100, 100), verbose=True, activation_fn=torch.nn.ReLU(),
                 use_bias=True, no_uncond_weights=False, no_cond_weights=False,
                 num_cond_embs=1, dropout_rate=-1, use_spectral_norm=False,
                 use_batch_norm=False, embd_dropout_rate=-1, *args, **kwargs):

        HMLP.__init__(self, target_shapes, uncond_in_size=uncond_in_size, cond_in_size=cond_in_size,
                 layers=layers, verbose=verbose, activation_fn=activation_fn,
                 use_bias=use_bias, no_uncond_weights=no_uncond_weights, no_cond_weights=no_cond_weights,
                 num_cond_embs=num_cond_embs, dropout_rate=dropout_rate, use_spectral_norm=use_spectral_norm,
                 use_batch_norm=use_batch_norm)

        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embd_dropout_rate = embd_dropout_rate

        if embd_dropout_rate != -1:
            assert embd_dropout_rate >= 0 and embd_dropout_rate <= 1, "Dropout rate should be contained in the [0,1] interval"
            self.embd_dropout = nn.Dropout(p = embd_dropout_rate)
        
        ### Create epsilons which are summed up to the constant value
        self._perturbated_eps_T = []

        for _ in range(num_cond_embs):
            self._perturbated_eps_T.append(
                F.softmax(torch.ones(self._cond_in_size), dim=-1).to(self._device)
                )
            
        self._is_properly_setup()
    
    @property
    def perturbated_eps_T(self):
        return self._perturbated_eps_T
    
    def detach_tensor(self, idx):
        """
        This method detaches an embedding from the computation graph
        """
        self.conditional_params[idx].requires_grad_(False)
    

    def forward(self, uncond_input=None, cond_input=None, cond_id=None,
                weights=None, distilled_params=None, condition=None,
                ret_format='squeezed', return_extended_output = False,
                perturbated_eps = None, common_emb=False):
        """Compute the weights of a target network.

        Args:
            (....): See docstring of method
                :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
            condition (int, optional): This argument will be passed as argument
                ``stats_id`` to the method
                :meth:`utils.batchnorm_layer.BatchNormLayer.forward` if batch
                normalization is used.
            return_extended_output: (bool) if true, then the function returns target weights,
                                            lower target weight, upper target weights and predicted radii 
                                            of intervals
            perturbated_eps: (float)

        Returns:
            (list or torch.Tensor): See docstring of method
            :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
        """

        uncond_input, cond_input, uncond_weights, _ = \
            self._preprocess_forward_args(uncond_input=uncond_input,
                cond_input=cond_input, cond_id=cond_id, weights=weights,
                distilled_params=distilled_params, condition=condition,
                ret_format=ret_format)

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

        ### Normalize the learnable radii tensor
            
        # cond_id may be a list while a regularization process
        if not isinstance(cond_id, list) and cond_id is not None:
            eps = perturbated_eps * F.softmax(
                    self._perturbated_eps_T[cond_id],
                    dim=-1)
            
        elif isinstance(cond_id, list) and cond_id is not None:
            eps = torch.stack([
                perturbated_eps * F.softmax(self._perturbated_eps_T[i], dim=-1) for i in range(len(cond_id))
            ], dim=0)

        else:
            eps = perturbated_eps * F.softmax(torch.ones_like(h), dim=-1)
        
        eps = eps.to(self._device)

        ### Extract layer weights ###
        bn_scales  = []
        bn_shifts  = []
        fc_weights = []
        fc_biases  = []

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
        
        if self.embd_dropout_rate != -1:
            h = self.embd_dropout(h)
        
        # Apply cos transformation
        sigma = 0.5 * perturbated_eps / self._cond_in_size
        h = h if common_emb else sigma * torch.cos(h)

        for i in range(len(fc_weights)):
            last_layer = i == (len(fc_weights) - 1)

            h = F.linear(h, fc_weights[i], bias=fc_biases[i])
            W = torch.abs(fc_weights[i])
            eps = F.linear(eps, W, bias=torch.zeros_like(fc_biases[i]))

            if not last_layer:

                # Batch-norm
                if self._use_batch_norm:
                   raise Exception("BatchNorm not implemented for hypernets!")

                # Non-linearity
                if self._act_fn is not None:
                    z_l, z_u = h - eps, h + eps
                    z_l, z_u = self._act_fn(z_l), self._act_fn(z_u)
                    h, eps   = (z_u + z_l) / 2, (z_u - z_l) / 2

        z_l, z_u = h-eps, h+eps

        ### Split output into target shapes ###
        ret = self._flat_to_ret_format(h, ret_format)

        if return_extended_output:
            ret_zl = self._flat_to_ret_format(z_l, ret_format)
            ret_zu = self._flat_to_ret_format(z_u, ret_format)
            # radii  = self._flat_to_ret_format(eps, ret_format)  # Calculate epsilon for each weight of a target network
            radii = eps

            return ret_zl, ret, ret_zu, radii
        else:
            return ret
        
       