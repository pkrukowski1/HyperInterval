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
# @title           :utils/hnet_regularizer.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :06/05/2019
# @version         :1.0
# @python_version  :3.6.8
"""
Hypernetwork Regularization
---------------------------

We summarize our own regularizers in this module. These regularizer ensure that
the output of a hypernetwork don't change.
"""

import torch
import numpy as np
import torch.nn.functional as F

from hypnettorch.hnets import HyperNetInterface

def get_current_targets(task_id, hnet, eps):
    r"""For all :math:`j < \text{task\_id}`, compute the output of the
    hypernetwork. This output will be detached from the graph before being added
    to the return list of this function.

    Note, if these targets don't change during training, it would be more memory
    efficient to store the weights :math:`\theta^*` of the hypernetwork (which
    is a fixed amount of memory compared to the variable number of tasks).
    Though, it is more computationally expensive to recompute
    :math:`h(c_j, \theta^*)` for all :math:`j < \text{task\_id}` everytime the
    target is needed.

    Note, this function sets the hypernet temporarily in eval mode. No gradients
    are computed.

    See argument ``targets`` of :func:`calc_fix_target_reg` for a use-case of
    this function.

    Args:
        task_id (int): The ID of the current task.
        hnet: An instance of the hypernetwork before learning a new task
            (i.e., the hypernetwork has the weights :math:`\theta^*` necessary
            to compute the targets).
        eps: (float) perturbated epsilon

    Returns:
        An empty tuple of lists, if ``task_id`` is ``0``. Otherwise, a list of
        ``task_id-1`` targets for the lower, middle and upper logits of an interval. 
        These targets can be passed to the function :func:`calc_fix_target_reg` while training on the new task.
    """
    # We temporarily switch to eval mode for target computation (e.g., to get
    # rid of training stochasticities such as dropout).
    hnet_mode = hnet.training
    hnet.eval()

    upper_ret  = []
    middle_ret = []
    lower_ret  = []

    with torch.no_grad():
        lower_weights, middle_weights, upper_weights, _ = hnet.forward(cond_id=list(range(task_id)),
                                                                        ret_format='sequential', 
                                                                        perturbated_eps=eps,
                                                                        return_extended_output=True)
        
        upper_ret  = [[p.detach() for p in W_tid] for W_tid in upper_weights]
        middle_ret = [[p.detach() for p in W_tid] for W_tid in middle_weights]
        lower_ret  = [[p.detach() for p in W_tid] for W_tid in lower_weights]

    hnet.train(mode=hnet_mode)

    return lower_ret, middle_ret, upper_ret

def calc_fix_target_reg(hnet, task_id, perturbated_eps, upper_targets=None,
                        middle_targets=None, lower_targets=None, dTheta=None, dTembs=None,
                        mnet=None, inds_of_out_heads=None,
                        fisher_estimates=None, prev_theta=None,
                        prev_task_embs=None, batch_size=None, reg_scaling=None):
    r"""This regularizer simply restricts the output-mapping for previous
    task embeddings

    Returns:
        The value of the regularizer.
    """
    assert isinstance(hnet, HyperNetInterface)
    assert task_id > 0
    assert hnet.unconditional_params is not None and \
        len(hnet.unconditional_params) > 0
    assert upper_targets is None or len(upper_targets) == task_id
    assert middle_targets is None or len(middle_targets) == task_id
    assert lower_targets is None or len(lower_targets) == task_id
    assert inds_of_out_heads is None or mnet is not None
    assert inds_of_out_heads is None or len(inds_of_out_heads) >= task_id
    assert upper_targets is None or (prev_theta is None and prev_task_embs is None)
    assert middle_targets is None or (prev_theta is None and prev_task_embs is None)
    assert lower_targets is None or (prev_theta is None and prev_task_embs is None)
    assert prev_theta is None or prev_task_embs is not None
    assert dTembs is None or len(dTembs) >= task_id
    assert reg_scaling is None or len(reg_scaling) >= task_id
    assert (upper_targets is not None and middle_targets is not None and lower_targets is not None) or \
            (upper_targets is None and middle_targets is None and lower_targets is None)

    # Number of tasks to be regularized.
    num_regs = task_id
    ids_to_reg = list(range(num_regs))
    if batch_size is not None and batch_size > 0:
        if num_regs > batch_size:
            ids_to_reg = np.random.choice(num_regs, size=batch_size,
                                          replace=False).tolist()
            num_regs = batch_size

    # FIXME Assuming all unconditional parameters are internal.
    assert len(hnet.unconditional_params) == \
        len(hnet.unconditional_param_shapes)
    
    # Sample n points from each interval
    # TODO: Do this for points_to_reg > 1
    # points_to_reg = 1

    radii = torch.stack([
        perturbated_eps*F.softmax(hnet.perturbated_eps_T[i], dim=-1) for i in range(task_id)
    ], dim=0)

    prev_embds = torch.stack([
        radii[i] * torch.tanh(hnet.conditional_params[i]) for i in range(task_id)
    ], dim=0)

    alpha1 = torch.rand_like(radii)
    alpha2 = torch.rand_like(radii)

    zl_prev_embds = prev_embds - alpha1*radii
    zu_prev_embds = prev_embds + alpha2*radii

    weights = dict()
    uncond_params = hnet.unconditional_params
    if dTheta is not None:
        uncond_params = hnet.add_to_uncond_params(dTheta, params=uncond_params)
    weights['uncond_weights'] = uncond_params

    if dTembs is not None:
        # FIXME That's a very unintutive solution for the user. The problem is,
        # that the whole function terminology is based on the old hypernet
        # interface. The new hypernet interface doesn't have the concept of
        # task embedding.
        # The problem is, the hypernet might not just have conditional input
        # embeddings, but also other conditional weights.
        # If it would just be conditional input embeddings, we could just add
        # `dTembs[i]` to the corresponding embedding and use the hypernet
        # forward argument `cond_input`, rather than passing conditional
        # parameters.
        # Here, we now assume all conditional parameters have been passed, which
        # is unrealistic. We leave the problem open for a future implementation
        # of this function.
        assert hnet.conditional_params is not None and \
            len(hnet.conditional_params) == len(hnet.conditional_param_shapes) \
            and len(hnet.conditional_params) == len(dTembs)
        weights['cond_weights'] = hnet.add_to_uncond_params(dTembs,
            params=hnet.conditional_params)

    if middle_targets is None:

        prev_weights = dict()
        prev_weights['uncond_weights'] = prev_theta
        # FIXME We just assume that `prev_task_embs` are all conditional
        # weights.
        prev_weights['cond_weights'] = prev_task_embs

    upper_half_reg = 0
    middle_reg = 0
    lower_half_reg = 0

    zeros = torch.zeros(hnet._cond_in_size)

    for i in ids_to_reg:
        lower_weights_predicted = hnet.forward(cond_input=zl_prev_embds, 
                                               weights=weights,
                                               return_extended_output=False,
                                               perturbated_eps=None,
                                               use_common_embedding=True,
                                               common_radii=zeros)
        
        middle_weights_predicted = hnet.forward(cond_id=i, 
                                               weights=weights,
                                               return_extended_output=False,
                                               perturbated_eps=perturbated_eps)
        
        upper_weights_predicted = hnet.forward(cond_input=zu_prev_embds, 
                                               weights=weights,
                                               return_extended_output=False,
                                               perturbated_eps=None,
                                               use_common_embedding=True,
                                               common_radii=zeros)

        if upper_targets is not None and \
            middle_targets is not None and \
            lower_targets is not None:

            upper_target  = upper_targets[i]
            middle_target = middle_targets[i]
            lower_target  = lower_targets[i]
        else:
            raise Exception("Not implemented yet!")
           
        # Regularize all weights of the main network.
        W_upper_target = torch.cat([w.view(-1) for w in upper_target])
        W_upper_predicted = torch.cat([w.view(-1) for w in upper_weights_predicted])

        W_middle_target = torch.cat([w.view(-1) for w in middle_target])
        W_middle_predicted = torch.cat([w.view(-1) for w in middle_weights_predicted])

        W_lower_target = torch.cat([w.view(-1) for w in lower_target])
        W_lower_predicted = torch.cat([w.view(-1) for w in lower_weights_predicted])

        if fisher_estimates is not None:
            raise Exception("Not implemented!")
        else:
            upper_reg_i  = (W_upper_target - W_upper_predicted).pow(2).sum()
            middle_reg_i = (W_middle_target - W_middle_predicted).pow(2).sum()
            lower_reg_i  = (W_lower_target - W_lower_predicted).pow(2).sum()

        if reg_scaling is not None:
            upper_half_reg += reg_scaling[i] * upper_reg_i
            middle_reg += reg_scaling[i] * middle_reg_i
            lower_half_reg += reg_scaling[i] * lower_reg_i
        else:
            upper_half_reg  += upper_reg_i
            middle_reg += middle_reg_i
            lower_half_reg  += lower_reg_i

    return (upper_half_reg + middle_reg + lower_half_reg) / num_regs

if __name__ == '__main__':
    pass


