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
        eps (float): a perturbated epsilon

    Returns:
        An empty list, if ``task_id`` is ``0``. Otherwise, a list of
        ``task_id-1`` lower, middle and upper targets. These targets can be passed to the function
        :func:`calc_fix_target_reg` while training on the new task.
    """
    # We temporarily switch to eval mode for target computation (e.g., to get
    # rid of training stochasticities such as dropout).
    hnet_mode = hnet.training
    hnet.eval()

    upper_ret  = []
    middle_ret = []
    lower_ret  = []

    with torch.no_grad():
        W_lower, W_middle, W_upper, _ = hnet.forward(cond_id=list(range(task_id)),
                                                ret_format='sequential',
                                                perturbated_eps=eps,
                                                return_extended_output=True
                                                )
        upper_ret  = [[p.detach() for p in W_tid] for W_tid in W_upper]
        middle_ret = [[p.detach() for p in W_tid] for W_tid in W_middle]
        lower_ret  = [[p.detach() for p in W_tid] for W_tid in W_lower]

    hnet.train(mode=hnet_mode)

    return lower_ret, middle_ret, upper_ret

def calc_fix_target_reg(hnet, task_id, eps, lower_targets=None, middle_targets=None, 
                        upper_targets=None, mnet=None,
                        prev_theta=None, prev_task_embs=None):
    r"""This regularizer simply restricts the output-mapping for previous
    task embeddings. I.e., for all :math:`j < \text{task\_id}` minimize:

    .. math::
        \lVert \text{target}_j - h(c_j, \theta + \Delta\theta) \rVert^2

    where :math:`c_j` is the current task embedding for task :math:`j` (and we
    assumed that ``dTheta`` was passed).

    Args:
        hnet: The hypernetwork whose output should be regularized; has to
            implement the interface
            :class:`hnets.hnet_interface.HyperNetInterface`.
        task_id (int): The ID of the current task (the one that is used to
            compute ``dTheta``).
        eps (float): a perturbated epsilon
        targets (list): A list of outputs of the hypernetwork. Each list entry
            must have the output shape as returned by the
            :meth:`hnets.hnet_interface.HyperNetInterface.forward` method of the
            ``hnet``. Note, this function doesn't detach targets. If desired,
            that should be done before calling this function.

            Also see :func:`get_current_targets`.
        mnet: Instance of the main network. Has to be provided if
            ``inds_of_out_heads`` are specified.
        prev_theta (list, optional): If given, ``prev_task_embs`` but not
            ``targets`` has to be specified. ``prev_theta`` is expected to be
            the internal unconditional weights :math:`theta` prior to learning
            the current task. Hence, it can be used to compute the targets on
            the fly (which is more memory efficient (constant memory), but more
            computationally demanding).
            The computed targets will be detached from the computational graph.
            Independent of the current hypernet mode, the targets are computed
            in ``eval`` mode.
        prev_task_embs (list, optional): If given, ``prev_theta`` but not
            ``targets`` has to be specified. ``prev_task_embs`` are the task
            embeddings (conditional parameters) of the hypernetwork.
            See docstring of ``prev_theta`` for more details.
        
    Returns:
        The value of the regularizer.
    """
    assert isinstance(hnet, HyperNetInterface)
    assert task_id > 0
    # FIXME We currently assume the hypernet has all parameters internally.
    # Alternatively, we could allow the parameters to be passed to us, that we
    # will then pass to the forward method.
    assert hnet.unconditional_params is not None and \
        len(hnet.unconditional_params) > 0
    assert middle_targets is None or len(middle_targets) == task_id
    assert mnet is not None
    assert middle_targets is None or (prev_theta is None and prev_task_embs is None)
    assert prev_theta is None or prev_task_embs is not None

    # Number of tasks to be regularized.
    num_regs = task_id
    ids_to_reg = list(range(num_regs))

    # FIXME Assuming all unconditional parameters are internal.
    assert len(hnet.unconditional_params) == \
        len(hnet.unconditional_param_shapes)

    weights = dict()
    uncond_params = hnet.unconditional_params
    weights['uncond_weights'] = uncond_params

    upper_reg  = 0
    middle_reg = 0
    lower_reg  = 0

    for i in ids_to_reg:
    
        curr_embd  = hnet.conditional_params[i]
        curr_radii = eps * F.softmax(
            hnet.perturbated_eps_T[i], dim=-1
        )

        lower_logit = curr_embd - curr_radii
        upper_logit = curr_embd + curr_radii

        lower_weights_predicted = hnet.forward(cond_input=lower_logit.view(1, -1), 
                                               weights=weights, 
                                               common_radii=torch.zeros_like(lower_logit))
        
        middle_weights_predicted = hnet.forward(cond_id=i, weights=weights, perturbated_eps=eps)

        upper_weights_predicted = hnet.forward(cond_input=upper_logit.view(1, -1), 
                                               weights=weights, 
                                               common_radii=torch.zeros_like(upper_logit))

        lower_target  = lower_targets[i]
        middle_target = middle_targets[i]
        upper_target  = upper_targets[i]
    
        # Regularize all weights of the main network.
        lower_W_target = torch.cat([w.view(-1) for w in lower_target])
        lower_W_predicted = torch.cat([w.view(-1) for w in lower_weights_predicted])

        middle_W_target = torch.cat([w.view(-1) for w in middle_target])
        middle_W_predicted = torch.cat([w.view(-1) for w in middle_weights_predicted])

        upper_W_target = torch.cat([w.view(-1) for w in upper_target])
        upper_W_predicted = torch.cat([w.view(-1) for w in upper_weights_predicted])
        
        upper_reg_i = (upper_W_target - upper_W_predicted).pow(2).sum()
        middle_reg_i = (middle_W_target - middle_W_predicted).pow(2).sum()
        lower_reg_i = (lower_W_target - lower_W_predicted).pow(2).sum()

        upper_reg  += upper_reg_i
        middle_reg += middle_reg_i
        lower_reg  += lower_reg_i

    return (upper_reg_i + middle_reg_i + lower_reg_i) / (3*num_regs)


if __name__ == '__main__':
    pass