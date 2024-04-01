from typing import cast
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class IntervalModuleWithWeights(nn.Module, ABC):
    def __init__(self):
        super().__init__()

class IntervalLinear(IntervalModuleWithWeights):
    def __init__(self, in_features: int, out_features: int) -> None:
        nn.Module.__init__()

        self.in_features  = in_features
        self.out_features = out_features
       
    @staticmethod
    def apply_linear( 
                x: Tensor, 
                upper_weights: Tensor,
                middle_weights: Tensor,
                lower_weights: Tensor,
                upper_bias: Tensor,
                middle_bias: Tensor,
                lower_bias: Tensor
                ) -> Tensor:  # type: ignore

        assert (lower_weights <= middle_weights).all(), "Lower bound must be less than or equal to middle bound."
        assert (middle_weights <= upper_weights).all(), "Middle bound must be less than or equal to upper bound."
        assert (lower_bias <= middle_bias).all(), "Lower bias must be less than or equal to middle bias."
        assert (middle_bias <= upper_bias).all(), "Middle bias must be less than or equal to upper bias."

        x = x.refine_names("N", "bounds", "features")  # type: ignore
        assert (x.rename(None) >= 0.0).all(), "All input features must be non-negative."  # type: ignore

        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        assert (x_lower <= x_middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (x_middle <= x_upper).all(), "Middle bound must be less than or equal to upper bound."


        w_lower_pos = lower_weights.clamp(min=0)
        w_lower_neg = lower_weights.clamp(max=0)
        w_upper_pos = upper_weights.clamp(min=0)
        w_upper_neg = upper_weights.clamp(max=0)

        # Further splits only needed for numeric stability with asserts
        w_middle_pos = middle_weights.clamp(min=0)
        w_middle_neg = middle_weights.clamp(max=0)

        lower = x_lower @ w_lower_pos.t() + x_upper @ w_lower_neg.t()
        upper = x_upper @ w_upper_pos.t() + x_lower @ w_upper_neg.t()
        middle = x_middle @ w_middle_pos.t() + x_middle @ w_middle_neg.t()

        b_middle = middle_bias
        b_lower = lower_bias
        b_upper = upper_bias
        lower = lower + b_lower
        upper = upper + b_upper
        middle = middle + b_middle

        assert (lower <= middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (middle <= upper).all(), "Middle bound must be less than or equal to upper bound."

        return torch.stack([lower, middle, upper], dim=1).refine_names("N", "bounds", "features")  # type: ignore
        

class IntervalDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.scale = 1. / (1 - self.p)

    def forward(self, x):
        if self.training:
            x = x.refine_names("N", "bounds", ...)
            x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)),
                                             x.unbind("bounds"))  # type: ignore
            mask = torch.bernoulli(self.p * torch.ones_like(x_middle)).long()
            x_lower = x_lower.where(mask != 1, torch.zeros_like(x_lower)) * self.scale
            x_middle = x_middle.where(mask != 1, torch.zeros_like(x_middle)) * self.scale
            x_upper = x_upper.where(mask != 1, torch.zeros_like(x_upper)) * self.scale

            return torch.stack([x_lower, x_middle, x_upper], dim=1)
        else:
            return x


class IntervalMaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, x):
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        x_lower = super().forward(x_lower)
        x_middle = super().forward(x_middle)
        x_upper = super().forward(x_upper)

        return torch.stack([x_lower, x_middle, x_upper], dim=1).refine_names("N", "bounds", "C", "H",
                                                                             "W")  # type: ignore
    
    @staticmethod
    def apply_max_pool2d(x, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        
        x_lower = F.max_pool2d(x_lower, kernel_size, stride=stride, padding=padding,
                                dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)
        x_middle = F.max_pool2d(x_middle, kernel_size, stride=stride, padding=padding,
                                dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)
        x_upper = F.max_pool2d(x_upper, kernel_size, stride=stride, padding=padding,
                                dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)

        return torch.stack([x_lower, x_middle, x_upper], dim=1).refine_names("N", "bounds", "C", "H",
                                                                             "W")  # type: ignore


class IntervalAvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None):
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    def forward(self, x):
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        x_lower = super().forward(x_lower)
        x_middle = super().forward(x_middle)
        x_upper = super().forward(x_upper)
        return torch.stack([x_lower, x_middle, x_upper], dim=1).refine_names("N", "bounds", "C", "H",
                                                                             "W")  # type: ignore
    
    @staticmethod
    def apply_avg_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None):
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        
        x_lower = F.avg_pool2d(x_lower, kernel_size, stride=stride, padding=padding,
                                ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)
        x_middle = F.avg_pool2d(x_middle, kernel_size, stride=stride, padding=padding,
                                ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)
        x_upper = F.avg_pool2d(x_upper, kernel_size, stride=stride, padding=padding,
                                ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)

        return torch.stack([x_lower, x_middle, x_upper], dim=1).refine_names("N", "bounds", "C", "H",
                                                                             "W")  # type: ignore
    

class IntervalBatchNorm2d(IntervalModuleWithWeights):
    def __init__(self, num_features,
                 upper_gamma: Tensor = None,
                 middle_gamma: Tensor = None,
                 lower_gamma: Tensor = None,
                 upper_beta: Tensor = None,
                 middle_beta: Tensor = None,
                 lower_beta: Tensor = None,
                 interval_statistics: bool = False,
                 affine: bool = True,
                 momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.interval_statistics = interval_statistics
        self.affine = affine
        self.momentum = momentum
        self._param_shapes = [[num_features], [num_features]]

        self.epsilon = 1e-5
        if self.affine:
            self.upper_beta  = upper_beta
            self.middle_beta = middle_beta
            self.lower_beta  = lower_beta

            self.upper_gamma  = upper_gamma
            self.middle_gamma = middle_gamma
            self.lower_gamma  = lower_gamma

        self.register_buffer('running_mean', torch.zeros(num_features, requires_grad=False))
        self.register_buffer('running_var', torch.ones(num_features, requires_grad=False))
    
    @property
    def param_shapes(self):
        return self._param_shapes


    def forward(self, x, upper_gamma, middle_gamma, lower_gamma, upper_beta, middle_beta, lower_beta):

        x = x.refine_names("N", "bounds", "C", "H", "W")  # type: ignore
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore

        if self.affine:
            self.upper_beta  = upper_beta
            self.middle_beta = middle_beta
            self.lower_beta  = lower_beta

            self.upper_gamma  = upper_gamma
            self.middle_gamma = middle_gamma
            self.lower_gamma  = lower_gamma

        if self.interval_statistics and self.training:
            # Calculating whitening nominator: x - E[x]
            mean_lower = x_lower.mean([0, 2, 3], keepdim=True)
            mean_middle = x_middle.mean([0, 2, 3], keepdim=True)
            mean_upper = x_upper.mean([0, 2, 3], keepdim=True)

            nominator_upper = x_upper - mean_lower
            nominator_middle = x_middle - mean_middle
            nominator_lower = x_lower - mean_upper

            # Calculating denominator: sqrt(Var[x] + eps)
            # Var(x) = E[x^2] - E[x]^2
            mean_squared_lower = torch.where(
                torch.logical_and(x_lower <= 0, 0 <= x_upper),
                torch.zeros_like(x_middle),
                torch.minimum(x_upper ** 2, x_lower ** 2)).mean([0, 2, 3], keepdim=True)
            mean_squared_middle = (x_middle ** 2).mean([0, 2, 3], keepdim=True)
            mean_squared_upper = torch.maximum(x_upper ** 2, x_lower ** 2).mean([0, 2, 3], keepdim=True)

            squared_mean_lower = torch.where(
                torch.logical_and(mean_lower <= 0, 0 <= mean_upper),
                torch.zeros_like(mean_middle),
                torch.minimum(mean_lower ** 2, mean_upper ** 2))
            squared_mean_middle = mean_middle ** 2
            squared_mean_upper = torch.maximum(mean_lower ** 2, mean_upper ** 2)

            var_lower = mean_squared_lower - squared_mean_upper
            var_middle = mean_squared_middle - squared_mean_middle
            var_upper = mean_squared_upper - squared_mean_lower

            assert torch.all(var_lower <= var_middle)
            assert torch.all(var_middle <= var_upper)

            var_lower = torch.clamp(var_lower, min=0)
            assert torch.all(var_lower >= 0.), "Variance has to be non-negative"
            assert torch.all(var_middle >= 0.), "Variance has to be non-negative"
            assert torch.all(var_upper >= 0.), "Variance has to be non-negative"

            denominator_lower = (var_lower + self.epsilon).sqrt()
            denominator_middle = (var_middle + self.epsilon).sqrt()
            denominator_upper = (var_upper + self.epsilon).sqrt()

            # Dividing nominator by denominator
            whitened_lower = torch.where(nominator_lower > 0,
                                         nominator_lower / denominator_upper,
                                         nominator_lower / denominator_lower)
            whitened_middle = nominator_middle / denominator_middle
            whitened_upper = torch.where(nominator_upper > 0,
                                         nominator_upper / denominator_lower,
                                         nominator_upper / denominator_upper)

        else:

            if self.training:
                mean_middle = x_middle.mean([0, 2, 3], keepdim=True)
                var_middle = x_middle.var([0, 2, 3], keepdim=True)
            else:
                mean_middle = self.running_mean.view(1, -1, 1, 1)
                var_middle = self.running_var.view(1, -1, 1, 1)

            nominator_lower = x_lower - mean_middle
            nominator_middle = x_middle - mean_middle
            nominator_upper = x_upper - mean_middle

            denominator = (var_middle + self.epsilon).sqrt()

            whitened_lower = nominator_lower / denominator
            whitened_middle = nominator_middle / denominator
            whitened_upper = nominator_upper / denominator

        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_middle.view(-1).detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_middle.view(-1).detach()

        assert (whitened_lower <= whitened_middle).all()
        assert (whitened_middle <= whitened_upper).all()

        if self.affine:

            gamma_middle = self.middle_gamma
            gamma_lower = self.lower_gamma
            gamma_upper = self.upper_gamma

            beta_middle = self.middle_beta
            beta_lower = self.lower_beta
            beta_upper = self.upper_beta

            gamma_lower = gamma_lower.view(1, -1, 1, 1)
            gamma_middle = gamma_middle.view(1, -1, 1, 1)
            gamma_upper = gamma_upper.view(1, -1, 1, 1)

            gammafied_all = torch.stack([
                gamma_lower * whitened_lower,
                gamma_lower * whitened_upper,
                gamma_upper * whitened_lower,
                gamma_upper * whitened_upper], dim=-1)
            gammafied_lower, _ = gammafied_all.min(-1)
            gammafied_middle = gamma_middle * whitened_middle
            gammafied_upper, _ = gammafied_all.max(-1)

            beta_lower = beta_lower.view(1, -1, 1, 1)
            beta_middle = beta_middle.view(1, -1, 1, 1)
            beta_upper = beta_upper.view(1, -1, 1, 1)

            final_lower = gammafied_lower + beta_lower
            final_middle = gammafied_middle + beta_middle
            final_upper = gammafied_upper + beta_upper

            assert (final_lower <= final_middle).all()
            assert (final_middle <= final_upper).all()

            return torch.stack([final_lower, final_middle, final_upper], dim=1).refine_names("N", "bounds", "C", "H",
                                                                                             "W")
        else:
            return torch.stack([whitened_lower, whitened_middle, whitened_upper], dim=1).refine_names("N", "bounds",
                                                                                                      "C", "H", "W")


class IntervalAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size):
        super().__init__(output_size)

    def forward(self, x):
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        x_lower = super().forward(x_lower)
        x_middle = super().forward(x_middle)
        x_upper = super().forward(x_upper)
        return torch.stack([x_lower, x_middle, x_upper], dim=1).refine_names("N", "bounds", "C", "H",
                                                                             "W")  # type: ignore


class IntervalConv2d(nn.Conv2d, IntervalModuleWithWeights):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            lower_weights: Tensor,
            middle_weights: Tensor,
            upper_weights: Tensor,
            lower_bias: Tensor,
            middle_bias: Tensor,
            upper_bias: Tensor,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True
    ) -> None:
        IntervalModuleWithWeights.__init__(self)
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.lower_weights  = lower_weights
        self.middle_weights = middle_weights
        self.upper_weights  = upper_weights

        self.lower_bias  = lower_bias
        self.middle_bias = middle_bias
        self.upper_bias  = upper_bias
    
    @staticmethod
    def apply_conv2d(x: Tensor,
                    lower_weights: Tensor,
                    middle_weights: Tensor,
                    upper_weights: Tensor,
                    lower_bias: Tensor,
                    middle_bias: Tensor,
                    upper_bias: Tensor,
                    stride: int = 1,
                    padding: int = 0,
                    dilation: int = 1,
                    groups: int = 1,
                    bias: bool = True) -> Tensor:  # type: ignore
        x = x.refine_names("N", "bounds", "C", "H", "W")
        assert (x.rename(None) >= 0.0).all(), "All input features must be non-negative."  # type: ignore
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        assert (x_lower <= x_middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (x_middle <= x_upper).all(), "Middle bound must be less than or equal to upper bound."

      
        w_middle: Tensor = middle_weights
        w_lower  = lower_weights
        w_upper  = upper_weights
        b_middle = middle_bias
        b_lower  = lower_bias
        b_upper  = upper_bias

        w_lower_pos = w_lower.clamp(min=0)
        w_lower_neg = w_lower.clamp(max=0)
        w_upper_pos = w_upper.clamp(min=0)
        w_upper_neg = w_upper.clamp(max=0)

        # Further splits only needed for numeric stability with asserts
        w_middle_neg = w_middle.clamp(max=0)
        w_middle_pos = w_middle.clamp(min=0)

        l_lp = F.conv2d(x_lower, w_lower_pos, None, stride, padding, dilation, groups)
        u_ln = F.conv2d(x_upper, w_lower_neg, None, stride, padding, dilation, groups)
        u_up = F.conv2d(x_upper, w_upper_pos, None, stride, padding, dilation, groups)
        l_un = F.conv2d(x_lower, w_upper_neg, None, stride, padding, dilation, groups)
        m_mp = F.conv2d(x_middle, w_middle_pos, None, stride, padding, dilation, groups)
        m_mn = F.conv2d(x_middle, w_middle_neg, None, stride, padding, dilation, groups)

        lower = l_lp + u_ln
        upper = u_up + l_un
        middle = m_mp + m_mn

        if bias is not None:
            lower = lower + b_lower.view(1, b_lower.size(0), 1, 1)
            upper = upper + b_upper.view(1, b_upper.size(0), 1, 1)
            middle = middle + b_middle.view(1, b_middle.size(0), 1, 1)

        assert (lower <= middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (middle <= upper).all(), "Middle bound must be less than or equal to upper bound."

        # Safety net for rare numerical errors.
        if not (lower <= middle).all():
            diff = torch.where(lower > middle, lower - middle, torch.zeros_like(middle)).abs().sum()
            print(f"Lower bound must be less than or equal to middle bound. Diff: {diff}")
            lower = torch.where(lower > middle, middle, lower)
        if not (middle <= upper).all():
            diff = torch.where(middle > upper, middle - upper, torch.zeros_like(middle)).abs().sum()
            print(f"Middle bound must be less than or equal to upper bound. Diff: {diff}")
            upper = torch.where(middle > upper, middle, upper)

        return torch.stack([lower, middle, upper], dim=1).refine_names("N", "bounds", "C", "H", "W")  # type: ignore
    