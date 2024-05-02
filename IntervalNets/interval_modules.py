from typing import cast
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from warnings import warn

def parse_logits(x):
    """
    Parse the output of a target network to get lower, middle and upper predictions

    Arguments:
    ----------
        *x*: (torch.Tensor) the output to be parsed
    
    Returns:
    --------
        a tuple of lower, middle and upper predictions
    """

    return map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore

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

        if bias is not None and b_lower is not None and \
            b_middle is not None and \
            b_upper is not None:
            
            lower = lower + b_lower.view(1, b_lower.size(0), 1, 1)
            upper = upper + b_upper.view(1, b_upper.size(0), 1, 1)
            middle = middle + b_middle.view(1, b_middle.size(0), 1, 1)

        # Safety net for rare numerical errors.
        if not (lower <= middle).all():
            diff = torch.where(lower > middle, lower - middle, torch.zeros_like(middle)).abs().sum()
            print(f"Lower bound must be less than or equal to middle bound. Diff: {diff}")
            lower = torch.where(lower > middle, middle, lower)
        if not (middle <= upper).all():
            diff = torch.where(middle > upper, middle - upper, torch.zeros_like(middle)).abs().sum()
            print(f"Middle bound must be less than or equal to upper bound. Diff: {diff}")
            upper = torch.where(middle > upper, middle, upper)

        assert (lower <= middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (middle <= upper).all(), "Middle bound must be less than or equal to upper bound."

        return torch.stack([lower, middle, upper], dim=1).refine_names("N", "bounds", "C", "H", "W")  # type: ignore
    