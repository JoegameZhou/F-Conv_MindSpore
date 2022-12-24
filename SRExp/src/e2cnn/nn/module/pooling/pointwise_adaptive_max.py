

from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import constexpr

from typing import List, Tuple, Any, Union

@constexpr
def compute_kernel_size(inp_shape, output_size):
    kernel_width, kernel_height = inp_shape[2], inp_shape[3]
    if isinstance(output_size, int):
        kernel_width = math.ceil(kernel_width / output_size) 
        kernel_height = math.ceil(kernel_height / output_size)
    elif isinstance(output_size, list) or isinstance(output_size, tuple):
        kernel_width = math.ceil(kernel_width / output_size[0]) 
        kernel_height = math.ceil(kernel_height / output_size[1])
    return (kernel_width, kernel_height)

__all__ = ["PointwiseAdaptiveMaxPool"]


class PointwiseAdaptiveMaxPool(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 output_size: Union[int, Tuple[int, int]]
                 ):
        r"""

        Module that implements adaptive channel-wise max-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.AdaptiveMaxPool2D`, wrapping it in the
        :class:`~e2cnn.nn.EquivariantModule` interface.
        
        Notice that not all representations support this kind of pooling. In general, only representations which support
        pointwise non-linearities do.

        Args:
            in_type (FieldType): the input field type
            output_size: the target output size of the image of the form H x W

        """

        assert isinstance(in_type.gspace, GeneralOnR2)
        
        for r in in_type.representations:
            assert 'pointwise' in r.supported_nonlinearities, \
                f"""Error! Representation "{r}" does not support pointwise non-linearities
                so it is not possible to pool each channel independently"""
        
        super(PointwiseAdaptiveMaxPool, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def construct(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert input.type == self.in_type
        
        # run the common max-pooling
        inp_shape = input.tensor.shape
        kernel_size = compute_kernel_size(inp_shape, self.output_size)
        output = mindspore.ops.MaxPool(kernel_size, kernel_size)(input.tensor)
                
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, c, hi, wi = input_shape

        return b, self.out_type.size, self.output_size, self.output_size

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
    
        # this kind of pooling is not really equivariant so we can not test equivariance
        pass

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.AdaptiveAvgPool2d` module and set to "eval" mode.

        """
    
        self.eval()

        inp_shape = input.tensor.shape
        kernel_size = compute_kernel_size(inp_shape, self.output_size)
    
        return mindspore.ops.MaxPool(kernel_size, kernel_size)().eval()
