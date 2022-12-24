
from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import mindspore

from typing import List, Tuple, Any

import numpy as np

__all__ = ["ELU"]


class ELU(EquivariantModule):
    
    def __init__(self, in_type: FieldType, alpha: float = 1.0, inplace: bool = False):
        r"""
        
        Module that implements a pointwise ELU to every channel independently.
        The input representation is preserved by this operation and, therefore, it equals the output
        representation.
        
        Only representations supporting pointwise non-linearities are accepted as input field type.
        
        Args:
            in_type (FieldType):  the input field type
            alpha (float): the :math:`\alpha` value for the ELU formulation. Default: 1.0
            inplace (bool, optional): can optionally do the operation in-place. Default: ``False``
            
        """
        
        assert isinstance(in_type.gspace, GeneralOnR2)
        
        super(ELU, self).__init__()
        
        for r in in_type.representations:
            assert 'pointwise' in r.supported_nonlinearities, \
                'Error! Representation "{}" does not support "pointwise" non-linearity'.format(r.name)
        
        self.space = in_type.gspace
        self.in_type = in_type
        self.alpha = alpha
        
        # the representation in input is preserved
        self.out_type = in_type
        
        self._inplace = inplace
    
    def construct(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Applies ELU function on the input fields

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map after elu has been applied

        """
        
        assert input.type == self.in_type
        elu = mindspore.ops.Elu(alpha=self.alpha)
        return GeometricTensor(elu(input.tensor), self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        
        b, c, hi, wi = input_shape
        
        return b, self.out_type.size, hi, wi
    
    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        
        c = self.in_type.size
        
        stdnormal = mindspore.ops.StandardNormal()
        x = stdnormal((3, c, 10, 10))
        
        x = GeometricTensor(x, self.in_type)
        
        errors = []
        
        for el in self.space.testing_elements:
            out1 = self(x).transform_fibers(el)
            out2 = self(x.transform_fibers(el))
            
            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
            
            tensor_to_numpy1 = out1.tensor.asnumpy()
            tensor_to_numpy2 = out2.tensor.asnumpy()
            assert np.allclose(tensor_to_numpy1, tensor_to_numpy2, atol=atol, rtol=rtol), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())
            
            errors.append((el, errs.mean()))
        
        return errors

    def extra_repr(self):
        return 'alpha={}, inplace={}, type={}'.format(
            self.alpha, self._inplace, self.in_type
        )

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.ELU` module and set to "eval" mode.

        """
    
        self.eval()
    
        return mindspore.nn.ELU(alpha=self.alpha)

