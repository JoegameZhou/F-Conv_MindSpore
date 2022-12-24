

from e2cnn.nn.modules.r2_conv.basisexpansion import BasisExpansion

from collections import defaultdict

import mindspore
from scipy import stats
import math

__all__ = ["generalized_he_init", "deltaorthonormal_init"]


def generalized_he_init(tensor: mindspore.Tensor, basisexpansion: BasisExpansion):
    r"""
    
    Initialize the weights of a convolutional layer with a generalized He's weight initialization method.
    
    Args:
        tensor (mindspore.Tensor): the tensor containing the weights
        basisexpansion (BasisExpansion): the basis expansion method

    """
    # Initialization
    
    assert tensor.shape == (basisexpansion.dimension(), )
    
    vars = mindspore.ops.OnesLike(tensor)
    
    inputs_count = defaultdict(lambda: set())
    basis_count = defaultdict(int)
    
    for attr in basisexpansion.get_basis_info():
        i, o = attr["in_irreps_position"], attr["out_irreps_position"]
        in_irrep, out_irrep = attr["in_irrep"], attr["out_irrep"]
        inputs_count[o].add(in_irrep)
        basis_count[(in_irrep, o)] += 1
    
    for o in inputs_count.keys():
        inputs_count[o] = len(inputs_count[o])
    
    for w, attr in enumerate(basisexpansion.get_basis_info()):
        i, o = attr["in_irreps_position"], attr["out_irreps_position"]
        in_irrep, out_irrep = attr["in_irrep"], attr["out_irrep"]
        vars[w] = 1. / math.sqrt(inputs_count[o] * basis_count[(in_irrep, o)])
    
    # for i, o in basis_count.keys():
    #     print(i, o, inputs_count[o],  basis_count[(i, o)])
    
    stdnormal = mindspore.ops.StandardNormal()
    tensor[:] = vars * stdnormal(tensor.shape)


def deltaorthonormal_init(tensor: mindspore.Tensor, basisexpansion: BasisExpansion):
    r"""
    
    Initialize the weights of a convolutional layer with *delta-orthogonal* initialization.
    
    Args:
        tensor (mindspore.Tensor): the tensor containing the weights
        basisexpansion (BasisExpansion): the basis expansion method

    """
    # Initialization

    assert tensor.shape == (basisexpansion.dimension(), )
    
    tensor.fill_(0.)
    
    counts = defaultdict(lambda: defaultdict(lambda: []))
    
    for p, attr in enumerate(basisexpansion.get_basis_info()):
        i = attr["in_irrep"]
        o = attr["out_irrep"]
        ip = attr["in_irreps_position"]
        op = attr["out_irreps_position"]
        r = attr["radius"]
        
        if i == o and r == 0.:
            counts[i][(ip, op)].append(p)
    
    def same_content(l):
        l = list(l)
        return all(ll == l[0] for ll in l)
    
    for irrep, count in counts.items():
        assert same_content([len(x) for x in count.values()]), [len(x) for x in count.values()]
        in_c = defaultdict(int)
        out_c = defaultdict(int)
        for ip, op in count.keys():
            in_c[ip] += 1
            out_c[op] += 1
        assert same_content(in_c.values()), count.keys()
        assert same_content(out_c.values()), count.keys()
        
        assert list(in_c.values())[0] == len(out_c.keys())
        assert list(out_c.values())[0] == len(in_c.keys())
        
        s = len(list(count.values())[0])
        i = len(in_c.keys())
        o = len(out_c.keys())
        
        # assert i <= o, (i, o, s, irrep, self._input_size, self._output_size)
        # if i > o:
        #     print("Warning: using delta orthogonal initialization to map to a larger number of channels")
        
        if max(o, i) > 1:
            W = stats.ortho_group.rvs(max(i, o))[:o, :i]
            # W = np.eye(o, i)
        else:
            shape = (1, 1)
            minval = mindspore.Tensor(0, mindspore.int32)
            maxval = mindspore.Tensor(1, mindspore.int32)
            uniform_int = mindspore.ops.UniformInt()
            W = 2 * uniform_int(shape, minval, maxval) - 1
            # W = np.array([[1]])
        
        stdnormal = mindspore.ops.StandardNormal()
        w = stdnormal((o, s))
        # w = mindspore.ops.Ones((o, s))
        w *= 5.
        w /= (w ** 2).sum(dim=1, keepdim=True).sqrt()
        
        for i, ip in enumerate(in_c.keys()):
            for o, op in enumerate(out_c.keys()):
                for p, pp in enumerate(count[(ip, op)]):
                    tensor[pp] = w[o, p] * W[o, i]

