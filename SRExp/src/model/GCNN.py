import numpy as np
import mindspore
import mindspore.nn as nn
#import math
import mindspore.ops as  P
#import MyLibForSteerCNN as ML
import scipy.io as sio    
import math
from PIL import Image
from math import *

class Fconv_PCA(nn.Cell):

    def __init__(self,  sizeP, inNum, outNum, tranNum=8, inP = None, padding=None, ifIni=0, bias=True, 
                 Smooth = True, iniScale = 1.0, Bscale = 1.0):
       
        super(Fconv_PCA, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_PCA(sizeP,tranNum,inP, Smooth = Smooth)
        self.Basis = mindspore.Parameter(Basis*Bscale, requires_grad=False)       
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        iniw = Getini_reg(Basis.shape[3], inNum, outNum, self.expand, weight)*iniScale
        self.weights = mindspore.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        self.c = mindspore.Parameter(P.Zeros()((1,outNum,1,1), mindspore.float32), requires_grad=bias)

    def construct(self, input):
        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = P.Einsum('ijok,mnak->monaij')((self.Basis, self.weights*1.0))

            Num = tranNum//expand
            tempWList = [tempW[:,0:Num,:,-0:,:,:]]
            tempWList += [P.Concat(3)([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]]) for i in range(1,expand)]
            tempW = P.Concat(1)(tempWList)

            _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP ])
            _bias = mindspore.numpy.tile(self.c, (1,1,tranNum,1)).reshape([1,outNum*tranNum,1,1])

        else:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = P.Einsum('ijok,mnak->monaij')((self.Basis, self.weights*1.0))
            Num = tranNum//expand
            tempWList = [tempW[:,0:Num,:,-0:,:,:]]
            tempWList += [P.Concat(3)([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]]) for i in range(1,expand)]
            tempW = P.Concat(1)(tempWList)
            _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP ])
            _bias = mindspore.numpy.tile(self.c, (1,1,tranNum,1)).reshape([1,outNum*tranNum,1,1])
            self.filter = mindspore.Parameter(_filter, requires_grad=False)
            self.bias = mindspore.Parameter(_bias, requires_grad=False)
            _filter = self.filter
            _bias   = self.bias
        
        conv2d = P.Conv2D(out_channel = _filter.shape[0],
                          kernel_size = (_filter.shape[2], _filter.shape[3]),
                          pad_mode="pad",
                          pad=self.padding,
                          dilation=1,
                          group=1)
        output = conv2d(input, _filter)
        return output + _bias
        
#     def set_train(self, mode=True):
#         if mode:
#             # TODO thoroughly check this is not causing problems
#             if hasattr(self, "filter"):
#                 del self.filter
#                 del self.bias
#         elif self.training:
#             # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
#             tranNum = self.tranNum
#             outNum = self.outNum
#             inNum = self.inNum
#             expand = self.expand
#             tempW = P.Einsum('ijok,mnak->monaij')((self.Basis, self.weights*1.0))
#             Num = tranNum//expand
#             tempWList = [tempW[:,0:Num,:,-0:,:,:]]
#             tempWList += [P.Concat(3)([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]]) for i in range(1,expand)]
#             tempW = P.Concat(1)(tempWList)
#             _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP ])
#             _bias = mindspore.numpy.tile(mindspore.Tensor(self.c), (1,1,tranNum,1)).reshape([1,outNum*tranNum,1,1])
#             self.filter = mindspore.Parameter(_filter, requires_grad=False)
#             self.bias = mindspore.Parameter(_bias, requires_grad=False)

#         return super(Fconv_PCA, self).set_train(mode)       
    

class Fconv_1X1(nn.Cell):
    
    def __init__(self, inNum, outNum, tranNum=8, ifIni=0, bias=True, Smooth = True, iniScale = 1.0):
       
        super(Fconv_1X1, self).__init__()

        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum

                
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        iniw = Getini_reg(1, inNum, outNum, self.expand)*iniScale
        self.weights = mindspore.Parameter(iniw, requires_grad=True)

        self.padding = 0
        self.bias = bias

        if bias:
            self.c = mindspore.Parameter(P.Zeros()((1,outNum,1,1), mindspore.float32), requires_grad=True)
        else:
            self.c = P.Zeros()((1,outNum,1,1), mindspore.float32)

    def construct(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        weights = P.ExpandDims()(self.weights*1.0, 4)
        weights = P.ExpandDims()(weights*1.0, 1)
        tempW = mindspore.numpy.tile(weights, (1,tranNum,1,1,1,1))
        Num = tranNum//expand
        
        tempWList = [tempW[:,0:Num,:,-0:,:,:]]
        tempWList += [P.Concat(3)([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]]) for i in range(1,expand)]
        tempW = P.Concat(1)(tempWList)

        _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, 1, 1 ])

        bias = mindspore.numpy.tile(self.c, (1,1,tranNum,1)).reshape([1,outNum*tranNum,1,1])   

        conv2d = P.Conv2D(out_channel = _filter.shape[0],
                          kernel_size = (_filter.shape[2], _filter.shape[3]),
                          pad_mode="pad",
                          pad=self.padding,
                          dilation=1,
                          group=1)
        output = conv2d(input, _filter)
        return output+bias  
    
class ResBlock(nn.Cell):
    def __init__(
        self, conv, n_feats, kernel_size, tranNum=8, inP = None, 
        bias=True, bn=False, act=nn.ReLU(), res_scale=1,  Smooth = True, iniScale = 1.0):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(kernel_size, n_feats, n_feats, tranNum=tranNum, inP = inP, padding=(kernel_size-1)//2,  bias=bias, Smooth = Smooth, iniScale = iniScale))
            if bn:
                m.append(F_BN(n_feats, tranNum))
            if i == 0:
                m.append(act)

        self.body = nn.SequentialCell(m)
        self.res_scale = res_scale

    def construct(self, x):
        res = P.Mul()(self.body(x), self.res_scale)
        res += x

        return res

def Getini_reg2(nNum, inNum, outNum,expand, weight = 1): 
    A = (np.random.rand(outNum,inNum,expand,nNum)-0.5)*2*2.4495/np.sqrt((inNum)*nNum)*np.expand_dims(np.expand_dims(np.expand_dims(weight, axis = 0),axis = 0),axis = 0)
    return mindspore.Tensor(A, dtype=mindspore.float32)

def Getini_reg(sizeP, inNum, outNum,expand, weight = 1): 
    A = (np.random.rand(outNum,1,inNum,expand,sizeP,sizeP)-0.5)*2*2.4495/np.sqrt((inNum)*sizeP*sizeP)/10
    return mindspore.Tensor(A, dtype=mindspore.float32)

def GetBasis_PDOe(sizeP,tranNum=8,inP = None, Smooth = 1):
    p = tranNum
    partial_dict = [[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,0,0,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,0,1/2,0,0],[0,0,0,0,0],[0,0,-1/2,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,-1/4,0,1/4,0],[0,0,0,0,0],[0,1/4,0,-1/4,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,0,1,0,0],[0,0,-2,0,0],[0,0,1,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,0,0,0,0],[-1/2,1,0,-1,1/2],[0,0,0,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,1/2,-1,1/2,0],[0,0,0,0,0],[0,-1/2,1,-1/2,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,1,0,-1,0],[0,-1/2,0,1/2,0],[0,0,0,0,0]],
					[[0,0,1/2,0,0],[0,0,-1,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,-1/2,0,0]],
					[[0,0,0,0,0],[0,0,0,0,0],[1,-4,6,-4,1],[0,0,0,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[-1/4,1/2,0,-1/2,1/4],[0,0,0,0,0],[1/4,-1/2,0,1/2,-1/4],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,1,-2,1,0],[0,-2,4,-2,0],[0,1,-2,1,0],[0,0,0,0,0]],
					[[0,-1/4,0,1/4,0],[0,1/2,0,-1/2,0],[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,1/4,0,-1/4,0]],
					[[0,0,1,0,0],[0,0,-4,0,0],[0,0,6,0,0],[0,0,-4,0,0],[0,0,1,0,0]]]


    group_angle = [2*k*pi/p+pi/8 for k in range(p)]
    tran_to_partial_coef = [np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
									 [0,cos(x),sin(x),0,0,0,0,0,0,0,0,0,0,0,0],
									 [0,-sin(x),cos(x),0,0,0,0,0,0,0,0,0,0,0,0],
									 [0,0,0,pow(cos(x),2),2*cos(x)*sin(x),pow(sin(x),2),0,0,0,0,0,0,0,0,0],
									 [0,0,0,-cos(x)*sin(x),pow(cos(x),2)-pow(sin(x),2),sin(x)*cos(x),0,0,0,0,0,0,0,0,0],
									 [0,0,0,pow(sin(x),2),-2*cos(x)*sin(x),pow(cos(x),2),0,0,0,0,0,0,0,0,0],
									 [0,0,0,0,0,0,-pow(cos(x),2)*sin(x),pow(cos(x),3)-2*cos(x)*pow(sin(x),2),-pow(sin(x),3)+2*pow(cos(x),2)*sin(x), pow(sin(x),2)*cos(x),0,0,0,0,0],
									 [0,0,0,0,0,0,cos(x)*pow(sin(x),2),-2*pow(cos(x),2)*sin(x)+pow(sin(x),3),pow(cos(x),3)-2*cos(x)*pow(sin(x),2),sin(x)*pow(cos(x),2),0,0,0,0,0],
									 [0,0,0,0,0,0,0,0,0,0,pow(sin(x),2)*pow(cos(x),2),-2*pow(cos(x),3)*sin(x)+2*cos(x)*pow(sin(x),3),pow(cos(x),4)-4*pow(cos(x),2)*pow(sin(x),2)+pow(sin(x),4),-2*cos(x)*pow(sin(x),3)+2*pow(cos(x),3)*sin(x),pow(sin(x),2)*pow(cos(x),2)]]) for x in group_angle]

    partial_dict = mindspore.Tensor(np.array(partial_dict), dtype=mindspore.Float32) # (15,5,5)
    tran_to_partial_coef = mindspore.Tensor(np.array(tran_to_partial_coef), dtype=mindspore.Float32) #(8,9,15) 
    BasisR = P.Einsum('ihw,tci->hwtc')((partial_dict, tran_to_partial_coef))   
    return BasisR/10   

def MaskC(SizeP):
        p = (SizeP-1)/2
        x = np.arange(-p,p+1)/p
        X,Y  = np.meshgrid(x,x)
        C    =X**2+Y**2
        
        Mask = np.ones([SizeP,SizeP])
#        Mask[C>(1+1/(4*p))**2]=0
        Mask = np.exp(-np.maximum(C-1,0)/0.2)
        
        return X, Y, Mask
    
    
class PointwiseAvgPoolAntialiased(nn.Cell):
    
    def __init__(self, sizeF, stride, padding=None ):
        super(PointwiseAvgPoolAntialiased, self).__init__()
        sigma = (sizeF-1)/2/3
        self.kernel_size = (sizeF, sizeF)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        
        if padding is None:
            padding = int((sizeF-1)//2)
            
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        # Build the Gaussian smoothing filter
        grid_x = mindspore.numpy.arange(sizeF).repeat(sizeF).view(sizeF, sizeF)
        grid_y = grid_x.T
        grid  = P.Stack(-1)([grid_x, grid_y])
        mean = (sizeF - 1) / 2.
        variance = sigma ** 2.
        r = -P.ReduceSum()((grid - mean) ** 2., -1)
        _filter = P.Exp()(r / (2 * variance))
        _filter /= P.ReduceSum()(_filter)
        _filter = _filter.view(1, 1, sizeF, sizeF)
        self.filter = mindspore.Parameter(_filter, requires_grad=False)
    
    def construct(self, input):
        _filter = mindspore.numpy.tile(self.filter, (input.shape[1], 1, 1, 1))
        conv2d = P.Conv2D(out_channel = _filter.shape[0],
                          kernel_size = (_filter.shape[2], _filter.shape[3]),
                          pad_mode="pad",
                          pad=0,
                          stride=self.stride,
                          group=input.shape[1])
        output = conv2d(input, _filter)
        return output
        

class F_BN(nn.Cell):
    def __init__(self,channels, tranNum=8):
        super(F_BN, self).__init__()
        self.BN = nn.BatchNorm2d(channels)
        self.tranNum = tranNum
    def construct(self, X):
        X = self.BN(X.reshape([X.shape[0], int(X.shape[1]/self.tranNum), self.tranNum*X.shape[2], X.shape[3]]))
        return X.reshape([X.shape[0], self.tranNum*X.shape[1],int(X.shape[2]/self.tranNum), X.shape[3]])



class F_Dropout(nn.Cell):
    def __init__(self,zero_prob = 0.5,  tranNum=8):
        # nn.Dropout2d
        self.tranNum = tranNum
        super(F_Dropout, self).__init__()
        self.Dropout = nn.Dropout(1-zero_prob)
    def construct(self, X):
        X = self.Dropout(X.reshape([X.shape[0], int(X.shape[1]/self.tranNum), self.tranNum*X.shape[2], X.shape[3]]))
        return X.reshape([X.shape[0], self.tranNum*X.shape[1],int(X.shape[2]/self.tranNum), X.shape[3]])


def build_mask(s, margin=2, dtype=mindspore.float32):
    mask = P.Zeros()((1, 1, s, s), dtype)
    c = (s-1) / 2
    t = (c - margin/100.*c)**2
    sig = 2.
    for x in range(s):
        for y in range(s):
            r = (x - c) ** 2 + (y - c) ** 2
            if r > t:
                mask[..., x, y] = math.exp((t - r)/sig**2)
            else:
                mask[..., x, y] = 1.
    return mask


class MaskModule(nn.Cell):

    def __init__(self, S: int, margin: float = 0.):

        super(MaskModule, self).__init__()

        self.margin = margin
        self.mask = mindspore.Parameter(build_mask(S, margin=margin), requires_grad=False)


    def construct(self, input):

        assert input.shape[2:] == self.mask.shape[2:]

        out = input * self.mask
        return out