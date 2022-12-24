import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as  P
#import MyLibForSteerCNN as ML
# import scipy.io as sio    
import math
from PIL import Image

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

    
class Fconv_PCA_out(nn.Cell):
    
    def __init__(self,  sizeP, inNum, outNum, tranNum=8, inP = None, padding=None, ifIni=0, bias=True, Smooth = True,iniScale = 1.0):
       
        super(Fconv_PCA_out, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_PCA(sizeP,tranNum,inP, Smooth = Smooth) 
        self.Basis = mindspore.Parameter(Basis, requires_grad=False)     
        # self.register_buffer("Basis", Basis)#.cuda())        

        iniw = Getini_reg(Basis.shape[3], inNum, outNum, 1, weight)*iniScale
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
            tempW = P.Einsum('ijok,mnak->monaij')((self.Basis, self.weights*1.0))
            _filter = tempW.reshape([outNum, inNum*tranNum , self.sizeP, self.sizeP ])
        else:
            tranNum = self.tranNum
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            tempW = P.Einsum('ijok,mnak->monaij')((self.Basis, self.weights*1.0))
            _filter = tempW.reshape([outNum, inNum*tranNum , self.sizeP, self.sizeP ])
            self.filter = mindspore.Parameter(_filter, requires_grad=False)
            _filter = self.filter
        _bias = self.c
        conv2d = P.Conv2D(out_channel = _filter.shape[0],
                          kernel_size = (_filter.shape[2], _filter.shape[3]),
                          pad_mode="pad",
                          pad=self.padding,
                          dilation=1,
                          group=1)
        output = conv2d(input, _filter)
        return output + _bias
        
    # def set_train(self, mode=True):
    #     if mode:
    #         # TODO thoroughly check this is not causing problems
    #         if hasattr(self, "filter"):
    #             del self.filter
    #     elif self.training:
    #         # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
    #         tranNum = self.tranNum
    #         tranNum = self.tranNum
    #         outNum = self.outNum
    #         inNum = self.inNum
    #         tempW = P.Einsum('ijok,mnak->monaij')((self.Basis, self.weights*1.0))
    #         _filter = tempW.reshape([outNum, inNum*tranNum , self.sizeP, self.sizeP ])
    #         self.filter = mindspore.Parameter(_filter, requires_grad=False)
    #         # self.register_buffer("filter", _filter)
    #     return super(Fconv_PCA_out, self).set_train(mode)      
    
    

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
    
def Getini_reg(nNum, inNum, outNum,expand, weight = 1): 
    A = (np.random.rand(outNum,inNum,expand,nNum)-0.5)*2*2.4495/np.sqrt((inNum)*nNum)*np.expand_dims(np.expand_dims(np.expand_dims(weight, axis = 0),axis = 0),axis = 0)
    return mindspore.Tensor(A, dtype=mindspore.float32)

  

def GetBasis_PCA(sizeP, tranNum=8, inP=None, Smooth = True):
    if inP==None:
        inP = sizeP
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX,2)
    Y0 = np.expand_dims(inY,2)
    Mask = np.expand_dims(Mask,2)
    theta = np.arange(tranNum)/tranNum*2*np.pi
    theta = np.expand_dims(np.expand_dims(theta,axis=0),axis=0)
#    theta = torch.FloatTensor(theta)
    X = np.cos(theta)*X0-np.sin(theta)*Y0
    Y = np.cos(theta)*Y0+np.sin(theta)*X0
#    X = X.unsqueeze(3).unsqueeze(4)
    X = np.expand_dims(np.expand_dims(X,3),4)
    Y = np.expand_dims(np.expand_dims(Y,3),4)
    v = np.pi/inP*(inP-1)
    p = inP/2
    
    k = np.reshape(np.arange(inP),[1,1,1,inP,1])
    l = np.reshape(np.arange(inP),[1,1,1,1,inP])
    
    
    BasisC = np.cos((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y)
    BasisS = np.sin((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y)
    
    BasisC = np.reshape(BasisC,[sizeP, sizeP, tranNum, inP*inP])*np.expand_dims(Mask,3)
    BasisS = np.reshape(BasisS,[sizeP, sizeP, tranNum, inP*inP])*np.expand_dims(Mask,3)

    BasisC = np.reshape(BasisC,[sizeP*sizeP*tranNum, inP*inP])
    BasisS = np.reshape(BasisS,[sizeP*sizeP*tranNum, inP*inP])

    BasisR = np.concatenate((BasisC, BasisS), axis = 1)
    
    U,S,VT = np.linalg.svd(np.matmul(BasisR.T,BasisR))

    Rank   = np.sum(S>0.0001)
    BasisR = np.matmul(np.matmul(BasisR,U[:,:Rank]),np.diag(1/np.sqrt(S[:Rank]+0.0000000001))) 
    BasisR = np.reshape(BasisR,[sizeP, sizeP, tranNum, Rank])
    
    temp = np.reshape(BasisR, [sizeP*sizeP, tranNum, Rank])
    var = (np.std(np.sum(temp, axis = 0)**2, axis=0)+np.std(np.sum(temp**2*sizeP*sizeP, axis = 0),axis = 0))/np.mean(np.sum(temp, axis = 0)**2+np.sum(temp**2*sizeP*sizeP, axis = 0),axis = 0)
    Trod = 1
    Ind = var<Trod
    Rank = np.sum(Ind)
    Weight = 1/np.maximum(var, 0.04)/25
    if Smooth:
        BasisR = np.expand_dims(np.expand_dims(np.expand_dims(Weight,0),0),0)*BasisR

    return mindspore.Tensor(BasisR, dtype=mindspore.float32), Rank, Weight

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

class GroupPooling(nn.Cell):
    def __init__(self, tranNum):
        super(GroupPooling, self).__init__()
        self.tranNum = tranNum
        
    def construct(self, input):
        
        output = input.reshape([input.shape[0], -1, self.tranNum, input.shape[2], input.shape[3]]) 
        _, output = P.ArgMaxWithValue(2)(output)
        return output
    
    
class GroupMeanPooling(nn.Cell):
    def __init__(self, tranNum):
        super(GroupMeanPooling, self).__init__()
        self.tranNum = tranNum
        
    def construct(self, input):
        
        output = input.reshape([input.shape[0], -1, self.tranNum, input.shape[2], input.shape[3]]) 
        output = P.ReduceMean()(output,2)
        return output
        



class Fconv(nn.Cell):
    
    def __init__(self,  sizeP, inNum, outNum, tranNum=8, inP = None, padding=None, ifIni=0, bias=True):
       
        super(Fconv, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        BasisC, BasisS = GetBasis(sizeP,tranNum,inP)

        self.Basis = mindspore.Parameter(P.Concat(3)((BasisC,BasisS)), requires_grad=False)  

                
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
#        iniw = torch.randn(outNum, inNum, self.expand, self.Basis.size(3))*0.03
        iniw = Getini(inP, inNum, outNum, self.expand)
        self.weights = mindspore.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
           
        if bias:
            self.c = mindspore.Parameter(P.Zeros()((1,outNum,1,1), mindspore.float32), requires_grad=True)
        else:
            self.c = P.Zeros()((1,outNum,1,1), mindspore.float32) 
        self.filter = mindspore.Parameter(P.Zeros()((outNum*tranNum, inNum*self.expand, sizeP, sizeP), mindspore.float32), requires_grad=False)

    def construct(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        tempW = P.Einsum('ijok,mnak->monaij')((self.Basis, self.weights*1.0))
               
        for i in range(expand):
            ind = np.hstack((np.arange(expand-i,expand), np.arange(expand-i) ))
            tempW[:,i,:,:,...] = tempW[:,i,:,ind,...]
        _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP ])
                
#        sio.savemat('Filter2.mat', {'filter': _filter.cpu().detach().numpy()})
        bias = mindspore.numpy.tile(self.c, (1,1,tranNum,1)).reshape([1,outNum*tranNum,1,1])

        conv2d = P.Conv2D(out_channel = _filter.shape[0],
                          kernel_size = (_filter.shape[2], _filter.shape[3]),
                          pad_mode="pad",
                          pad=self.padding,
                          dilation=1,
                          group=1)
        output = conv2d(input, _filter)
        return output + bias
    
class FCNN_reg(nn.Cell):
    
    def __init__(self,  sizeP, inNum, outNum, Basisin, tranNum=8, inP = None, padding=None, ifIni=0, bias=True):
       
        super(FCNN_reg, self).__init__()
#        self.sampled_basis = Basisin.cuda()
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.Basis = mindspore.Parameter(Basisin, requires_grad=False)
        # self.register_buffer("Basis", Basisin.cuda())
                
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        iniw = P.StandardNormal()(outNum, inNum, self.expand, self.Basis.shape[3])*0.03
        self.weights = mindspore.Parameter(iniw, requires_grad=True)
        self.padding = padding

    def construct(self, input):
        

#        _filter = self._filter
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        tempW = P.Einsum('ijok,mnak->monaij')((self.Basis, self.weights*1.0))
               
        for i in range(expand):
            ind = np.hstack((np.arange(expand-i,expand), np.arange(expand-i) ))
            tempW[:,i,:,:,...] = tempW[:,i,:,ind,...]
        _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP ])

        conv2d = P.Conv2D(out_channel = _filter.shape[0],
                          kernel_size = (_filter.shape[2], _filter.shape[3]),
                          pad_mode="pad",
                          pad=self.padding,
                          dilation=1,
                          group=1)
        output = conv2d(input, _filter)
            
        
        return output

 
    
def Getini(sizeP, inNum, outNum, expand):
    
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX,2)
    Y0 = np.expand_dims(inY,2)
    X0 = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(X0,0),0),4),0)
    y  = Y0[:,1]
    y = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(y,0),0),3),0)

    orlW = np.zeros([outNum,inNum,expand,sizeP,sizeP,1,1])
    for i in range(outNum):
        for j in range(inNum):
            for k in range(expand):
                temp = np.array(Image.fromarray(((np.random.randn(3,3))*2.4495/np.sqrt((inNum)*sizeP*sizeP))).resize((sizeP,sizeP)))
                orlW[i,j,k,:,:,0,0] = temp
             
    v = np.pi/sizeP*(sizeP-1)
    k = np.reshape((np.arange(sizeP)),[1,1,1,1,1,sizeP,1])
    l = np.reshape((np.arange(sizeP)),[1,1,1,1,1,sizeP])

    tempA =  np.sum(np.cos(k*v*X0)*orlW,4)/sizeP
    tempB = -np.sum(np.sin(k*v*X0)*orlW,4)/sizeP
    A     =  np.sum(np.cos(l*v*y)*tempA+np.sin(l*v*y)*tempB,3)/sizeP
    B     =  np.sum(np.cos(l*v*y)*tempB-np.sin(l*v*y)*tempA,3)/sizeP 
    A     = np.reshape(A, [outNum,inNum,expand,sizeP*sizeP])
    B     = np.reshape(B, [outNum,inNum,expand,sizeP*sizeP]) 
    iniW  = np.concatenate((A,B), axis = 3)
    return mindspore.Tensor(iniW, dtype=mindspore.float32)

def GetBasis(sizeP,tranNum=8,inP = None):
    if inP==None:
        inP = sizeP
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX,2)
    Y0 = np.expand_dims(inY,2)
    Mask = np.expand_dims(Mask,2)
    theta = np.arange(tranNum)/tranNum*2*np.pi
    theta = np.expand_dims(np.expand_dims(theta,axis=0),axis=0)
#    theta = torch.FloatTensor(theta)
    X = np.cos(theta)*X0-np.sin(theta)*Y0
    Y = np.cos(theta)*Y0+np.sin(theta)*X0
#    X = X.unsqueeze(3).unsqueeze(4)
    X = np.expand_dims(np.expand_dims(X,3),4)
    Y = np.expand_dims(np.expand_dims(Y,3),4)
    v = np.pi/inP*(inP-1)
    p = inP/2
    
    k = np.reshape(np.arange(inP),[1,1,1,inP,1])
    l = np.reshape(np.arange(inP),[1,1,1,1,inP])

    BasisC = np.cos((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y)
    BasisS = np.sin((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y)
    
    BasisC = np.reshape(BasisC,[sizeP, sizeP, tranNum, inP*inP])*np.expand_dims(Mask,3)
    BasisS = np.reshape(BasisS,[sizeP, sizeP, tranNum, inP*inP])*np.expand_dims(Mask,3)
    return mindspore.Tensor(BasisC, dtype=mindspore.float32), mindspore.Tensor(BasisS, dtype=mindspore.float32)   

#SizeP = 7
#p   = 3
#tranNum = 8
#
#conv1 = Fconv(SizeP, 3, 2,  tranNum, ifIni=1, inP=p).cuda()
#conv2 = Fconv(SizeP, 2, 2, tranNum, inP=p).cuda()
#
#X = torch.randn([2,3,29,29]).cuda()
#X = conv1(X)
#print(X.shape)
#X = conv2(X)
#print(X.shape)

#BasisC, BasisS = GetBasis_3_5(tranNum)    
##BasisC, BasisS = GetBasis_plus(SizeP, tranNum, inP=p)  
#iniData = sio.loadmat("HBasis7.mat")
#HBasis5 = torch.FloatTensor(iniData['Basis']).cuda()
#
#
#

##conv1 = FCNN_complex(SizeP, 3, 2, BasisC, BasisS,  tranNum, ifIni=1, inP=p)
##conv2 = FCNN_complex(SizeP, 2, 2, BasisC, BasisS,  tranNum, inP=p)
#conv1 = FCNN_reg(SizeP, 3, 2, HBasis5,  tranNum, ifIni=1, inP=p)
#conv2 = FCNN_reg(SizeP, 2, 2, HBasis5,  tranNum, inP=p)
#
#
#BN1   = F_BN(2, tranNum).cuda()
#Pool1 = PointwiseAvgPoolAntialiased(2,2).cuda()
#X = conv1(X)
#print(X.shape)
#X = conv2(X)
#print(X.shape)
#X = BN1(X)
#print(X.shape)
#X = Pool1(X)
#print(X.shape)
#X = Pool1(X)
#print(X.shape)
##