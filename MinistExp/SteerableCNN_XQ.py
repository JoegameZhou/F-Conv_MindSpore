import mindspore
import F_Conv as fn
import mindspore.nn as nn
import MyLib as ML
import numpy as np

class MaxPool2d(nn.Cell):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.max_pool = mindspore.ops.MaxPool(kernel_size, stride)
        self.use_pad = padding != 0
        if isinstance(padding, tuple):
            assert len(padding) == 2
            paddings = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
        elif isinstance(padding, int):
            paddings = ((0, 0),) * 2 + ((padding, padding),) * 2
        else:
            raise ValueError('padding should be a tuple include 2 numbers or a int number')
        self.pad = mindspore.ops.Pad(paddings)
    
    def construct(self, x):
        if self.use_pad:
            x = self.pad(x)
        return self.max_pool(x)
         
class MinstSteerableCNN(nn.Cell):
    
    def __init__(self, n_classes=10, tranNum = 8):
        
        super(MinstSteerableCNN, self).__init__()

        self.tranNum = tranNum
        inP = 4
        zero_prob = 0.25
        
        self.block1 = nn.SequentialCell([
            fn.Fconv_PCA(7,1,16,tranNum,inP=5,padding=1, ifIni=1, bias=False),
            fn.F_BN(16,tranNum),
            nn.ReLU()
        ])
        
        self.block2 = nn.SequentialCell([
            fn.Fconv_PCA(5,16,16,tranNum,inP,2,bias=False),
            fn.F_BN(16,tranNum),
            nn.ReLU(),fn.F_Dropout(zero_prob,tranNum)
        ])

        self.block3 = nn.SequentialCell([
            fn.Fconv_PCA(5,16,32,tranNum,inP,2,bias=False),
            fn.F_BN(32,tranNum),
            nn.ReLU(),fn.F_Dropout(zero_prob,tranNum)
        ])
        
        self.block4 = nn.SequentialCell([
            fn.Fconv_PCA(5,32,32,tranNum,inP,2,bias=False),
            fn.F_BN(32,tranNum),
            nn.ReLU(),fn.F_Dropout(zero_prob,tranNum)
        ])

        self.block5 = nn.SequentialCell([
            fn.Fconv_PCA(5,32,32,tranNum,inP,2,bias=False),
            fn.F_BN(32,tranNum),
            nn.ReLU(),fn.F_Dropout(zero_prob,tranNum)
        ]) 
        
        self.block6 = nn.SequentialCell([
            fn.Fconv_PCA(5,32,64,tranNum,inP,2,bias=False),
            fn.F_BN(64,tranNum),
            nn.ReLU(),fn.F_Dropout(zero_prob,tranNum)
        ])
        
        self.block7 = nn.SequentialCell([
            fn.Fconv_PCA(5,64,96,tranNum,inP,1,bias=False),
            fn.F_BN(96,tranNum),
            nn.ReLU(),fn.F_Dropout(zero_prob,tranNum)
        ])
        
        self.pool1 = MaxPool2d(2,2,1)
        self.pool2 = MaxPool2d(2,2,1)        
        self.pool3 = fn.PointwiseAvgPoolAntialiased(5, 1, padding=0)
        self.gpool = fn.GroupPooling(tranNum)
        
        self.fully_net = nn.SequentialCell([
            nn.Dense(96, 96),
            nn.BatchNorm1d(num_features=96),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Dense(96, n_classes),
        ])
    def construct(self, input: mindspore.Tensor, ifshow=0):
        x = self.block1(input)
        x1 = self.block2(x)
        x = self.pool1(x1)
        x = self.block3(x)
        x3 = self.block4(x)
        x = self.pool2(x3)
        x = self.block5(x)
        x2 = self.block6(x)
        x = self.block7(x2)
        
        x = self.pool3(x)
        x = self.gpool(x)


        
        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        if ifshow:
            
            I = mindspore.ops.Transpose()(x1[0,:,:,:], (1,2,0))
            sizeX = I.shape[0]
            I1 = mindspore.ops.Transpose()(I.reshape(sizeX,sizeX, -1, self.tranNum)[:,:,0:3,:], (0,3,1,2))
            I1 = I1.reshape(sizeX, sizeX*self.tranNum,3)
            I2 = mindspore.ops.Transpose()(I.reshape(sizeX,sizeX, self.tranNum, -1)[:,:,:,0:3], (0,2,1,3))
            I2 = I2.reshape(sizeX, sizeX*self.tranNum,3)
            ML.imshow(np.vstack((I1.asnumpy(),I2.asnumpy())))
            
            
            I = mindspore.ops.Transpose()(x3[0,:,:,:], (1,2,0))
            sizeX = I.shape[0]
            I1 = mindspore.ops.Transpose()(I.reshape(sizeX,sizeX, -1, self.tranNum)[:,:,0:3,:], (0,3,1,2))
            I1 = I1.reshape(sizeX, sizeX*self.tranNum,3)
            I2 = mindspore.ops.Transpose()(I.reshape(sizeX,sizeX, self.tranNum, -1)[:,:,:,0:3], (0,2,1,3))
            I2 = I2.reshape(sizeX, sizeX*self.tranNum,3)
            ML.imshow(np.vstack((I1.asnumpy(),I2.asnumpy())))
            
            
            I = mindspore.ops.Transpose()(x2[0,:,:,:], (1,2,0))
            sizeX = I.shape[0]
            I1 = mindspore.ops.Transpose()(I.reshape(sizeX,sizeX, -1, self.tranNum)[:,:,0:3,:], (0,3,1,2))
            I1 = I1.reshape(sizeX, sizeX*self.tranNum,3)
            I2 = mindspore.ops.Transpose()(I.reshape(sizeX,sizeX, self.tranNum, -1)[:,:,:,0:3], (0,2,1,3))
            I2 = I2.reshape(sizeX, sizeX*self.tranNum,3)
            ML.imshow(np.vstack((I1.asnumpy(),I2.asnumpy())))
            
            
            pass
        return x
    
#    def load_state_dict(self, state_dict, strict=False):
#        own_state = self.state_dict()
#        for name, param in state_dict.items():
#            if name in own_state:
#                if isinstance(param, nn.Parameter):
#                    param = param.data
#                try:
#                    own_state[name].copy_(param)
#                except Exception:
#                    raise RuntimeError('While copying the parameter named {}, '
#                                           'whose dimensions in the model are {} and '
#                                           'whose dimensions in the checkpoint are {}.'
#                                           .format(name, own_state[name].size(), param.size()))
#            elif strict:
#                if name.find('tail') == -1:
#                    raise KeyError('unexpected key "{}" in state_dict'
#                                   .format(name))
#
#        if strict:
#            missing = set(own_state.keys()) - set(state_dict.keys())
#            if len(missing) > 0:
#                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
    
    
    
class MinstSteerableCNN_simple(nn.Cell):
    
    def __init__(self, n_classes=10, tranNum = 8):
        
        super(MinstSteerableCNN_simple, self).__init__()

        self.tranNum = tranNum
        inP = 4
        
        self.block1 = nn.SequentialCell([
            fn.Fconv_PCA(7,1,10,tranNum,inP=5,padding=1, ifIni=1, bias=False),
            fn.F_BN(10,tranNum),
            nn.ReLU()
        ])
        
        self.block2 = nn.SequentialCell([
            fn.Fconv_PCA(5,10,10,tranNum,inP,2,bias=False),
            fn.F_BN(10,tranNum),
            nn.ReLU(),#fn.F_Dropout(zero_prob,tranNum)
        ])

        self.block3 = nn.SequentialCell([
            fn.Fconv_PCA(5,10,10,tranNum,inP,2,bias=False),
            fn.F_BN(10,tranNum),
            nn.ReLU(),#fn.F_Dropout(zero_prob,tranNum)
        ])
        
        self.block4 = nn.SequentialCell([
            fn.Fconv_PCA(5,10,10,tranNum,inP,2,bias=False),
            fn.F_BN(10,tranNum),
            nn.ReLU(),#fn.F_Dropout(zero_prob,tranNum)
        ])

        self.block5 = nn.SequentialCell([
            fn.Fconv_PCA(5,10,10,tranNum,inP,2,bias=False),
            fn.F_BN(10,tranNum),
            nn.ReLU(),#fn.F_Dropout(zero_prob,tranNum)
        ]) 
        
        
        self.block6 = nn.SequentialCell([
            fn.Fconv_PCA(5,10,10,tranNum,inP,1,bias=False),
            fn.F_BN(10,tranNum),
            nn.ReLU(),#fn.F_Dropout(zero_prob,tranNum)
        ])
        
        self.pool1 = MaxPool2d(2,2,1)
        self.pool2 = MaxPool2d(2,2,1)        
        self.pool3 = fn.PointwiseAvgPoolAntialiased(5, 1, padding=0)
        self.gpool = fn.GroupPooling(tranNum)
        
        self.fully_net = nn.SequentialCell([
            nn.Dense(10, 10),
            nn.BatchNorm1d(num_features=10),
            nn.ELU(),nn.Dropout(0.8),
            nn.Dense(10, n_classes),
        ])
    
    def construct(self, input: mindspore.Tensor, ifshow=0):
        # wrap the input tensor in a GeometricTensor

    
        x = self.block1(input)
        x1 = self.block2(x)
        x = self.pool1(x1)
        x = self.block3(x)
        x3 = self.block4(x)
        x = self.pool2(x3)
        x = self.block5(x)
        x2 = self.block6(x)      
        x = self.pool3(x2)
        x = self.gpool(x)

        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        if ifshow:
            
            I = mindspore.ops.Transpose()(x1[0,:,:,:], (1,2,0))
            sizeX = I.shape[0]
            I1 = mindspore.ops.Transpose()(I.reshape(sizeX,sizeX, -1, self.tranNum)[:,:,0:3,:], (0,3,1,2))
            I1 = I1.reshape(sizeX, sizeX*self.tranNum,3)
            I2 = mindspore.ops.Transpose()(I.reshape(sizeX,sizeX, self.tranNum, -1)[:,:,:,0:3], (0,2,1,3))
            I2 = I2.reshape(sizeX, sizeX*self.tranNum,3)
            ML.imshow(np.vstack((I1.asnumpy(),I2.asnumpy())))
            
            
            I = mindspore.ops.Transpose()(x3[0,:,:,:], (1,2,0))
            sizeX = I.shape[0]
            I1 = mindspore.ops.Transpose()(I.reshape(sizeX,sizeX, -1, self.tranNum)[:,:,0:3,:], (0,3,1,2))
            I1 = I1.reshape(sizeX, sizeX*self.tranNum,3)
            I2 = mindspore.ops.Transpose()(I.reshape(sizeX,sizeX, self.tranNum, -1)[:,:,:,0:3], (0,2,1,3))
            I2 = I2.reshape(sizeX, sizeX*self.tranNum,3)
            ML.imshow(np.vstack((I1.asnumpy(),I2.asnumpy())))
            
            
            I = mindspore.ops.Transpose()(x2[0,:,:,:], (1,2,0))
            sizeX = I.shape[0]
            I1 = mindspore.ops.Transpose()(I.reshape(sizeX,sizeX, -1, self.tranNum)[:,:,0:3,:], (0,3,1,2))
            I1 = I1.reshape(sizeX, sizeX*self.tranNum,3)
            I2 = mindspore.ops.Transpose()(I.reshape(sizeX,sizeX, self.tranNum, -1)[:,:,:,0:3], (0,2,1,3))
            I2 = I2.reshape(sizeX, sizeX*self.tranNum,3)
            ML.imshow(np.vstack((I1.asnumpy(),I2.asnumpy())))

        return x
#    def load_state_dict(self, state_dict, strict=False):
#        own_state = self.state_dict()
#        for name, param in state_dict.items():
#            if name in own_state:
#                if isinstance(param, nn.Parameter):
#                    param = param.data
#                try:
#                    own_state[name].copy_(param)
#                except Exception:
#                    raise RuntimeError('While copying the parameter named {}, '
#                                           'whose dimensions in the model are {} and '
#                                           'whose dimensions in the checkpoint are {}.'
#                                           .format(name, own_state[name].size(), param.size()))
#        if strict:
#            missing = set(own_state.keys()) - set(state_dict.keys())
#            if len(missing) > 0:
#                raise KeyError('missing keys in state_dict: "{}"'.format(missing))