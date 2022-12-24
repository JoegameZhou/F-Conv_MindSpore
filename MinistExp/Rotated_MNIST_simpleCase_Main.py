import mindspore
import mindspore.dataset as ds
import os
import argparse
import scipy.io as sio    
from SteerableCNN_XQ import MinstSteerableCNN_simple
from DataLoader import MnistRotDataset 
from mindspore.dataset.vision.c_transforms import RandomRotation, Resize
from mindspore.dataset.vision import Inter
from mindspore.dataset.transforms.c_transforms import Compose
from mindspore.dataset.vision.py_transforms import ToTensor
from MyLib import rotate_im
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--dir',type = str, default = 'SimpleNet' )
# parser.add_argument('--testModel', type = str, default='./Model/Model_best_red.pt')
parser.add_argument('--device', type = str, default = 'Ascend')
parser.add_argument('--mode',type = str, default = 'test' )
parser.add_argument('--weight_decay', type = float, default = 1e-2)
parser.add_argument('--InP', type = int, default = 4)
args = parser.parse_args()

mindspore.set_context(mode=mindspore.context.PYNATIVE_MODE, device_target=args.device)

mode = args.mode
tranNum  = 8
iniEpoch = 0
maxEpoch = 100
modelDir = os.path.join('./Models/'+ args.dir+ '/')
resultDir = os.path.join('./Results/'+ args.dir+ '/')
print(modelDir)
testModel = os.path.join(modelDir,'Model_best.ckpt')
testEvery = 1
saveEveryStep = True
use_test_time_augmentation = False
use_train_time_augmentation = False


ifshow = 0 # if show the Feature maps


# device = 'cuda:'+args.device if torch.cuda.is_available() else 'cpu'

model = MinstSteerableCNN_simple(10,tranNum)
milestone = [30,60,150,300]
learning_rates = [5e-3,1e-3,5e-4,1e-4]
scheduler = mindspore.nn.piecewise_constant_lr(milestone, learning_rates)
optimizer = mindspore.nn.Adam(model.trainable_params(), learning_rate=scheduler, weight_decay=args.weight_decay) 


def test_with_aug(test_loader,model,use_test_time_augmentation):
    total = 0
    correct = 0
    for data in test_loader:
        image = data['image']
        label = data['label']
        out = None
        if use_test_time_augmentation:
            #Run same sample with different orientations through network and average output
            rotations = [-15,0,15]
        else:
            rotations = [0]
                
        for rotation in rotations:
            for i in range(image.shape[0]):
                im = image[i,:,:,:].asnumpy().squeeze()
                # im = rotate_im(im, rotation)
                im = im.reshape([1,28,28])
                image[i,:,:,:] = mindspore.Tensor(im, dtype=mindspore.float32)
            if out is None:
                out = mindspore.ops.Softmax(1)(model(image))
            else:
                out+= mindspore.ops.Softmax(1)(model(image))
        out/= len(rotations)
        prediction, _ = mindspore.ops.ArgMaxWithValue(1)(out)
        total += label.shape[0]
        correct += (prediction == label).sum()
    return correct*1.0/total*100

totensor = ToTensor()    
if mode == 'train':
    
    try:
        os.makedirs(modelDir)
    except OSError:
        pass
    
    try:
        os.makedirs(resultDir)
    except OSError:
        pass
    
    if iniEpoch:
        # load the previous trained model
        model_checkpoint = mindspore.load_checkpoint(os.path.join(modelDir, str(iniEpoch) + '_' + 'model' + '.ckpt'))
        optimizer_checkpoint = mindspore.load_checkpoint(os.path.join(modelDir, str(iniEpoch) + '_' + 'optimizer' + '.ckpt'))
        mindspore.load_param_into_net(model, model_checkpoint)
        mindspore.load_param_into_net(optimizer, optimizer_checkpoint)
        print('loaded checkpoints, epoch ' + str(iniEpoch))
        
    if use_train_time_augmentation:
        train_transform = Compose([
            Resize(87, Inter.BILINEAR),
            RandomRotation(360, resample=Inter.BILINEAR, expand=False),
            Resize(28, Inter.BILINEAR),
            totensor,
        ])
    else:
        train_transform = totensor
    mnist_train = MnistRotDataset(mode='train', transform=train_transform)
    train_dataset = ds.GeneratorDataset(source=mnist_train, column_names=['image','label'],shuffle=True)
    train_dataset = train_dataset.batch(batch_size=128)
    train_loader = train_dataset.create_dict_iterator()
    
    test_transform = totensor
    mnist_test = MnistRotDataset(mode='test', transform=test_transform)
    test_dataset = ds.GeneratorDataset(source=mnist_test, column_names=['image','label'])
    test_dataset = test_dataset.batch(batch_size=100)
    test_loader = test_dataset.create_dict_iterator()
    loss_function = mindspore.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    loss_net = mindspore.nn.WithLossCell(model, loss_function)
    train_net = mindspore.nn.TrainOneStepCell(loss_net, optimizer)

    best_acc = 0
    ifshow = 0
    
    for epoch in range(iniEpoch, maxEpoch):
        train_net.set_train(True)
        total = 0
        correct = 0 
        # one epoch training
        for data in train_loader:
            # optimizer.zero_grad()    
            # x = x.to(device)
            # t = t.to(device)
            image = data['image']
            label = data['label']
            train_net(image, label)
            # loss_class = loss_function(y, label)
            # l1_regularization = 0
            # loss = loss_class + l1_regularization
            # loss.backward()#retain_graph=True
            # optimizer.step()
            y = model(image,ifshow)
            argmax = mindspore.ops.ArgMaxWithValue(1)
            prediction, _ = argmax(y)
            total += label.shape[0]
            correct += (prediction == label).sum()
        train_acc = correct*1.0/total*100  
        print(f"epoch {epoch} | train accuracy: {train_acc}")
        # scheduler.step()
        if epoch == 200:
            milestone = [90,300]
            learning_rates = [5e-5,5e-6]
            scheduler = mindspore.nn.piecewise_constant_lr(milestone, learning_rates)
            optimizer = mindspore.nn.Adam(model.trainable_params(), learning_rate=scheduler, weight_decay=5e-3)

        # test the model and save the best model     
        if epoch % testEvery == 0 and epoch>30:
            train_net.set_train(False)
            test_acc = test_with_aug(test_loader, model,use_test_time_augmentation)
            print(f"The test accuracy: {test_acc}")
            print('The test error is %.5f' %(100-test_acc))
            if test_acc > best_acc:
                best_acc = test_acc
                save_path_model = os.path.join(modelDir, 'Model_best.ckpt')
                mindspore.save_checkpoint(model, save_path_model)
                # sio.savemat(resultDir+'acc.mat',{'acc':best_acc})
            print('The best error is %.5f' %(100-best_acc))
            print('=================================================')    

        # save the model     
        if saveEveryStep:
            save_path_model = os.path.join(modelDir, str(epoch+1) + '_' + 'model' + '.ckpt')
            save_path_optimizer = os.path.join(modelDir, str(epoch+1) + '_' + 'optimizer' + '.ckpt')
            mindspore.save_checkpoint(model, save_path_model)
            mindspore.save_checkpoint(optimizer, save_path_optimizer)
else:
    mindspore.load_param_into_net(model, mindspore.load_checkpoint(testModel))
    test_transform = totensor
    mnist_test = MnistRotDataset(mode='test', transform=test_transform)
    dataset = ds.GeneratorDataset(source=mnist_test, column_names=['image','label'])
    dataset = dataset.batch(batch_size=100)
    test_loader = dataset.create_dict_iterator()
    
    test_acc = test_with_aug(test_loader, model,use_test_time_augmentation)
    print(f"epoch {iniEpoch} | test error: {100-test_acc}")