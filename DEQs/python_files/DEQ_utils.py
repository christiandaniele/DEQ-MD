import torch
from DNCNN import DnCNN
from ICNN import ICNN
from f_theta_2 import f_theta_2
import deepinv as dinv
from DEQ import DEQFixedPoint
from utils.download_weights import download_weight
from Utils import RED_reg,torch_kernels


def get_path(alpha_poisson=100,kernel='Gauss'):

    path='Weights/'
    path = path + 'DnCNN_weights/' + kernel + '/best_model_checkpoint_' + kernel
    path= path +'_RED_DnCNN_alpha_' + str(alpha_poisson) + '.pth'

    return path

def create_operator(device='cuda',kernel='Gauss',shape=256):

    if kernel=='Gauss':
        filter=dinv.physics.blur.gaussian_blur(sigma=(1.2, 1.2), angle=0.0)

    elif kernel=='Uniform':
        filter=(1/81)*torch.ones((1,1,9,9))

    elif kernel=='Motion_7':
        kernels=torch_kernels('Levin09.mat')
        filter= kernels[7]

    else:
        print('Error: Kernel not implemented')
        return None,None

    physics=dinv.physics.BlurFFT(((1,1,shape,shape)),filter,device=device)
    return physics

def create_DEQ_model(device='cuda',kernel='Gauss', regularisation='RED',noise_level='low'):
    
    print(regularisation)
    
    physics = create_operator(device, kernel)
    
    weights_path = download_weight(regularization=regularisation,
                                                kernel=kernel,
                                                intensity=noise_level)
    checkpoint = torch.load(weights_path, map_location=device,weights_only=False)
    
    if regularisation == 'RED':
        
        net = DnCNN(depth=5)
        conv_net = RED_reg(net)
        funz = f_theta_2(network=conv_net)
        model = DEQFixedPoint(device, funz, physics, 0.5, 0.5, conv_net)

        #checkpoint = torch.load(path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        model = model.to(device)
        
        return model
            

    elif regularisation == 'Scalar':

        conv_net = ICNN(3)
        funz = f_theta_2(network=conv_net)
        model = DEQFixedPoint(device, funz, physics, 0.5, 0.5, conv_net)

        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        except Exception as e:
            conv_net = ICNN(3,ks=8)
            funz = f_theta_2(network=conv_net)
            model = DEQFixedPoint(device, funz, physics, 0.5, 0.5, conv_net)

            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        model = model.to(device)
        return model

    else:
        return None, None
