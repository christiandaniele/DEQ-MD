import torch
from DNCNN import DnCNN
from ICNN import ICNN
from f_theta_2 import f_theta_2
import deepinv as dinv
from DEQ import DEQFixedPoint
from Utils import torch_kernels,RED_reg,create_image_tensor,add_poisson_noise
from deepinv.loss.metric import PSNR,SSIM,LPIPS


def get_path(alpha_poisson=100,kernel='Gauss'):

    path='Weights/'
    path = path + 'DnCNN_weights/' + kernel + '/best_model_checkpoint_' + kernel
    path= path +'_RED_DnCNN_alpha_' + str(alpha_poisson) + '.pth'

    return path

def create_operator(device='cuda',kernel='Gauss'):

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

    physics=dinv.physics.BlurFFT(((1,3,256,256)),filter,device=device)
    return physics

def create_DEQ_model(device='cuda', alpha_poisson=100, kernel='Gauss', regularisation='RED'):
    
    path = 'Weights/'
    physics = create_operator(device, kernel)

    if regularisation == 'RED':
        try:
            net = DnCNN(depth=5)
            conv_net = RED_reg(net)
            funz = f_theta_2(network=conv_net)
            model = DEQFixedPoint(device, funz, physics, 0.5, 0.5, conv_net)
            path = path + 'DnCNN_weights/' + kernel + '/weights_' + kernel
            path = path + '_RED_DnCNN_alpha_' + str(alpha_poisson) + '.pth'
            
            checkpoint = torch.load(path, weights_only=False, map_location=device)
            checkpoint['model_state_dict']['physics.filter'] = physics.filter
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            
            return model, checkpoint
        except(RuntimeError, FileNotFoundError, KeyError):
            net = DnCNN(depth=5)
            conv_net = RED_reg(net)
            funz = f_theta_2(network=conv_net)
            model = DEQFixedPoint(device, funz, physics, 0.5, 0.5, conv_net)
            path = path + 'DnCNN_weights/' + kernel + '/weights_' + kernel
            path = path + '_RED_DnCNN_alpha_' + str(alpha_poisson) + '.pth'
            
            checkpoint = torch.load(path, weights_only=False, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            
            return model, checkpoint
            

    elif regularisation == 'Scalar':
        try:
            conv_net = ICNN(3)
            funz = f_theta_2(network=conv_net)
            model = DEQFixedPoint(device, funz, physics, 0.5, 0.5, conv_net)
            path_icnn = path + 'ICNN_weights/' + kernel + '/weights_' + kernel + '_Scalar_alpha_' + str(alpha_poisson) + '.pth'
            checkpoint = torch.load(path_icnn, weights_only=False, map_location=device)
            checkpoint['model_state_dict']['physics.filter'] = physics.filter
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            return model, checkpoint
        
        except (RuntimeError, FileNotFoundError, KeyError):
            try:
                conv_net_ks8 = ICNN(3, ks=8)
                funz_ks8 = f_theta_2(network=conv_net_ks8)
                model_ks8 = DEQFixedPoint(device, funz_ks8, physics, 0.5, 0.5, conv_net_ks8)
                path_ks8 = path + 'ICNN_weights/' + kernel + '/weights_' + kernel + '_Scalar_alpha_' + str(alpha_poisson) + '.pth'
                checkpoint_ks8 = torch.load(path_ks8, weights_only=False, map_location=device)
                checkpoint_ks8['model_state_dict']['physics.filter'] = physics.filter
                model_ks8.load_state_dict(checkpoint_ks8['model_state_dict'])
                model_ks8 = model_ks8.to(device)
                return model_ks8, checkpoint_ks8
            
            except (RuntimeError, FileNotFoundError, KeyError):
                try:
                    conv_net_ks8 = ICNN(3, ks=8)
                    funz_ks8 = f_theta_2(network=conv_net_ks8)
                    model_ks8 = DEQFixedPoint(device, funz_ks8, physics, 0.5, 0.5, conv_net_ks8)
                    path_ks8 = path + 'ICNN_weights/' + kernel + '/weights_' + kernel + '_Scalar_alpha_' + str(alpha_poisson) + '.pth'
                    checkpoint_ks8 = torch.load(path_ks8, weights_only=False, map_location=device)
                    model_ks8.load_state_dict(checkpoint_ks8['model_state_dict'])
                    model_ks8 = model_ks8.to(device)
                    return model_ks8, checkpoint_ks8
                
                except (RuntimeError, FileNotFoundError, KeyError):
                    try:
                        conv_net = ICNN(3)
                        funz = f_theta_2(network=conv_net)
                        model = DEQFixedPoint(device, funz, physics, 0.5, 0.5, conv_net)
                        path_icnn = path + 'ICNN_weights/' + kernel + '/weights_' + kernel + '_Scalar_alpha_' + str(alpha_poisson) + '.pth'
                        checkpoint = torch.load(path_icnn, weights_only=False, map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model = model.to(device)
                        return model, checkpoint
                    
                    except (RuntimeError, FileNotFoundError, KeyError):
                        return None, None
    else:
        return None, None


def test_on_images(images_path, device='cuda', alphas=[100, 60, 40], kernels=['Gauss', 'Motion_7', 'Uniform'],regs=['RED']):

    psnr = PSNR()
    ssim = SSIM()
    lpips_loss = LPIPS()
    lpips_loss=lpips_loss.to(device)

    images = create_image_tensor(images_path).to(device)

    all_metrics = {}
    i=0
    for kernel in kernels:
        all_metrics[kernel] = {}
        for alpha in alphas:
            all_metrics[kernel][alpha] = {}
            for reg in regs:
                i+=1
                print(i)
                all_metrics[kernel][alpha][reg] = {}

                if reg == 'RED':
                    deq_model = create_DEQ_model(device, alpha, kernel, reg)
                elif reg == 'Scalar':
                    deq_model = create_DEQ_model(device, alpha, kernel, reg)
                else:
                    print(f"Warning: Regularization '{reg}' not recognized. Skipping.")
                    continue

                mes = add_poisson_noise(deq_model.physics(images), alpha_poisson=alpha)
                rec, _ = deq_model(mes)
                
                with torch.no_grad():
                    psnr_val = psnr(images, rec)
                    ssim_val = ssim(images, rec)
                    lpips_val = lpips_loss(images, rec)

                all_metrics[kernel][alpha][reg]['psnr'] = psnr_val
                all_metrics[kernel][alpha][reg]['ssim'] = ssim_val
                all_metrics[kernel][alpha][reg]['lpips'] = lpips_val

    return all_metrics
