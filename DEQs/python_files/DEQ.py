import torch
import torch.nn as nn
from deepinv.loss.metric import PSNR

psnr=PSNR()

def forward_pass(device, f, y, x, eps, physics, L_0, chsi, lam=1, tol=2.5e-5, Proj=True, alpha_poisson=1, GT=None, log_metrics=False):
    x = x.to(device)
    x[x>1]=1
    L_0 = L_0.to(device)
    f0, L0, _ = f.forward(y, x, eps, physics, L_0, chsi, Proj=Proj, alpha_poisson=alpha_poisson)
    active_mask = torch.ones(f0.shape[0], dtype=torch.bool, device=device)
    
    metrics_by_image = [{'psnr_vett': [], 'residuals': [], 'stepsize_vett': [], 'fun_values': []} for _ in range(y.shape[0])]
    
    if log_metrics and GT is not None:
        for i in range(y.shape[0]):
            metrics_by_image[i]['psnr_vett'].append(psnr(physics.A_adjoint(y[i].unsqueeze(0)), GT[i].unsqueeze(0)).detach().cpu().numpy())

    iter = 1
    while active_mask.any():
        L = L0
        iter += 1
        
        f0_old = f0.clone()
        f0_active = f0[active_mask]
        L_active = L[active_mask]
        f0_old_active = f0_old[active_mask]
        
        f0_new_active, L0_new_active, fun_val = f.forward(y[active_mask], f0_old_active, eps, physics, L_active, chsi, lam, Proj=Proj, alpha_poisson=alpha_poisson)
        f0[active_mask] = f0_new_active
        L0[active_mask] = L0_new_active

        with torch.no_grad():
            
            residual = torch.norm(f0_new_active - f0_old_active, p=2, dim=(1, 2, 3)) / (torch.norm(f0_new_active, p=2, dim=(1, 2, 3)) + 1e-6)
            
            if log_metrics:
                active_indices = torch.where(active_mask)[0]
                for i, idx in enumerate(active_indices):

                    metrics_by_image[idx]['residuals'].append(residual[i].detach().cpu().numpy())
                    metrics_by_image[idx]['stepsize_vett'].append((1/L0_new_active[i]).detach().cpu().numpy())
                    metrics_by_image[idx]['fun_values'].append(fun_val[i].detach().cpu().numpy())
                    if GT is not None:
                        metrics_by_image[idx]['psnr_vett'].append(psnr(f0_new_active[i].unsqueeze(0), GT[idx].unsqueeze(0)).detach().cpu().numpy())
        
        active_mask[active_mask.clone()] = residual >= tol
        if not active_mask.any():
            break
        del f0_active, L_active, f0_old_active, f0_new_active, L0_new_active, residual
        torch.cuda.empty_cache()
    
    return f0, L0, metrics_by_image

class DEQFixedPoint(nn.Module):
    def __init__(self, device, f, physics, L_0, chsi, network, **kwargs):
        super().__init__()
        self.f = f
        self.physics = physics
        self.L_0 = L_0
        self.chsi = chsi
        self.network = network
        self.kwargs = kwargs
        self.device = device
        
    def forward(self, y, lam=1, training=False, Proj=True, inizialization=None, alpha_poisson=1, GT=None, log_metrics=False):
        L_0 = self.L_0 * torch.ones(y.shape[0])
        if inizialization is not None:
             x_0 = inizialization
        else:
             x_0 = self.physics.A_adjoint(y)
        
        z, L0, metrics_by_image = forward_pass(self.device, self.f, y, x_0, 1e-8, self.physics, L_0, self.chsi, lam, Proj=Proj, alpha_poisson=alpha_poisson, GT=GT, log_metrics=log_metrics, **self.kwargs)
        
        L0 = L0.to(self.device)
        z, _, _ = self.f.forward(y, z, 1e-8, self.physics, L0, self.chsi, lam, create_grapho=False, Proj=Proj, alpha_poisson=alpha_poisson)
        z0 = z.clone().detach().requires_grad_()
        f0, _, _ = self.f.forward(y, z0, 1e-8, self.physics, L0, self.chsi, lam, create_grapho=training, Proj=Proj, alpha_poisson=alpha_poisson)
        
        return f0, metrics_by_image

