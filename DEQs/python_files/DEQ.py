import torch
import torch.nn as nn
from deepinv.loss.metric import PSNR

psnr=PSNR()

def forward_pass(device,f, y, x, eps, physics, L_0, chsi, lam=1, tol=2.5e-5,Proj=True,alpha_poisson=1,GT=None):
    psnr_vett=[]
    psnr_vett.append(psnr(physics.A_adjoint(y),GT).detach().cpu().numpy() if GT is not None else 0)
    stepsize_vett=[]
    fun_values=[]
    residuals=[]
    x       = x.to(device)
    L_0     = L_0.to(device)
    f0, L0,_  = f.forward(y, x, eps, physics, L_0, chsi,Proj=Proj,alpha_poisson=alpha_poisson)
    active_mask = torch.ones(f0.shape[0], dtype=torch.bool, device=device)
    residual=1
    N_iter=active_mask.int()
    iter=1
    while active_mask.any():
    #for i in range(1):
        L = L0
        iter+=1
        #print(f"Iteration {iter}, active elements: {active_mask.sum()}")
        # if iter % 1 == 0:
        #     print(f"Iteration {iter}, active elements: {active_mask.sum()}")
        f0_old = f0.clone()
        f0_active = f0[active_mask]
        L_active = L[active_mask]
        f0_old_active = f0_old[active_mask]
        f0_new_active, L0_new_active,fun_val = f.forward(y[active_mask], f0_old_active, eps, physics, L_active, chsi, lam,Proj=Proj,alpha_poisson=alpha_poisson)
        f0[active_mask] = f0_new_active
        L0[active_mask] = L0_new_active
        with torch.no_grad():
            residual=torch.norm(f0_new_active - f0_old_active, p=2, dim=(1, 2, 3)) / (torch.norm(f0_new_active, p=2, dim=(1, 2, 3))+1e-6)
            # residuals.append(residual.detach().cpu().numpy())
            # stepsize_vett.append((1/L0[active_mask]).detach().cpu().numpy())
            # fun_values.append(fun_val.detach().cpu().numpy())
            # if GT is not None:
            #     psnr_vett.append(psnr(f0[active_mask],GT[active_mask]).detach().cpu().numpy())
        active_mask[active_mask.clone()] = residual >= tol
        N_iter+=active_mask.int()
        if not active_mask.any():
            break
        del f0_active, L_active, f0_old_active, f0_new_active, L0_new_active, residual
        torch.cuda.empty_cache()
    return f0,L,psnr_vett,residuals,stepsize_vett,fun_values

class DEQFixedPoint(nn.Module):
    def __init__(self, device,f, physics, L_0, chsi, network, **kwargs):
        super().__init__()
        self.f = f
        self.physics = physics
        self.L_0 = L_0
        self.chsi = chsi
        self.network = network
        self.kwargs = kwargs
        self.device=device
        
    def forward(self, y,lam=1,training=False,Proj=True,inizialization=None,alpha_poisson=1,GT=None):
        L_0=self.L_0*torch.ones(y.shape[0])
        if inizialization is not None:
             x_0=inizialization
        else:
             x_0=self.physics.A_adjoint(y)
        z,L_0,psnr_vett,residuals,stepsize_vett,fun_values = forward_pass(self.device,self.f, y,x_0, 1e-10,self.physics, L_0, self.chsi,lam,Proj=Proj,alpha_poisson=alpha_poisson,GT=GT,**self.kwargs)
        L_0=L_0.to(self.device)
        z,_,_= self.f.forward(y, z, 1e-10, self.physics,L_0, self.chsi,lam,create_grapho=False,Proj=Proj,alpha_poisson=alpha_poisson)
        z0 = z.clone().detach().requires_grad_()
        f0,_,_ = self.f.forward(y, z0, 1e-10, self.physics,L_0, self.chsi,lam,create_grapho=training,Proj=Proj,alpha_poisson=alpha_poisson)
        return f0,psnr_vett,residuals,stepsize_vett,fun_values

