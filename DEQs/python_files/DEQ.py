import torch
import torch.nn as nn

def forward_pass(device,f, y, x, eps, physics, L_0, chsi, lam=1, tol=1e-8,Proj=True,alpha_poisson=1,net=None):
    iterates=[]
    x       = x.to(device)
    L_0     = L_0.to(device)
    f0, L0  = f.forward(y, x, eps, physics, L_0, chsi,Proj=Proj,alpha_poisson=alpha_poisson)
    active_mask = torch.ones(f0.shape[0], dtype=torch.bool, device=device)
    residual=0
    N_iter=active_mask.int()
    while active_mask.any():
        L = L0
        f0_old = f0.clone()
        f0_active = f0[active_mask]
        L_active = L[active_mask]
        f0_old_active = f0_old[active_mask]
        f0_new_active, L0_new_active = f.forward(y[active_mask], f0_old_active, eps, physics, L_active, chsi, lam,Proj=Proj,alpha_poisson=alpha_poisson)
        f0[active_mask] = f0_new_active
        L0[active_mask] = L0_new_active
        with torch.no_grad():
            residual = torch.norm(f0_new_active - f0_old_active, p=2, dim=(1, 2, 3)) / torch.norm(f0_old_active, p=2, dim=(1, 2, 3))
        active_mask[active_mask.clone()] = residual >= tol
        N_iter+=active_mask.int()
        if not active_mask.any():
            break
        del f0_active, L_active, f0_old_active, f0_new_active, L0_new_active, residual
        torch.cuda.empty_cache()
    return f0,L,iterates

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
        
    def forward(self, y,lam=1,training=False,Proj=True,inizialition=None,alpha_poisson=1):
        L_0=self.L_0*torch.ones(y.shape[0])
        if inizialition is not None:
             x_0=inizialition
        else:
             x_0=self.physics.A_adjoint(y)
        z,L_0,_= forward_pass(self.device,self.f, y,x_0, 1e-10,self.physics, L_0, self.chsi,lam,Proj=Proj,**self.kwargs,alpha_poisson=alpha_poisson)
        L_0=L_0.to(self.device)
        z,_= self.f.forward(y, z, 1e-10, self.physics,L_0, self.chsi,lam,create_grapho=False,Proj=Proj,alpha_poisson=alpha_poisson)
        z0 = z.clone().detach().requires_grad_()
        f0,_ = self.f.forward(y, z0, 1e-10, self.physics,L_0, self.chsi,lam,create_grapho=training,Proj=Proj,alpha_poisson=alpha_poisson)
        return f0,z0

