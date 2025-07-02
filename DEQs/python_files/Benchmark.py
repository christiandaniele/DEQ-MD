import torch
from Utils import grad_KL, KL, Bregman_h
from TV import TVPrior
from deepinv.loss.metric import PSNR,SSIM

tv=TVPrior()
psnr=PSNR()
ssim=SSIM()

class Benchmark:

    def __init__(self, regularisation, device):
        self.regularisation = regularisation
        self.device = device

    def forward(self,y,physics,x,L_0=1,chsi=0.5,reg_param=1e-3,eps=1e-8,alpha_poisson=1):

        if self.regularisation=='TV':

            L=L_0
            stepsize = 1 / (2 * L)
            x.requires_grad_()

            old=tv.g(x)

            grad=torch.autograd.grad(old,x,torch.ones_like(old))[0]

            with torch.no_grad():
                    grad_kl=grad_KL(y, x, eps, physics,alpha=alpha_poisson)

            while True:

                with torch.no_grad():

                    T_L = (x / (1 + stepsize[:,None,None,None] * x * (grad_kl+reg_param*grad)))
                    T_L[T_L>1]=1
                    d = (0.8 / stepsize) * Bregman_h(T_L, x, 0)
                    kl_diff = (KL(y, physics(T_L), eps,alpha=alpha_poisson) - KL(y, physics(x), eps,alpha=alpha_poisson))
                    phi_diff = reg_param*(tv.g(T_L) - old)
                    stop_condition = (-kl_diff - phi_diff) > d
                    stop_condition_equal = (-kl_diff - phi_diff) >= 0

                if (stop_condition + stop_condition_equal).all():
                    break

                with torch.no_grad():
                    stepsize = torch.where(
                        torch.min(stop_condition + stop_condition_equal, torch.ones(stop_condition.shape[0], dtype=torch.bool, device=x.device)),
                        stepsize,
                        stepsize * chsi
                    )
                    
                    L = torch.where(
                        torch.min(stop_condition + stop_condition_equal, torch.ones(stop_condition.shape[0], dtype=torch.bool, device=x.device)),
                        L,
                        L / chsi
                    )
            return T_L,L

        else:

            T_L=(physics.A_adjoint(y/(physics(x)+eps))**0.1)*x
            T_L=torch.clamp(T_L,0,1)

            return T_L

    
    def forward_pass(self,y, x_0, eps, physics, L_0=1, chsi=0.5, lam=1e-3, tol=1e-10,max_iter=5000,alpha_poisson=1):
        x_0       = x_0.to(self.device)
        x= x_0.clone()
        if self.regularisation=='TV':
            L_0=L_0*torch.ones(y.shape[0])
            L_0     = L_0.to(self.device)
            f0, L0  = self.forward(y,physics,x,L_0,chsi=chsi,reg_param=lam,eps=eps,alpha_poisson=alpha_poisson)
            active_mask = torch.ones(f0.shape[0], dtype=torch.bool, device=self.device)
            residual=0
            i=0
            while active_mask.any():
            #for i in  range(max_iter):
                L = L0
                f0_old = f0.clone()
                f0_active = f0[active_mask]
                L_active = L[active_mask]
                f0_old_active = f0_old[active_mask]
                f0_new_active, L0_new_active = self.forward(y[active_mask],physics, f0_old_active, L_active,chsi=chsi,reg_param=lam,eps=eps,alpha_poisson=alpha_poisson)
                f0[active_mask] = f0_new_active
                L0[active_mask] = L0_new_active
                residual = torch.norm(f0_new_active - f0_old_active, p=2, dim=(1, 2, 3)) / torch.norm(f0_old_active, p=2, dim=(1, 2, 3))
                active_mask[active_mask.clone()] = residual >= tol

                if not active_mask.any():
                    break
                del f0_active, L_active, f0_old_active, f0_new_active, L0_new_active, residual
                torch.cuda.empty_cache()
                i+=1

            return f0
        
        else:

            iterates = []
            f0=self.forward(y,physics,x)
            iterates.append(f0.detach())

            for i in range(max_iter-1):

                f0=self.forward(y,physics,f0)
                iterates.append(f0.detach())

            return f0, iterates

    def Get_best_iter(self,y,x_0,eps,physics,GT,L_0=1, chsi=0.5,tol=1e-10,max_iter=5000,alpha_poisson=1,metric='PSNR',reg_param_list=None):

        batch_size = y.shape[0]
        best_reconstructions = torch.zeros_like(GT).to(self.device)
        best_metric_values = torch.full((batch_size,), -torch.inf if metric == 'PSNR' else -1.0, dtype=torch.float32).to(self.device)

        if reg_param_list is not None:

            best_lambda_values = torch.zeros(batch_size).to(self.device)
            best_reconstructions = x_0.clone().detach() 

            for lam in reg_param_list:
                x_rec_batch = self.forward_pass(y, x_0, eps, physics, L_0=L_0, chsi=chsi, lam=lam, tol=tol, max_iter=max_iter, alpha_poisson=alpha_poisson)
                
                if metric == 'PSNR':
                    metric_batch = psnr(x_rec_batch, GT)
                elif metric == 'SSIM':
                    metric_batch = ssim(x_rec_batch, GT)
                else:
                    raise ValueError(f"Metrica non supportata: {metric}")

                for j in range(batch_size):
                    current_metric = metric_batch[j]

                    if current_metric > best_metric_values[j]:
                        best_metric_values[j] = current_metric
                        best_reconstructions[j] = x_rec_batch[j].clone().detach()
                        best_lambda_values[j] = lam

            return best_reconstructions, best_metric_values, best_lambda_values
        
        else: 
            
            best_num_iter = torch.ones(batch_size).to(self.device)
            _,iter_rec_batch=self.forward_pass(y, x_0, eps, physics,max_iter=max_iter)

            for n in range(max_iter):

                x_rec_batch = iter_rec_batch[n].clone()

                if metric == 'PSNR':
                    metric_batch = psnr(x_rec_batch, GT)
                elif metric == 'SSIM':
                    metric_batch = ssim(x_rec_batch, GT)
                else:
                    raise ValueError(f"Metrica non supportata: {metric}")
                
                for j in range(batch_size):
                    current_metric = metric_batch[j]

                    if current_metric > best_metric_values[j]:
                        best_metric_values[j] = current_metric
                        best_reconstructions[j] = x_rec_batch[j].clone().detach()
                        best_num_iter[j] = n

            return best_reconstructions, best_metric_values, best_num_iter






      