import torch
from Utils import grad_KL, KL, Bregman_h

class f_theta_2():
    def __init__(self, network):
        self.network = network
    
    def forward(self, y, x, eps, physics, L_0, chsi, lam=1, create_grapho=False,Proj=True,alpha_poisson=1):
        x.requires_grad_()  
        L = L_0
        stepsize = 1 / (2 * L)

        phi = lam * self.network(x)
        grad_phi = torch.autograd.grad(
            outputs=phi,
            inputs=x,
            grad_outputs=torch.ones_like(phi),
            create_graph=create_grapho
        )[0]
        with torch.no_grad() if not create_grapho else torch.enable_grad():
            grad_kl = grad_KL(y, x, eps, physics,alpha=alpha_poisson)

        while True:
            with torch.no_grad() if not create_grapho else torch.enable_grad():
                T_L = x / (1 + stepsize[:, None, None, None] * x * (grad_kl + grad_phi))
                if Proj:
                    T_L[T_L>1]=1
                d = (0.8 / stepsize) * Bregman_h(T_L, x, 0)
                kl_diff = (KL(y, physics(T_L), eps,alpha=alpha_poisson) - KL(y, physics(x), eps,alpha=alpha_poisson))
                phi_diff = (lam * self.network(T_L) - phi).flatten()
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

        # Libera memoria per tensori intermedi
        del phi, grad_phi, grad_kl, kl_diff, phi_diff
        torch.cuda.empty_cache()
        return T_L, L
