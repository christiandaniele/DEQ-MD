###code to normalize the forward operator
import deepinv as dinv
import torch

def create_normalized_operator(filter,device,scale_factor=1)

  aux=dinv.physics.Downsampling(factor=scale_factor,device=device,img_size=(3,256,256),filter=filter)
  
  #norm computation
  x_toy = torch.randn((1,3,256,256)).to(device)
  norm = aux.compute_norm(x_toy)
  
  #def normalized forward and adjoint
  def A_normalized(x, **kwargs):
      return aux.A(x) / norm.sqrt()
  
  def A_adjoint_normalized(x, **kwargs):
      return aux.A_adjoint(x) / norm.sqrt()
  
  physics = dinv.physics.LinearPhysics(A=A_normalized, 
                                       A_adjoint=A_adjoint_normalized, 
                                       device=device)
  return physics
