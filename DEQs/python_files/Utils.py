import numpy as np
import os
import torch
import math
import torch
import os
from skimage import io
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import hdf5storage


def Bregman_L2(x,y):
    return 0.5*torch.sum((x-y)**2,dim=(1,2,3))

def L2_funz(x,y):
    return 0.5*torch.linalg.norm((x - y).flatten(start_dim=1), 2, dim=1)**2

def L2_grad(x,y,physics):
    return physics.A_adjoint(physics(x)-y)

def KL(y, x, eps,alpha=1):
    mask = (y == 0)
    mask_2 = (y != 0)
    kl = torch.zeros_like(y)
    kl[mask_2] = y[mask_2] * torch.log(y[mask_2] / (x[mask_2]*alpha + eps)) + x[mask_2] - y[mask_2]
    kl[mask] = x[mask]*alpha
    return torch.sum(kl, dim=(1, 2, 3))

def grad_KL(y, x, eps, physics,alpha=1):
    return (1/alpha)*physics.A_adjoint(torch.ones_like(x) - y / (alpha*physics(x) + eps))

def Bregman_h(x, u, eps):
    log_term = -torch.log(x + eps) + torch.log(u + eps)
    term1 = torch.sum(log_term, dim=(1, 2, 3))
    dot_term = (-1 / (u + eps)) * (x - u)
    term2 = torch.sum(dot_term, dim=(1, 2, 3))
    return term1 - term2

def Bregman_h_double(x, u, eps=1e-4):
    log_term=-torch.log((x+eps)*(1-x+eps))+torch.log((u+eps)*(1-u+eps))
    term1=torch.sum(log_term,dim=(1,2,3))
    dot_term=((1-2*u)/((u+eps)*(1-u+eps)))*(x-u)
    term2=torch.sum(dot_term,dim=(1,2,3))
    return term1+term2

def nabla_h_double(x,eps):
    return (2*x-1)/((x+eps)*(1-x+eps))

def nabla_h_double_star(x,eps):
    return (x-2+torch.sqrt(x**2+4))/(2*x+eps)

def print_gpu_memory():
    """Print GPU memory."""
    used_memory = torch.cuda.memory_allocated() / 1e9 
    cached_memory = torch.cuda.memory_reserved() / 1e9  
    print(f"Memory Allocated: {used_memory:.2f} GB")
    print(f"Memory Reserved: {cached_memory:.2f} GB")


def add_poisson_noise(tensor: torch.Tensor, alpha_poisson: float) -> torch.Tensor:
    """
    Add Poisson noise
    """
    tensor_np = tensor.cpu().numpy()  
    lam = np.maximum(alpha_poisson * tensor_np, 0)  
    noisy_np = np.random.poisson(lam).astype(np.float32) / alpha_poisson 
    noisy_tensor = torch.from_numpy(noisy_np).to(tensor.device) 
    return noisy_tensor


class RED_reg(nn.Module):
    def __init__(self, net):
        """
        Inizializza la rete neurale.

        Args:
            net (nn.Module): Una rete neurale che accetta un input vettoriale e restituisce un output della stessa dimensione.
        """
        super(RED_reg, self).__init__()
        self.net = net

    def forward(self, x):

        diff = x - self.net(x)  # (x - N(x))
        reg = 0.5 * torch.linalg.norm(diff.flatten(start_dim=1), 2, dim=1)**2  # 0.5*||(x - N(x))||^2
        return reg

    def denoising(self,x,param=1):
        """
        Calcola il valore scalare f(x) = 0.5*||x - net(x)||^2.

        Args:
            x (torch.Tensor): Input tensore.

        Returns:
            torch.Tensor: Valore scalare risultante.
        """
        # Calcolo del residuo: x - net(x)
        x.requires_grad_()
        aux=self.forward(x)
        grad_g=torch.autograd.grad(outputs=aux, inputs=x, grad_outputs=torch.ones_like(aux), create_graph=True)[0]
        return x-param*grad_g


def crop_center(img, cropx, cropy):
    y, x = img.shape[1],img.shape[2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:,starty : starty + cropy, startx : startx + cropx]

class CustomDataset(Dataset):
    def __init__(self, clean_images, noisy_images, params):
        self.clean_images = clean_images
        self.noisy_images = noisy_images
        self.params = params

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean_image = self.clean_images[idx]
        noisy_image = self.noisy_images[idx]
        param = self.params[idx]

        return clean_image, noisy_image,param

def GD(A,y,eps=1e-4,tol=1e-5):
    x_0=y
    x=x_0.clone().requires_grad_(True)
    aux=x_0.clone().requires_grad_(True)
    stepsize=1/(4+eps)
    while True:
        aux=x
        x=x-stepsize*(A(A(x)-y)+eps*x)
        if torch.norm((x-aux).flatten())<tol:
            break
    return x

class ReadDataset(Dataset):
    def __init__(self, path, list_image_paths, transform, physics):
        self.path = path
        self.list_image_paths = list_image_paths
        self.transform = transform
        self.physics = physics

    def __len__(self):
        return len(self.list_image_paths)

    def __getitem__(self, idx):
        img_path = self.list_image_paths[idx]
        image = Image.open(os.path.join(self.path, img_path)).convert("RGB")

        if self.transform:
            image = self.transform(image)
            image = image.to('cuda')

        blurred_image = self.physics(image)[0]

        return blurred_image, image, img_path

def create_image_tensor(folder_path,size=(256,256),cropping='True',device='cuda',PNG=False):

    if cropping:

        transform = transforms.Compose([
        transforms.ToTensor()
        ])

    else:

        transform = transforms.Compose([
        transforms.Resize(size),  
        transforms.ToTensor()
        ])
    images = []

    folder_path=folder_path


    if PNG:
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png')):
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path)
                img_tensor = transform(img)  
                if cropping:
                    img_tensor=crop_center(img_tensor,size[0],size[1])
                images.append(img_tensor)

    else:

        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg','.png','.bmp')):
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path)  
                img_tensor = transform(img)  
                if cropping:
                    img_tensor=crop_center(img_tensor,size[0],size[1])
                images.append(img_tensor)  

    images = torch.stack(images)
    images = images.to(device)

    return images

class CustomTensorDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, z_tensor):
        assert len(x_tensor) == len(y_tensor) == len(z_tensor)
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.z_tensor = z_tensor

    def __len__(self):
        return len(self.x_tensor)

    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx], self.z_tensor[idx]

def torch_kernels(mat_file_path):
	data = hdf5storage.loadmat(mat_file_path)
	kernels = data["kernels"]  

	torch_kernels = []

	for i in range(8):
		kernel = np.float32(kernels[0, i])
		torch_kernels.append(torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0))

	return torch_kernels


def print_network_parameters(model, model_name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"--- Statistics Parameters of {model_name} ---")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Paramters: {trainable_params:,}")
    print(f"Not Trainable Paramters: {non_trainable_params:,}")
    print("-" * (30 + len(model_name)))

