import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import time
import random
from Utils import create_image_tensor,RED_reg
from deepinv.loss.metric import PSNR
from DNCNN import DnCNN
from deepinv.physics import GaussianNoise
import random
from pathlib import Path
# Controlla se la GPU è disponibile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent
train_path = base_dir / 'training_images' / 'train'
val_path = base_dir / 'training_images' / 'val'
training_set = create_image_tensor(train_path,device=device)
validation_set = create_image_tensor(val_path,device=device)

blurred_train=(training_set)
blurred_val=(validation_set)

############TRAINING############
batch_size=15
val_batch_size=15
train_loader = DataLoader(TensorDataset(blurred_train,training_set), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(blurred_val,validation_set), batch_size=val_batch_size, shuffle=False)

net=DnCNN(depth=5)
model=RED_reg(net)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-4)  # Adam optimizer
scheduler = StepLR(optimizer, step_size=300, gamma=0.5)

start_time=time.time()
num_epochs=1200
Proj=True
N=training_set.shape[0]
loss_train_vett=[]
psnr_val_vett=[]
best_val_PSNR=0
noise_operator=GaussianNoise(0.2)
sigma_min=0.025
sigma_max=0.25
# tv_eps=1e-3

psnr=PSNR()

for epoch in range(num_epochs):
    model.train()
    train_loss=0
    i=1
    for batch_idx, batch in enumerate(train_loader):

        noisy_images_batch, clean_images_batch = batch
        n_examples = noisy_images_batch.shape[0]

        sigma = random.uniform(sigma_min,sigma_max)
        noise_operator.update_parameters(sigma=sigma)
        noisy_images_batch= noise_operator(noisy_images_batch)

        f0=model.denoising(noisy_images_batch)
        
        loss = torch.linalg.norm((f0 - clean_images_batch).flatten(start_dim=1), 2, dim=1)**2
        loss = loss.mean() 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss +=loss.detach().cpu().numpy()
    avg_train_loss = train_loss / len(train_loader)
    loss_train_vett.append(avg_train_loss)



    model.eval()

    PSNR_total=0
    
    for batch in val_loader:
        noisy_images_batch,clean_images_batch= batch
        noise_operator.update_parameters(sigma=0.1)
        noisy_images_batch= noise_operator(noisy_images_batch)

        output= model.denoising(noisy_images_batch)

        PSNR_val=psnr(output,clean_images_batch).detach().cpu().numpy()

        PSNR_val_mean=np.mean(PSNR_val)

        PSNR_total+=PSNR_val_mean

    PSNR=PSNR_total/len(val_loader)
    psnr_val_vett.append(PSNR)
    if PSNR > best_val_PSNR:
        best_val_PSNR = PSNR
        torch.save({
            'model_state_dict': model.state_dict()
        }, 'weights_GSDnCNN_denoiser_depth_5.pth')
        print(f'Model JFB saved at epoch {epoch + 1} with validation PSNR: {PSNR:.4f}')
    print(f'Model JFB, Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f} PSNR: {PSNR:.4f}')
    scheduler.step()
end_time=time.time()
print(f"Training time: {end_time-start_time}")
torch.save({
    'loss_train_vett': loss_train_vett,
}, 'training_GSDnCNN_denoiser_depth_5.pth')


print("Training complete, best model and metrics saved.")