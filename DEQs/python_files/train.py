import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import time
from Utils import create_image_tensor, add_poisson_noise,RED_reg
from DEQ_utils import create_operator
from deepinv.loss.metric import PSNR
from TV import TVPrior
from DNCNN import DnCNN
from f_theta_2 import f_theta_2
from DEQ import DEQFixedPoint
from ICNN import ICNN
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description='script for training of DEQ-MD method.')

parser.add_argument(
    '--kernel_type',
    type=str,
    choices=['Gauss', 'Motion_7', 'Uniform'],
    required=True,
    help='Specifica il tipo di kernel da usare.'
)

parser.add_argument(
    '--noise_level',
    type=str,
    choices=['high', 'medium', 'low'],
    required=True,
    help='Specifica il livello di rumore.'
)

parser.add_argument(
    '--model',
    type=str,
    choices=['DEQ-RED', 'DEQ-S'],
    required=True,
    help='Specifica il modello da usare.'
)

args = parser.parse_args()

print(f"Tipo di kernel selezionato: {args.kernel_type}")
print(f"Livello di rumore selezionato: {args.noise_level}")
print(f"Modello selezionato: {args.model}")

if args.noise_level=='high':
    alpha_poisson=40
elif args.noise_level=='medium':
    alpha_poisson=60
else:
    alpha_poisson=100

psnr=PSNR()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

physics=create_operator(device,args.kernel_type)

if args.model=='DEQ-RED':
    net=DnCNN(depth=5)
    conv_net = RED_reg(net)
    weights = torch.load('weights_GSDnCNN_denoiser_depth_5.pth', map_location=device,weights_only=False)
    conv_net.load_state_dict(weights['model_state_dict'])
    funz = f_theta_2(network=conv_net)
    model = DEQFixedPoint(device,funz,physics,0.5,0.5,conv_net)
    model.to(device)
elif args.model=='DEQ-S':
    conv_net = ICNN(ks=8)
    funz = f_theta_2(network=conv_net)
    model = DEQFixedPoint(device,funz,physics,0.5,0.5,conv_net)
    model.to(device)

script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent

train_path = base_dir / 'training_images' / 'train'
val_path = base_dir / 'training_images' / 'val'
training_set = create_image_tensor(train_path,device=device)
validation_set = create_image_tensor(val_path,device=device)

blurred_train=physics(training_set)
blurred_val=physics(validation_set)

############TRAINING############
batch_size=18
val_batch_size=16
train_loader = DataLoader(TensorDataset(blurred_train,training_set), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(blurred_val,validation_set), batch_size=val_batch_size, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=5e-4)  # Adam optimizer
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)


start_time=time.time()
num_epochs=50
Proj=True
tv_eps=1e-3
tv=TVPrior()
N=training_set.shape[0]
loss_train_vett=[]
psnr_val_vett=[]
best_val_PSNR=0

for epoch in range(num_epochs):
    model.train()
    train_loss=0
    i=1
    for batch_idx, batch in enumerate(train_loader):

        #print(i)
        i+=1
        noisy_images_batch, clean_images_batch = batch
        n_examples = noisy_images_batch.shape[0]

        # Forward pass
        noisy_images_batch=add_poisson_noise(noisy_images_batch,alpha_poisson)

        f0,_,_,_,_=model(noisy_images_batch,training=True,Proj=True)
        loss = torch.linalg.norm((f0 - clean_images_batch).flatten(start_dim=1), 2, dim=1)**2#+tv_eps*tv.forward(f0)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss +=loss.item()

    avg_train_loss = train_loss / len(train_loader)
    loss_train_vett.append(avg_train_loss)



    model.eval()

    PSNR_total=0
    
    for batch in val_loader:
        noisy_images_batch,clean_images_batch= batch
        noisy_images_batch=add_poisson_noise(noisy_images_batch,alpha_poisson)

        output,_ ,_,_,_= model(noisy_images_batch,training=False,Proj=True)

        PSNR_val=psnr(output,clean_images_batch)

        PSNR_val_mean=torch.mean(PSNR_val)

        PSNR_total+=PSNR_val_mean

    PSNR=PSNR_total/len(val_loader)
    psnr_val_vett.append(PSNR)
    
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir.parent / 'training_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    aux = f'weights_{args.model}_{args.kernel_type}_{args.noise_level}.pth'
    weights_path = results_dir / aux
    
    if PSNR > best_val_PSNR:
        best_val_PSNR = PSNR
        torch.save({
            'model_state_dict': model.state_dict(),
        }, weights_path) 
        print(f'Model JFB saved at epoch {epoch + 1} with validation PSNR: {PSNR:.4f}')
    print(f'Model JFB, Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f} PSNR: {PSNR:.4f}')
    scheduler.step()
end_time=time.time()
print(f"Training time: {end_time-start_time}")

aux = f'training_metrics_{args.model}_{args.kernel_type}_{args.noise_level}.pth'
metrics_path=results_dir / aux
torch.save({
    'loss_train_vett': loss_train_vett,
    'psnr_val_vett':psnr_val_vett
}, metrics_path)


print("Training complete, best model and metrics saved.")