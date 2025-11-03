import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import deepinv as dinv
from deepinv.loss.metric import PSNR, SSIM, LPIPS
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from DNCNN import DnCNN
from ICNN import ICNN
from f_theta_2 import f_theta_2
import deepinv as dinv
from DEQ import DEQFixedPoint
from Utils import RED_reg,create_image_tensor,add_poisson_noise
from deepinv.loss.metric import PSNR,SSIM,LPIPS
from torch.utils.data import TensorDataset, DataLoader
from DEQ_utils import create_DEQ_model


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for DEQ models.")
    
    parser.add_argument('--kernel', type=str, required=True, choices=['Gauss', 'Uniform', 'Motion_7'],
                        help='Type of degradation kernel.')
    parser.add_argument('--noise_level', type=str, required=True, choices=['low', 'medium', 'high'],
                        help='Level of Poisson noise.')
    parser.add_argument('--regularisation', type=str, required=True, choices=['RED', 'Scalar'],
                        help='Type of regularisation used in the model.')
    
    parser.add_argument('--input_path', type=str, default='set3c/',
                        help='Path to the images for inference.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of images to process simultaneously.')
    parser.add_argument('--plot_metrics', action='store_true',
                        help='If specified, plots the per-iteration metrics.')
    parser.add_argument('--save_images', action='store_true',
                        help='If specified, saves the GT, measurement, initialization, and reconstruction images.')
    parser.add_argument('--save_results', action='store_true',
                        help='If specified, saves average numerical metrics to a results.txt file.')

    return parser.parse_args()

def save_images(output_path, gt_image, y_image, init_image, rec_image, filename):
    output_path.mkdir(parents=True, exist_ok=True)
    
    h, w = gt_image.shape[-2:]
    combined_image = Image.new('RGB', (w * 4, h))

    #clamping in [0,1]
    combined_image.paste(to_pil_image(torch.clamp(gt_image, 0, 1)), (0, 0))
    combined_image.paste(to_pil_image(torch.clamp(y_image, 0, 1)), (w, 0))
    combined_image.paste(to_pil_image(torch.clamp(init_image, 0, 1)), (w * 2, 0))
    combined_image.paste(to_pil_image(torch.clamp(rec_image, 0, 1)), (w * 3, 0))

    combined_image.save(output_path / f'{filename}.png')

def plot_metrics(metrics_by_image, output_path, filename):
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, metrics in enumerate(metrics_by_image):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Metrics for Image {i+1}', fontsize=16)

        axes[0].plot(metrics['psnr_vett'], marker='o')
        axes[0].set_title('PSNR vs. Iteration')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('PSNR (dB)')

        axes[1].plot(metrics['residuals'], marker='o')
        axes[1].set_title('Residuals vs. Iteration')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('L2 Residual')
        axes[1].set_yscale('log')

        axes[2].plot(metrics['fun_values'], marker='o')
        axes[2].set_title('Function Values vs. Iteration')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Function Value')

        plt.tight_layout()
        plt.savefig(output_path / f'{filename}_metrics_img_{i+1}.png')
        plt.close(fig)

def save_results_to_txt(results, output_path, filename):
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / f'{filename}.txt', 'a') as f:
        f.write(f"--- Results for {filename} ---\n")
        f.write(f"PSNR (dB): {results['psnr']:.4f}\n")
        f.write(f"SSIM: {results['ssim']:.4f}\n")
        f.write(f"LPIPS: {results['lpips']:.4f}\n")
        f.write("--------------------------------\n\n")

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_base_path = Path('inference_results')
    output_dir_name = f"{args.kernel}_{args.noise_level}_{args.regularisation}"
    output_path = output_base_path / output_dir_name

    print("Loading the model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #create DEQ-MD model
    model=create_DEQ_model(device=device,
                           kernel=args.kernel,
                           regularisation=args.regularisation,
                           noise_level=args.noise_level)
    
    if model is None:
        print("Error: Could not load the model. Please check the parameters and file paths.")
        return
    
    print("Loading images...")
    gt_images = create_image_tensor(Path(args.input_path), device=device)
    
    dataset = TensorDataset(gt_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    model.eval()
    
    #metrics
    psnr = PSNR()
    ssim = SSIM()
    lpips = LPIPS(device=device)
    
    psnr_values = []
    ssim_values = []
    lpips_values = []
    all_metrics_by_image = []
    
    print("Running inference in batches...")
    physics = model.physics
    
    if args.noise_level == 'high':
        alpha_poisson = 40
    elif args.noise_level == 'medium':
        alpha_poisson = 60
    else:
        alpha_poisson = 100
    
    for i, (gt_batch,) in enumerate(tqdm(loader, desc="Inference")):
        y_blur = physics(gt_batch)
        y_batch=add_poisson_noise(y_blur, alpha_poisson=alpha_poisson)
        init_batch = physics.A_adjoint(y_batch)

        reconstructed_batch, metrics_by_image = model(y=y_batch, GT=gt_batch, log_metrics=args.plot_metrics)
        
        for j in range(gt_batch.shape[0]):
            gt_img = gt_batch[j].unsqueeze(0)
            rec_img = reconstructed_batch[j].unsqueeze(0)

            print(psnr(rec_img, gt_img))
            print(ssim(rec_img, gt_img))
            psnr_values.append(psnr(rec_img, gt_img).item())
            ssim_values.append(ssim(rec_img, gt_img).item())
            lpips_values.append(lpips(rec_img, gt_img).item())
            
            if args.plot_metrics:
                all_metrics_by_image.append(metrics_by_image[j])

            if args.save_images:
                y_img = y_batch[j].unsqueeze(0)
                init_img = init_batch[j].unsqueeze(0)
                save_images(output_path, gt_img.squeeze(0), y_img.squeeze(0), init_img.squeeze(0), rec_img.squeeze(0), f'img_{i*args.batch_size + j}')

    if args.plot_metrics:
        plot_metrics(all_metrics_by_image, output_path, f'metrics_{output_dir_name}')

    if args.save_results:
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        avg_lpips = np.mean(lpips_values)
        
        results_summary = {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'lpips': avg_lpips
        }
        save_results_to_txt(results_summary, output_path, 'results')

    print("\nInference complete!")
    if args.save_results:
        print(f"Average PSNR: {avg_psnr:.4f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average LPIPS: {avg_lpips:.4f}")
    
if __name__ == '__main__':
    main()
