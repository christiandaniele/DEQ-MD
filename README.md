# DEQ-MD
Repository for the paper: Deep Equilibrium models for Poisson inverse problems via Mirror Descent
https://arxiv.org/abs/2507.11461

Before starting
```bash
pip install -r requirements.txt
```

To train the model
```bash
python train.py --kernel_type Gauss --noise_level medium --model DEQ-RED
```

--kernel_type can be 'Gauss', 'Motion_7', 'Uniform' or you can add your own
and --noise_levl can be 'high', 'medium' or 'low.

To test the model 
```bash
python inference.py --kernel Gauss --noise_level medium --regularisation RED 
```
You can use the optional flags --plot_metrics, --save_images, and --save_results to plot performance metrics per iteration, save the ground truth and reconstructed images, and store the average results in a text file, respectively
