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

To test the model 
```bash
python inference.py --kernel Gauss --noise_level medium --regularisation RED --save_images
```
