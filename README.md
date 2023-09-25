# Truncated Ratio Maximization UAP
[[Paper]](https://iccv2023.thecvf.com/main.conference.program-107.php) [[Poster]](https://drive.google.com/file/d/16ljA-MjlF8dHHp5NVcHUtUjFX1u7HI8B/view) [[Presentation]](https://drive.google.com/file/d/16Rdu6pGuSuaK14H1MK7acxatkHMjj_oL/view)

Code for the method **TRM-UAP: Enhancing the Transferability of Data-Free Universal Adversarial Perturbation via Truncated Ratio Maximization (ICCV2023)**. A universal attack to craft the universal adversarial perturbation (UAP) via truncated ratio maximization. This code depends on PyTorch.

## Dependencies

This repo is tested with pytorch<=1.12.0, python<=3.6.13.
Install all python packages using following command:
```
pip install -r requirements.txt
```

## Usage Instructions

### 1. Preparation

ImageNet validation set:
   Load the parameters of pretrained models with PyTorch, download ImageNet dataset from (https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
- `TorchHub` : the directory saves PyTorch pretrained model parameters.
- `dataset` : the directory contains the datasets.
- `perturbations` : the directory stores the UAP crafted by universal attacks. 



### 2. Training

For example,run the following command:

```
python train.py --surrogate_model vgg16 --target_model vgg19 --val_dataset_name imagenet 
                --p_active True --n_active True --p_rate 0.8 --n_rate 0.7
```
This will start a training to craft a UAP from the surrogate model vgg16 and attack the target model vgg19 on ImageNet with the positive and negative truncated activations correspondingly.


### 3. Testing
After a UAP is generated and saved on the directory `perturbations`, you can also load the UAP to attack other models:
```
python attack_test.py --test_model vgg19 --val_dataset_name imagenet --uap_path perturbations/uap_vgg16.npy
```
This will load the UAP made by vgg16 from `perturbations` and attack the target model vgg19 on imagenet.


## Acknowledgements
The code refers to [pytorch-gd-uap](https://github.com/psandovalsegura/pytorch-gd-uap)

We thank the author for sharing.

## Citation
If you find this work is useful in your research, please cite our paper:
```
@inproceedings{TRMUAP_2023,
  title={TRM-UAP: Enhancing the Transferability of Data-Free Universal Adversarial Perturbation via Truncated Ratio Maximization},
  author={Yiran Liu, Xin Feng, Yunlong Wang, Wu Yang, Di Ming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```
