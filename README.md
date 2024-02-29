# Truncated Ratio Maximization UAP
[[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_TRM-UAP_Enhancing_the_Transferability_of_Data-Free_Universal_Adversarial_Perturbation_via_ICCV_2023_paper.pdf)] [[Supp](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Liu_TRM-UAP_Enhancing_the_ICCV_2023_supplemental.pdf)] [[Poster](https://drive.google.com/file/d/16ljA-MjlF8dHHp5NVcHUtUjFX1u7HI8B/view)] [[Presentation](https://drive.google.com/file/d/16Rdu6pGuSuaK14H1MK7acxatkHMjj_oL/view)]

Code for the method [**\[ICCV 2023\] "TRM-UAP: Enhancing the Transferability of Data-Free Universal Adversarial Perturbation via Truncated Ratio Maximization", Yiran Liu, Xin Feng, Yunlong Wang, Wu Yang, Di Ming\***](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_TRM-UAP_Enhancing_the_Transferability_of_Data-Free_Universal_Adversarial_Perturbation_via_ICCV_2023_paper.html). 

A data-free universal attack to craft the universal adversarial perturbation (UAP) via truncated ratio maximization. This code depends on PyTorch.

## Update
 - Feb 29, 2024: We updated the curriculum learning-based training strategy in the file `strategy.py` to provide a comprehensive illustration of the optimal experiment setup.

   Besides, the performance of TRM-UAP, as proposed in the paper, could be improved with further exploration of experimental hyperparameters.
## Dependencies

This repo is tested with pytorch<=1.12.0, python<=3.6.13.
Install all python packages using following command:
```
pip install -r requirements.txt
```

## Usage Instructions

### 1. Preparation

ImageNet validation set:
   Load the parameters of pretrained models with PyTorch, download ImageNet dataset from [here](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
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
The code refers to  [GD-UAP](https://github.com/val-iisc/GD-UAP/tree/master), [pytorch-gd-uap](https://github.com/psandovalsegura/pytorch-gd-uap).

We thank the authors for sharing sincerely.

## Citation
If you find this work is useful in your research, please cite our paper:
```
@InProceedings{Liu_2023_ICCV,
    author    = {Liu, Yiran and Feng, Xin and Wang, Yunlong and Yang, Wu and Ming, Di},
    title     = {TRM-UAP: Enhancing the Transferability of Data-Free Universal Adversarial Perturbation via Truncated Ratio Maximization},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {4762-4771}
}
```
## Contact

Yiran Liu: [lyr199804@qq.com](mailto:lyr199804@qq.com)

[Di Ming](https://midasdming.github.io/): [diming@cqut.edu.cn](mailto:diming@cqut.edu.cn)
