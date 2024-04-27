import argparse

import matplotlib.pyplot as plt
from torchvision import models
import torch
from trmuap import *
from functions import validate_arguments

import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import os



download_path = 'TorchHub/'
torch.hub.set_dir(download_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surrogate_model', default='vgg19',
                        help='The substitute network eg. vgg19')
    parser.add_argument('--target_model', default='vgg19',
                        help='The target model eg. vgg19')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size to use for training and testing')
    parser.add_argument('--patience_interval', type=int, default=5,
                        help='The number of iterations to wait to verify convergence')
    parser.add_argument('--val_dataset_name', default='imagenet',choices=['imagenet'],
                        help='The dataset to be used as test')

    parser.add_argument('--p_active', action="store_true",
                        help='maximize the positive activation the conv layer')
    parser.add_argument('--p_rate', default=0.8, type=float,
                        help='positive proportion of conv layer used')
    parser.add_argument('--n_active', action="store_true",
                        help='minimize the negative activation the conv layer')
    parser.add_argument('--n_rate', default=0.8, type=float,
                        help='negative proportion of conv layer used(deactivation)')

    parser.add_argument('--seed', default=123, type=int,
                        help='random seed')
    parser.add_argument('--lam', default=1, type=float,
                        help='the parameter of negative activation')
    parser.add_argument('--epsilon', default=10/255, type=float,
                        help='the infinite norm limitation of UAP')
    parser.add_argument('--delta_size', default=224, type=int,
                        help='the size of delta')
    parser.add_argument('--uap_lr', default=0.1, type=float
                        help='the leraning rate of UAP')

    parser.add_argument('--prior', default='gauss',choices=['gauss','jigsaw','None'], type='str',
                        help='the range prior of perturbations')
    parser.add_argument('--prior_batch', default=1, type=int,
                        help='the batch size of prior')
    parser.add_argument('--std', default=10, type=int,
                        help='initialize the standard deviation of gaussian noise')
    parser.add_argument('--fre', default=1, type=int,
                        help='initialize the frequency of jigsaw image')
    parser.add_argument('--uap_path', default=None, type=str,
                        help='the path of UAP')
    parser.add_argument('--gauss_t0', default=400, type=int,
                        help='the threshold to adjust the increasing rate of standard deviation(gauss)')
    parser.add_argument('--gauss_gamma', default=10, type=int,
                        help='the step size(gauss)')
    parser.add_argument('--jigsaw_t0', default=600, type=int,
                        help='the threshold to adjust the increasing rate of standard deviation(jigsaw)')
    parser.add_argument('--jigsaw_gamma', default=1, type=int,
                        help='the step size(jigsaw)')
    parser.add_argument('--jigsaw_end_iter', default=4200, type=int,
                        help='the iterations which stop the increment of frequency(jigsaw)')

    #best set-up p_rate, n_rate, lambda, prior strategy t0 gamma in alexnet and vgg
    #alexnet 1.0,0.2,1.0, [gauss(std=10) 400,10]
    #vgg16 0.8 0.7 1.0, [gauss(std=10) 400,2]
    #vgg19 0.8 0.8 0.5, [gauss(std=10) 400,10]

    #best set-up p_rate, n_rate, lambda, prior strategy in resnet152 and googlenet
    #resnet152 0.3 0.2 1.5 [jigsaw for resnet152]
    #googlnet 0.65 0.2 1.0 [jigsaw for googlenet]
    #Note: Due to the randomness of artificial images, the results in our paper may be reproduced after conducting repeated experiments with different seeds. 


    args = parser.parse_args()
    validate_arguments(args.surrogate_model)
    validate_arguments(args.target_model)

    if torch.cuda.is_available():
        # set the random seed
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cudnn.benchmark = True


    # perpare for the surrogate model and target model
    model = prepare_for_model(args,args.surrogate_model,device,initialize=True)
    target_model = prepare_for_model(args,args.target_model,device,initialize=False)

    # craft UAP  by truncated ratio maximization or loading it from the file
    if args.uap_path == None:
        uap = truncated_ratio_maximization(model, args, device,prior=args.prior)
        filename = f"perturbations/uap_{args.surrogate_model}_dataset={args.val_dataset_name}_p_rate={args.p_rate}" \
                   f"_n_rate={args.n_rate}_seed={args.seed}_lambda={args.lam}_prior={args.prior}"
        np.save(filename, uap.cpu().detach().numpy())
        print(f'the UAP of surrogate model {args.surrogate_model} is crfted.')
    else:
        uap = get_uap(args.uap_path,device)


    test_loader = get_data_loader(args.val_dataset_name, batch_size=args.batch_size, shuffle=True, analyze=True)

    final_fooling_rate = get_fooling_rate(target_model,torch.clamp(uap,-args.epsilon,args.epsilon),test_loader,device)
    print(f'the FR of UAP ({args.surrogate_model}) on ({args.target_model}) is {final_fooling_rate}')

    print('finish')





if __name__ == '__main__':
    main()
