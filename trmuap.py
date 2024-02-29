import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions import *
from skimage import filters
from skimage.morphology import disk
from strategy import *



debug = False


def get_conv_layers(model):
    '''
    Get all the convolution layers in the network.
    '''
    return [module for module in model.modules() if type(module) == nn.Conv2d]


def l2_layer_loss(model, delta,args,device):
    '''
    Compute the loss of TRM
    '''
    loss = torch.tensor(0.)
    activations = []
    p_activations = []
    deactivations = []
    remove_handles = []

    def check_zero(tensor):
        if tensor.equal(torch.zeros_like(tensor)):
            return False
        else:
            return True
    def activation_recorder_hook(self, input, output):
        activations.append(output)
        return None

    for conv_layer in get_conv_layers(model):
        handle = conv_layer.register_forward_hook(activation_recorder_hook)
        remove_handles.append(handle)

    model.eval()
    model.zero_grad()
    model(delta)

    # unregister hook so activation tensors have no references
    for handle in remove_handles:
        handle.remove()

    #calculation of truncated positive activation
    if args.p_active == True:
        # calculate the number of retained layers
        truncate = int(len(activations)* args.p_rate)
        if truncate <=0 and args.p_rate != 0.0:
            #avoid the zero of the number of the retained layer, i.e. truncated>=1
            truncate += 1
        for i in range(truncate):
            ac_tensor = activations[i].view(-1)
            # activate the positve value like Relu
            ac_tensor = torch.where(ac_tensor>0,ac_tensor,torch.zeros_like(ac_tensor))
            p_activations.append(ac_tensor)

        truncate = int(len(activations)* args.n_rate)

        #calculation of truncated negative activation
        if truncate <=0 and args.n_rate != 0.0:
            truncate += 1
        for i in range(truncate):
            ac_tensor = activations[i].view(-1)
            # activate the negative value contrary to Relu
            ac_tensor = torch.where(ac_tensor > 0, ac_tensor, torch.zeros_like(ac_tensor))
            deactivations.append(ac_tensor)

    else:
        # maximize the positive activation of all layers
        for i in range(len(activations)):
            activations[i] = torch.where(activations[i] > 0, activations[i], torch.zeros_like(activations[i])).to(device)

    #calculate the loss by truncated ratio maximization problem
    if args.p_active == True:
        #add a tiny decement(1e-9) to avoid the zero value of activations
        p_loss = sum(list(map(lambda activation: torch.log(torch.sum(torch.square(activation)) / 2+ 1e-9), p_activations)))
        loss = -p_loss
        n_loss = 0
        #compute the loss of the negative part
        if args.n_active == True:
            n_loss = sum(list(map(lambda deactivation: torch.log(torch.norm(deactivation,2)/2+1e-9 ), deactivations)))
            loss = args.lam * n_loss - p_loss
            # #observe the change of loss
            # if args.loss_show == True:
            #     print(f'positivie loss:{p_loss} negetive loss:{n_loss} loss:{loss}')

            return loss,p_loss,n_loss
        return loss,p_loss,n_loss

    else:
        #calculate the loss of maximizing the positive activation of all layers
        loss = -sum(list(map(lambda activation: torch.log(torch.sum(torch.square(activation)) / 2 + 1e-9), activations)))

        return loss,loss,0.0


def get_fooling_rate(model, delta, data_loader, device):
    """
    Computes the fooling rate of the UAP on the dataset.
    """
    flipped = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(normalize(images))
            _, predicted = torch.max(outputs.data, 1)

            adv_images = torch.add(delta, images).clamp(0, 1)
            adv_outputs = model(normalize(adv_images))

            _, adv_predicted = torch.max(adv_outputs.data, 1)

            total += images.size(0)
            flipped += (predicted != adv_predicted).sum().item()

    return flipped / total


def get_rate_of_saturation(delta, xi):
    """
    Returns the proportion of pixels in delta
    that have reached the max-norm limit xi
    """
    return np.sum(np.equal(np.abs(delta), xi)) / np.size(delta)


def get_gauss_prior(args):
    '''
    The Gaussian noise is used
    as range-prior to simulate the real image.
    '''

    for i in range(args.prior_batch):
        im = None
        if args.prior == 'gauss':
            im = make_some_noise_gauss(args.std,args.delta_size)
        # elif args.range_prior == 'uniform':
        #     im = make_some_noise_uniform(args.std,args.delta_size)
        else:
            return None
            #im = make_some_noise_uniform(args.std)
            #im = make_cifar10_noise(args.std)
        # if prior_path == None and im == None:
        #     return None
        prior = img_preprocess(im = im,size=args.delta_size,augment=True)
        prior = np.moveaxis(prior, -1, 1)/255
        prior = torch.Tensor(prior)#.unsqueeze(0)
        if i == 0:
            prior_batch = prior
        else:
            prior_batch = torch.cat([prior_batch, prior], dim=0)


    return prior_batch

def get_jigsaw(img,args,min=0,max=256,filter=False):
    img_shape = torch.zeros_like(img.cpu().detach()).squeeze(0)
    img_batch = torch.zeros_like(img.cpu().detach()).squeeze(0)

    for j in range(args.prior_batch):
        #googlenet used the set of args.fre+2 nad args.fre for the jigsaw image
        ximg = shuffle(img_shape, args.fre, args.fre,min,max)

        if filter == True:
            ximg = ximg.numpy()
            for i in range(len(ximg)):
                ximg[i] = filters.median(ximg[i], disk(5))
            ximg = torch.Tensor(ximg)
        ximg = ximg.unsqueeze(0)  # .to(device)
        ximg = ximg / 255
        if j == 0:
            img_batch = ximg
        else:
            img_batch = torch.cat([img_batch, ximg], dim=0)
    return img_batch




def truncated_ratio_maximization(model, args, device, prior=False):
    """
    Compute the UAP with the truncated ratio maximization.
    Return a single UAP tensor.
    """

    max_iter = 10000
    size = args.delta_size

    sat_threshold = 0.00001
    sat_prev = 0
    sat = 0
    sat_change = 0
    sat_min = 0.5
    sat_should_rescale = False

    iter_since_last_fooling = 0
    iter_since_last_best = 0
    best_fooling_rate = 0
    iter_num = 0

    xi_min = -args.epsilon
    xi_max = args.epsilon
    args.std = 10


    delta = (xi_min - xi_max) * torch.rand((1, 3, size, size), device=device) + xi_max
    delta.requires_grad = True

    print(f"Initial norm: {torch.norm(delta, p=np.inf)}")

    optimizer = optim.Adam([delta], lr=args.uap_lr)

    val_loader,_ = get_data_loader(args.val_dataset_name, batch_size=args.batch_size)#,shuffle=True


    for i in tqdm(range(max_iter)):
        iter_num +=1
        iter_since_last_fooling += 1
        optimizer.zero_grad()
        # Sample artifical images from gaussian or jigsaw distribtuion
        if prior != None:


            if prior == 'gauss':

                args = curriculum_strategy_gauss(iter_num,args)
                random_batch = get_gauss_prior(args=args)


            elif prior == 'jigsaw':


                if args.surrogate_model == 'resnet152':
                    args = curriculum_strategy_jigsaw_resnet152(iter_num,args)
                elif args.surrogate_model == 'googlenet':
                    args = curriculum_strategy_jigsaw_googlenet(iter_num,args)
                else:
                    args = curriculum_strategy_jigsaw(iter_num,args)


                random_batch = get_jigsaw(delta,args,filter=True)



            if random_batch!=None:
                example_prior = delta + random_batch.to(device)
            else:
                example_prior = delta
        else:
            example_prior = delta


        loss,p_loss,n_loss= l2_layer_loss(model, example_prior,args,device)
        loss.backward()

        # args.loss_show = False
        # if iter_since_last_fooling %50 == 0:
        #     args.loss_show = True

        optimizer.step()

        # Clip the UAP to satisfy the restrain of the infinite norm
        with torch.no_grad():
            delta.clamp_(xi_min, xi_max)

        # Compute rate of saturation on a clamped UAP
        sat_prev = np.copy(sat)
        sat = get_rate_of_saturation(delta.cpu().detach().numpy(), xi_max)
        sat_change = np.abs(sat - sat_prev)

        if sat_change < sat_threshold and sat > sat_min:
            if debug:
                print(f"Saturated delta in iter {i} with {sat} > {sat_min}\nChange in saturation: {sat_change} < {sat_threshold}\n")
            sat_should_rescale = True

        # fooling rate is measured every 200 iterations if saturation threshold is crossed
        # otherwise, fooling rate is measured every 400 iterations
        if iter_since_last_fooling > 400 or (sat_should_rescale and iter_since_last_fooling > 200):
            iter_since_last_fooling = 0

            print("\nGetting latest fooling rate...")

            current_fooling_rate = get_fooling_rate(model, torch.clamp(delta,xi_min,xi_max), val_loader, device)
            print(f"\nLatest fooling rate: {current_fooling_rate}")

            if current_fooling_rate > best_fooling_rate:
                print(f"Best fooling rate thus far: {current_fooling_rate}")
                best_fooling_rate = current_fooling_rate
                #best_uap = delta
            else:
                iter_since_last_best += 1

            # if the best fooling rate has not been overcome after patience_interval iterations
            # then training is considered complete
            if iter_since_last_best >= args.patience_interval:
                break

        if sat_should_rescale:
            #if the UAP is saturated, then compress it
            with torch.no_grad():
                delta.data = delta.data / 2
            sat_should_rescale = False

    return delta
