import torch
import random
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    # set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed (seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def inv_rat_loss(inv_logits, envs, labels):
    """
    Compute the loss for the invariant rationalization training.
    Inputs:
        env_inv_logits -- logits of the predictor without env index
                          (batch_size, num_classes)
        env_enable_logits -- logits of the predictor with env index
                          (batch_size, num_classes)        
        labels -- the groundtruth one-hot labels 
    """

    criterion = nn.CrossEntropyLoss(reduction='none')

    inv_losses = criterion(inv_logits, labels)
    losses = []
    for i in range (envs.shape[-1]):
        ind = envs[:,i]==1
        if not torch.any(ind):
          continue
        losses.append(torch.mean(inv_losses[ind]))
    var_loss = torch.var(torch.cat([x.unsqueeze(0) for x in losses]))

    total_loss = torch.mean(torch.cat([x.unsqueeze(0) for x in losses]))
    return total_loss, var_loss

def variance_loss (logits1, logits2, labels):
  criterion = nn.CrossEntropyLoss(reduction='none')
  losses = [criterion(logits1, labels) , criterion(logits2, labels)]

  means_of_var = torch.mean((losses[0]-losses[1])**2)

  return  means_of_var

def KL_loss (logits1, logits2):
  p2 = F.softmax(logits2, dim=-1)
  p1 = F.softmax (logits1, dim = -1)

  return   (p2 * (p2 / p1).log()).sum()/p1.shape[0]

def cal_sparsity_loss(z):
    """
    Exact sparsity loss in a batchwise sense.
    Inputs:
        z -- (batch_size, sequence_length)
        mask -- (batch_size, seq_length)
        level -- sparsity level
    """

    sparsity = torch.mean(torch.sum(z, dim=-1))
    return sparsity

def visualize (path, model, image, rationales, logits, target, patch_size=16, image_size = 64, show_cam = False, label_map = None, id='image0'):
    num_patches = image_size // patch_size
    temp = rearrange(image, 'b c (h1 h2) (w1 w2) -> b h1 w1 (c h2 w2)', w2=patch_size, h2=patch_size) ### w2 and h2 are patch_size 

    rationales = rationales.reshape(-1,num_patches,num_patches,1)
    selected = temp*rationales

    masked = rearrange(selected, 'b h1 w1 (c h2 w2) -> b c (h1 h2) (w1 w2) ', w2=patch_size, h2=patch_size) ### w2 and h2 are patch_size


    if show_cam:
        target_layers = [model.layers[2]]
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        grayscale_cam = cam(input_tensor=image)
        grayscale_cam = grayscale_cam[0, :]
        im_visualization = show_cam_on_image(torch.squeeze(image).permute(1,2,0).cpu().detach().numpy(), grayscale_cam, use_rgb=True)

        grayscale_cam = cam(input_tensor=masked)
        grayscale_cam = grayscale_cam[0, :]
        masked_visualization = show_cam_on_image(torch.squeeze(masked).permute(1,2,0).cpu().detach().numpy(), grayscale_cam, use_rgb=True)

        im_visualization, masked_visualization = im_visualization/255.0,  masked_visualization/255.0,
        im = np.concatenate ((torch.squeeze(image).permute([1,2,0]).cpu().detach(), torch.squeeze(masked).permute([1,2,0]).cpu().detach(), im_visualization, masked_visualization), axis = 1)

    if not show_cam:
        im = np.concatenate((torch.squeeze(image).permute([1,2,0]).cpu().detach(), torch.squeeze(masked).permute([1,2,0]).cpu().detach()), axis = 1)

    
    pred = torch.argmax(logits).item()

    plt.axis('off')
    if label_map!=None:
        plt.imsave(os.path.join(path, id+'_'+label_map[target]+'_'+label_map[pred]+'.png') ,im)

    if label_map==None:
        plt.imsave(os.path.join(path, id+'_'+str(target)+'_'+str(pred)+'.png'), im)


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                        "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                        -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    pe = rearrange(pe, 'd w h -> (w h) d')

    return pe.to(device)

