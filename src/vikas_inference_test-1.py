import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from imageio import imsave
import numpy as np
from path import Path

import sys
sys.path.append('./common/')

import models 
import utils.custom_transforms as custom_transforms
from utils.utils import tensor2array


data = "/home/vk/03/ThermalSfMLearner/ProcessedData"
sequence = 'indoor_robust_dark'
output_dir = 'output'
img_exts = 'jpg'
resnet_layers = 18
workers = 4
batch_size = 1
scene_type = 'indoor'
interval = 1
sequence_length = 3
pretrained_disp = "/home/vk/03/ThermalSfMLearner/checkpoints/vivid_resnet18_indoor/dispnet_disp_model_best.pth.tar"



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



@torch.no_grad()
def main():
    
    print("=> creating model")
    disp_net = models.DispResNet(resnet_layers, False, num_channel=1).to(device)
    
    # load parameters 
    print("=> using pre-trained weights for DispResNet") 
    weights = torch.load(pretrained_disp, map_location=torch.device('cpu'))  
    disp_net.load_state_dict(weights['state_dict'], strict=False) 
    disp_net.eval()  

    tgt_thr_img = tgt_thr_img.to(device) 

    # compute output
    output_disp = disp_net(tgt_thr_img)
    output_depth = 1/output_disp
    output_depth = output_depth[:, 0]

    # Convert depth tensor to numpy array
    depth_numpy = output_depth.cpu().numpy() 
    
    depth_numpy = np.squeeze(depth_numpy) # Remove the channel dimension 

    np.savetxt("./vikas/data.csv", depth_numpy, delimiter=',') 


if __name__ == '__main__':
    main()
