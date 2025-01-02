from imageio import imread 
import numpy as np 
import sys
import torch
sys.path.append('./common/')

import utils.custom_transforms as custom_transforms 
import models 


resnet_layers = 18 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
pretrained_disp = "/home/vk/03/ThermalSfMLearner/checkpoints/vivid_resnet18_indoor/dispnet_disp_model_best.pth.tar" 
scene_type = 'indoor' 

import matplotlib.pyplot as plt

@torch.no_grad() 
def main():
    path = "/home/vk/03/ThermalSfMLearner/ProcessedData/indoor_robust_dark/Thermal/000004.png" 
    thermal_image = np.expand_dims(imread(path).astype(np.float32),  axis=2)  
    print("\033[92m[INFO]\033[00m Thermal Image Shape: ", thermal_image.shape) 
    
        # 1. data loader  
    if scene_type == 'indoor':  # indoor
        temp_min = 10
        temp_max = 40
        depth_max = 4 
    elif scene_type == 'outdoor': # outdoor
        temp_min = 0
        temp_max = 30
        depth_max = 10

    ArrToTen_thr = custom_transforms.ArrayToTensor_Thermal(temp_min, temp_max)
    normalize    = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    valid_tf_thr = custom_transforms.Compose([ArrToTen_thr, normalize]) 

    thermal_image_1, _ = valid_tf_thr(thermal_image, None) 
    thermal_image_2 = thermal_image_1[0] 

    print("=> creating model")
    disp_net = models.DispResNet(resnet_layers, False, num_channel=1).to(device)
    
    # load parameters 
    print("=> using pre-trained weights for DispResNet") 
    weights = torch.load(pretrained_disp, map_location=torch.device('cpu'))  
    disp_net.load_state_dict(weights['state_dict'], strict=False) 
    disp_net.eval()  

    thermal_image_2 = torch.from_numpy(thermal_image_2).unsqueeze(0).unsqueeze(0).to(device) 

    # compute output
    with torch.no_grad():
        output_disp = disp_net(thermal_image_2)
    output_depth = 1/output_disp
    output_depth = output_depth[0, 0].cpu().numpy()

    # visualize images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Thermal Image")
    plt.imshow(thermal_image_2[0, 0].cpu(), cmap='hot')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Output Depth Image")
    plt.imshow(output_depth, cmap='rainbow') 
    plt.colorbar()

    plt.show()


if __name__ == '__main__': 
    main() 





