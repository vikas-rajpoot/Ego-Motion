import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

import numpy as np
import sys
sys.path.append('./common/') 

import utils.custom_transforms as custom_transforms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import os 

data = "/home/vk/03/ThermalSfMLearner/ProcessedData" 
sequence = 'indoor_robust_dark' 
workers = 1
batch_size = 1
scene_type = 'indoor'
interval = 1
sequence_length = 3



@torch.no_grad()
def main():
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

    from dataloader.VIVID_validation_folders import ValidationSet 
    
    val_set = ValidationSet( 
        data,
        tf_thr       = valid_tf_thr,
        sequence_length = sequence_length,
        interval     = interval,
        scene_type   = scene_type,        
        inference_folder=sequence,
    ) 

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    import matplotlib.pyplot as plt

    for idx, (tgt_thr_img, depth_thr) in enumerate(val_loader):
        # Convert tensor to numpy array 
        tgt_thr_img_np = tgt_thr_img.squeeze().cpu().numpy()
        depth_thr_np = depth_thr.squeeze().cpu().numpy() 

        # Visualize the thermal image 
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(tgt_thr_img_np, cmap='hot')
        plt.title('Thermal Image')
        plt.colorbar()

        # Visualize the depth map 
        plt.subplot(1, 2, 2) 
        plt.imshow(depth_thr_np, cmap='plasma') 
        plt.title('Depth Map') 
        plt.colorbar() 

        plt.show() 


if __name__ == '__main__': 
    main() 





