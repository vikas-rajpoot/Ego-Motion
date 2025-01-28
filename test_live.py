from imageio import imread 
import numpy as np 
import sys
import torch
sys.path.append('./common/')
# sys.path.append('/home/vk/03/ThermalSfMLearner')  
import utils.custom_transforms as custom_transforms 
import models 
from evaluate_depth import evaluate_depth  # Import the evaluation function

import cv2
import numpy as np
from flirpy.camera.lepton import Lepton
from utils_live import images_to_video 

resnet_layers = 18 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
pretrained_disp = "/home/vk/03/ThermalSfMLearner/checkpoints/vivid_resnet18_indoor/dispnet_disp_model_best.pth.tar" 
scene_type = 'indoor' 
import matplotlib.pyplot as plt
import os


def predict(thermal_image): 
    
    print(thermal_image.shape) 
    
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

    thermal_image_1, _ = valid_tf_thr([thermal_image], None) 
    thermal_image_2 = thermal_image_1[0]  
    
    thermal_image_2 = torch.nn.functional.interpolate(thermal_image_2.unsqueeze(0), size=(256, 320), mode='bilinear', align_corners=False) 


    disp_net = models.DispResNet(resnet_layers, False, num_channel=1).to(device)
    weights = torch.load(pretrained_disp, map_location=torch.device('cpu'))  
    disp_net.load_state_dict(weights['state_dict'], strict=False) 
    disp_net.eval()  


    thermal_image_2 = thermal_image_2.to(device) 

    # compute output 
    with torch.no_grad():
        output_disp = disp_net(thermal_image_2) 
        
    output_depth = 1/output_disp
    output_depth = output_depth[0, 0].cpu().numpy() 
    
    # Save output_depth to a CSV file
    np.savetxt("data/output_depth.csv", output_depth, delimiter=",")
    
    # Evaluate the predicted depth with the original depth
    # mae = evaluate_depth(output_depth, depth_image)
    # print(f"\033[92mMean Absolute Error (MAE): {mae}\033[00m")
    # print(f"\033[93mRoot Mean Squared Error (RMSE): {rmse}\033[00m")
    # visualize images 
    
    
    return output_depth 
def raw_to_celsius( raw_data):
    SCALE_FACTOR = 0.01
    OFFSET = 273.15
    return (raw_data * SCALE_FACTOR) - OFFSET
def main():
    camera = Lepton() 

    count = 0
    path_imgs = 'data/output_depth_live' 
    while True:
        count += 1 
        thermal_image = camera.grab().astype(np.float32)
        thermal_image_celsius = raw_to_celsius(thermal_image)
        # thermal_image_celsius_clipped = np.clip(thermal_image_celsius, self.min_temp, self.max_temp)

        normalized_img = cv2.normalize(
            thermal_image_celsius, None, 0, 255, cv2.NORM_MINMAX
        )
        normalized_img = np.uint8(normalized_img)
        color_mapped_img = cv2.applyColorMap(normalized_img, cv2.COLORMAP_JET)
        
        # # print("\033[92m [INFO] \033[0m Thermal camera live feed started") 
        # print("\033[92m [INFO] \033[0m image:  ", thermal_image)  
        # print("\033[92m [INFO] \033[0m Shape : ", thermal_image.shape)   
        # print("\033[92m [INFO] \033[0m type: ", type(thermal_image))    
        
        # print("\033[92m [INFO] \033[0m Press 'ESC' to exit")   
        
        thermal_image_1 = thermal_image[:, :, np.newaxis]
        
        # output_depth = predict(thermal_image_1)   
        
        plt.figure(figsize=(10, 10)) 
        fig, ax = plt.subplots()
        # ax.imshow(output_depth, cmap='viridis')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # plt.savefig('data/output_depth_live/output_image'+ str(count) +'.png', bbox_inches='tight', pad_inches=0)  
        plt.close(fig)  
        
        
        # thermal_image = cv2.resize(thermal_image, (1080,720), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('thermal_image',color_mapped_img) 
        
        
        if count == 250:
            break 
        
        if cv2.waitKey(1) & 0xFF == 27:
            break 
        
        # break 
        
        
    camera.close() 

    images_to_video(path_imgs, 'data/output_video_live.mp4', fps=10)    
    

if __name__ == '__main__': 
    main()




