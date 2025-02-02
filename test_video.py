from imageio import imread 
import numpy as np 
import sys
import torch
sys.path.append('./common/')
import utils.custom_transforms as custom_transforms 
import models 
from evaluate_depth import evaluate_depth  # Import the evaluation function

resnet_layers = 18 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
pretrained_disp = "/home/vk/03/ThermalSfMLearner/checkpoints/vivid_resnet18_indoor/dispnet_disp_model_best.pth.tar" 
scene_type = 'indoor' 
import matplotlib.pyplot as plt
import os


def predict(path): 
    thermal_image = np.expand_dims(imread(path).astype(np.float32),  axis=2)     
    
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


def main():
    folder = "/home/vk/03/ThermalSfMLearner/ProcessedData/indoor_robust_local/Thermal"
    count = 0 
    path_list_1 = sorted(os.listdir(folder))  
    
    print(path_list_1) 
     
    for file_name in path_list_1:
        count += 1 
        file_path = os.path.join(folder, file_name)
        output_depth = predict(file_path)    
        
        plt.figure(figsize=(10, 10)) 
        fig, ax = plt.subplots()
        ax.imshow(output_depth, cmap='viridis')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig('data/output_depth/output_image'+ str(count) +'.png', bbox_inches='tight', pad_inches=0) 
        plt.close(fig)  
        

if __name__ == '__main__': 
    main()




