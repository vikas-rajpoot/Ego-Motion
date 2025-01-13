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


def main():
    # path = "/home/vk/03/ThermalSfMLearner/ProcessedData/indoor_robust_dark/Thermal/000004.png" 
    # path = "/home/vk/03/ThermalSfMLearner/images/02.png" 
    path = "images/frame_247.png" 
    # path = "/home/vk/03/ThermalSfMLearner/ProcessedData/our_data/frame_0441.jpg" 
    thermal_image = np.expand_dims(imread(path).astype(np.float32),  axis=2)     
    
    print(thermal_image.shape) 
    
    
    array_squeezed = np.squeeze(thermal_image, axis=2)
    np.savetxt("thermal_image.csv", array_squeezed, delimiter=",") 
    
    print("\033[92m[INFO]\033[00m Thermal Image Shape: ", thermal_image.shape)  
    # print("\033[92m[INFO]\033[00m Thermal Image Values: \n", thermal_image) 
    print("\033[92m[INFO]\033[00m Thermal Image Type : ", type(thermal_image))  
    
    path_d = "/home/vk/03/ThermalSfMLearner/ProcessedData/indoor_robust_dark/Depth_T/000004.npy" 
    depth_image = np.load(path_d).astype(np.float32) 
    depth_image = np.resize(depth_image, (256, 320))  # Resize depth_image to match output_depth shape
    print("\033[92m[INFO]\033[00m Depth Image Shape: ", depth_image.shape)
    
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

    thermal_image_1, _ = valid_tf_thr([thermal_image], None) 
    thermal_image_2 = thermal_image_1[0]  
    
    print("\033[92m[INFO]\033[00m Thermal Image 2 Shape: ", thermal_image_2.shape) 
    print("\033[92m[INFO]\033[00m Thermal Image 2 Type: ", type(thermal_image_2))  
    print("\033[92m[INFO]\033[00m Thermal Image 2 Values: \n", thermal_image_2) 
    
    thermal_image_2 = torch.nn.functional.interpolate(thermal_image_2.unsqueeze(0), size=(256, 320), mode='bilinear', align_corners=False) 


    print("=> creating model")  
    disp_net = models.DispResNet(resnet_layers, False, num_channel=1).to(device)
    
    # load parameters 
    print("=> using pre-trained weights for DispResNet") 
    weights = torch.load(pretrained_disp, map_location=torch.device('cpu'))  
    disp_net.load_state_dict(weights['state_dict'], strict=False) 
    disp_net.eval()  


    thermal_image_2 = thermal_image_2.to(device) 

    print("\033[92m[INFO]\033[00m thermal_image_2 Shape: ", thermal_image_2.shape) 

    # compute output 
    with torch.no_grad():
        output_disp = disp_net(thermal_image_2) 
        
    output_depth = 1/output_disp
    output_depth = output_depth[0, 0].cpu().numpy() 
    
    # Save output_depth to a CSV file
    np.savetxt("vikas_data/output_depth.csv", output_depth, delimiter=",")
    
    # Evaluate the predicted depth with the original depth
    # mae = evaluate_depth(output_depth, depth_image)
    # print(f"\033[92mMean Absolute Error (MAE): {mae}\033[00m")
    # print(f"\033[93mRoot Mean Squared Error (RMSE): {rmse}\033[00m")
    # visualize images 
    
    
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Thermal Image")
    plt.imshow(thermal_image_2[0, 0].cpu(), cmap='hot')
    plt.colorbar() 

    plt.subplot(1, 3, 2)
    plt.title("Depth Image")
    plt.imshow(depth_image, cmap='rainbow')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Output Depth Image")
    plt.imshow(output_depth, cmap='rainbow') 
    plt.colorbar()

    plt.savefig("images/output_images.png") 


if __name__ == '__main__': 
    main()

