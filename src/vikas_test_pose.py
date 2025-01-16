import torch
from skimage.transform import resize as imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import pandas as pd 
import sys
sys.path.append('./common/')
import models
from loss.inverse_warp import pose_vec2mat
from utils.custom_transforms import Celsius2Raw


pretrained_posenet = "/home/vk/03/ThermalSfMLearner/checkpoints/THERMAL-SFM-LEARNER/vivid_rgbt_indoor/exp_pose_pose_model_best.pth.tar" 
img_height = 256 
img_width = 320 
no_resize = False 
dataset_dir = "/home/vk/03/ThermalSfMLearner/ProcessedData"
sequence_length = 5  

sequences = ['indoor_robust_global']  
output_dir = None 
resnet_layers = 18 
input = "T" 
scene_type = "indoor"


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_tensor_image(img):
    h,w,_ = img.shape
    if (h != img_height or w != img_width):
        img = imresize(img, (img_height, img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(device)
    return tensor_img

def load_tensor_Timage_indoor(img):
    h,w,_ = img.shape
    if (h != img_height or w != img_width):
        img = imresize(img, (img_height, img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    Dmin = Celsius2Raw(10)
    Dmax = Celsius2Raw(40)
    img[img<Dmin] = Dmin
    img[img>Dmax] = Dmax
    img = (torch.from_numpy(img).float() - Dmin)/(Dmax - Dmin) 
    tensor_img = ((img.unsqueeze(0)-0.45)/0.225).to(device)
    return tensor_img

def load_tensor_Timage_outdoor(img ):
    h,w,_ = img.shape
    if (h != img_height or w != img_width):
        img = imresize(img, (img_height, img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    Dmin = Celsius2Raw(0)
    Dmax = Celsius2Raw(30)
    img[img<Dmin] = Dmin
    img[img>Dmax] = Dmax
    img = (torch.from_numpy(img).float() - Dmin)/(Dmax - Dmin) 
    tensor_img = ((img.unsqueeze(0)-0.45)/0.225).to(device)
    return tensor_img


@torch.no_grad()
def main():
    # only for the thermal images. 
    pose_net = models.PoseResNet(resnet_layers, False, num_channel=1).to(device)

    weights = torch.load(pretrained_posenet, map_location=torch.device('cpu')) 
    pose_net.load_state_dict(weights['state_dict'], strict=False)
    pose_net.eval() 

    seq_length = 5 

    global dataset_dir, output_dir, sequences, input 
    # only for the thermal images.
    load_tensor_img = load_tensor_Timage_indoor

    # load data loader 
    from eval_vivid.pose_evaluation_utils import test_framework_VIVID as test_framework
    dataset_dir = Path(dataset_dir) 
    framework = test_framework(dataset_dir, sequences, seq_length=seq_length, step=1, input_type=input)

    print('{} snippets to test'.format(len(framework))) 
    errors = np.zeros((len(framework), 2), np.float32)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.makedirs_p()
        predictions_array = np.zeros((len(framework), seq_length, 3, 4))

    df_pose = pd.DataFrame(columns=['tx', 'ty', 'tz', 'rx', 'ry', 'rz']) 

    df_global_pose = pd.DataFrame(columns=['frame', 'tx', 'ty', 'tz', 'r11', 'r12', 'r13', 
                                        'r21', 'r22', 'r23', 'r31', 'r32', 'r33'])

    for j, sample in enumerate(tqdm(framework)):
        imgs = sample['imgs']
        squence_imgs = []
        for i, img in enumerate(imgs):
            img = load_tensor_img(img) 
            squence_imgs.append(img)

        global_pose = np.eye(4)
        poses = []
        poses.append(global_pose[0:3, :])
        

        # Initialize a DataFrame to save poses



        for iter in range(seq_length - 1): 
            pose = pose_net(squence_imgs[iter], squence_imgs[iter + 1])  
            # pose return by this is in format to tx, ty, tz, rx, ry, rz. 
            pose_1 = pose.unsqueeze(1).tolist()[0][0] 

            df_pose.loc[len(df_pose)] = pose_1             
            
            # print("\033[92m pose_1 shape : \33[0m", pose_1) 
            # print("\033[92m pose_1 shape : \33[0m", len(pose_1))  
            
            pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy() 
            # print("\033[92m pose_mat : ", pose_mat, "\033[0m")   
            
            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])]) 
            global_pose = global_pose @  np.linalg.inv(pose_mat) 
            poses.append(global_pose[0:3, :])  
            
            # Save the global_pose matrix
            translation = global_pose[:3, 3]
            rotation = global_pose[:3, :3].flatten()
            df_global_pose.loc[len(df_global_pose)] = [iter] + translation.tolist() + rotation.tolist()


        final_poses = np.stack(poses, axis=0) 
        # print("\033[92m [INFO] \033[0m final_poses : ", final_poses.shape)  
        
        if output_dir is not None:
            predictions_array[j] = final_poses

        ATE, RE = compute_pose_error(sample['poses'], final_poses)
        errors[j] = ATE, RE

    # Save the global poses to a CSV file
    df_global_pose.to_csv("./vikas_data/global_pose.csv", index=False)

    df_pose.to_csv("./vikas_data/pose.csv", index=False)  
    
    
    
    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE', 'RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))

    print("\033[92m [INFO] \033[0m COMPLETED") 
    
    if output_dir is not None:
        np.save(output_dir/'predictions.npy', predictions_array)


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:, :, -1] * pred[:, :, -1])/np.sum(pred[:, :, -1] ** 2)
    ATE = np.linalg.norm((gt[:, :, -1] - scale_factor * pred[:, :, -1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:, :3] @ np.linalg.inv(pred_pose[:, :3])
        s = np.linalg.norm([R[0, 1]-R[1, 0],
                            R[1, 2]-R[2, 1],
                            R[0, 2]-R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return ATE/snippet_length, RE/snippet_length


def compute_pose(pose_net, tgt_img, ref_imgs):
    poses = []
    for ref_img in ref_imgs:
        pose = pose_net(tgt_img, ref_img).unsqueeze(1)
        poses.append(pose)
    poses = torch.cat(poses, 1)
    return poses


if __name__ == '__main__':
    main()






