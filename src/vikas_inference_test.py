import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from imageio import imsave
import numpy as np
from path import Path
import argparse

import sys
sys.path.append('./common/')

import models 
import utils.custom_transforms as custom_transforms
from utils.utils import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)



# parser.add_argument('data', metavar='DIR', help='path to dataset') 
# parser.add_argument("--sequence", default='.', type=str, help="Dataset directory") 
# parser.add_argument("--output-dir", default='output', type=str, help="Output directory") 
# parser.add_argument("--img-exts", default='jpg', choices=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob") 

# parser.add_argument('--resnet-layers', required=True, type=int, default=18, choices=[18, 50], 
#                     help='depth network architecture.') 
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers') 
# parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size') 

# parser.add_argument('--scene_type', type=str, choices=['indoor', 'outdoor'], default='indoor', required=True) 
# parser.add_argument('--interval', type=int, help='Interval of sequence', metavar='N', default=1) 
# parser.add_argument('--sequence_length', type=int, help='Length of sequence', metavar='N', default=3) 
# parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model') 




parser.add_argument('--data', default="/home/vk/03/ThermalSfMLearner/ProcessedData", metavar='DIR', help='path to dataset')
parser.add_argument("--sequence", default='indoor_robust_dark', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")
parser.add_argument("--img-exts", default='jpg', choices=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

parser.add_argument('--resnet-layers', required=False, type=int, default=18, choices=[18, 50],
                    help='depth network architecture.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')

parser.add_argument('--scene_type', type=str, choices=['indoor', 'outdoor'], default='indoor', required=False)
parser.add_argument('--interval', type=int, help='Interval of sequence', metavar='N', default=1)
parser.add_argument('--sequence_length', type=int, help='Length of sequence', metavar='N', default=3)
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default="/home/vk/03/ThermalSfMLearner/checkpoints/vivid_resnet18_indoor/dispnet_disp_model_best.pth.tar", 
                    metavar='PATH', help='path to pre-trained dispnet model' ) 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import os

@torch.no_grad()
def main():
    args = parser.parse_args()

    # 1. data loader 
    if args.scene_type == 'indoor':  # indoor
        args.temp_min = 10
        args.temp_max = 40
        args.depth_max = 4
    elif args.scene_type == 'outdoor': # outdoor
        args.temp_min = 0
        args.temp_max = 30
        args.depth_max = 10

    ArrToTen_thr = custom_transforms.ArrayToTensor_Thermal(args.temp_min, args.temp_max)
    normalize    = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    valid_tf_thr = custom_transforms.Compose([ArrToTen_thr, normalize])

    from dataloader.VIVID_validation_folders import ValidationSet 
    
    val_set = ValidationSet( 
        args.data,
        tf_thr       = valid_tf_thr,
        sequence_length = args.sequence_length,
        interval     = args.interval,
        scene_type   = args.scene_type,        
        inference_folder=args.sequence,
    ) 

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # 1. Load models
    # create model
    print("=> creating model")
    disp_net = models.DispResNet(args.resnet_layers, False, num_channel=1).to(device)
    
    # load parameters 
    print("=> using pre-trained weights for DispResNet") 
    weights = torch.load(args.pretrained_disp, map_location=torch.device('cpu'))  
    disp_net.load_state_dict(weights['state_dict'], strict=False) 
    disp_net.eval()  

    # 2. Load dataset
    output_dir = Path(args.output_dir+'/'+args.sequence)
    output_dir.makedirs_p()
    (output_dir/'thr_depth').makedirs_p()
    (output_dir/'thr_disp').makedirs_p()
    (output_dir/'thr_depth_csv').makedirs_p()  # Create folder for depth CSV files

    disp_net.eval()

    for idx, (tgt_thr_img, depth_thr) in enumerate(val_loader):
        # original validate_with_gt param  
        tgt_thr_img = tgt_thr_img.to(device)
        # Save the tgt_thr_img
        img_file_name = '{:06d}_tgt_thr_img.npy'.format(idx)
        np.save(output_dir/'thr_depth_csv'/img_file_name, tgt_thr_img.cpu().numpy())

        # Visualize the tgt_thr_img
        import matplotlib.pyplot as plt
        img_np = tgt_thr_img.cpu().numpy().squeeze()
        plt.imshow(img_np, cmap='hot')
        plt.colorbar()
        plt.title('Thermal Image')
        plt.show()

        # compute output
        output_disp = disp_net(tgt_thr_img)
        output_depth = 1/output_disp
        output_depth = output_depth[:, 0]

        # Convert depth tensor to numpy array
        depth_numpy = output_depth.cpu().numpy() 
        
        depth_numpy = np.squeeze(depth_numpy) # Remove the channel dimension 

        # Save each depth array as a .csv file
        csv_file_name = '{:06d}.csv'.format(idx)
        np.savetxt(output_dir/'thr_depth_csv'/csv_file_name, depth_numpy, delimiter=',')

        tgt_thr_disp = np.transpose(255*tensor2array(output_disp, max_value=None, colormap='magma'), (1,2,0)).astype(np.uint8)[:,:,:-1]
        tgt_thr_depth = np.transpose(255*tensor2array(output_depth, max_value=args.depth_max, colormap='rainbow'), (1,2,0)).astype(np.uint8)[:,:,:-1]

        # Save images 
        file_name = '{:06d}'.format(idx)
        imsave(output_dir/'thr_depth'/'{}.{}'.format(file_name, args.img_exts), tgt_thr_depth)
        imsave(output_dir/'thr_disp'/'{}.{}'.format(file_name, args.img_exts), tgt_thr_disp)

    print(f"Depth CSV files saved in {output_dir/'thr_depth_csv'}")

if __name__ == '__main__':
    main()




