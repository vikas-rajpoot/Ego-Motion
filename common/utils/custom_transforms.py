from __future__ import division
import torch
import random
import numpy as np
import cv2

# from scipy.misc import imresize
from skimage.transform import resize as imresize

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''

def Celsius2Raw(celcius_degree):
    R = 380747
    B = 1428
    F = 1
    O = -88.539
    raw_value = R / (np.exp(B / (celcius_degree + 273.15)) - F) + O;
    return raw_value

def Raw2Celsius(Raw):
    R = 380747
    B = 1428
    F = 1
    O = -88.539
    Celsius = B / np.log(R / (Raw - O) + F) - 273.15;
    return Celsius


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, intrinsics):
        for t in self.transforms:
            images, intrinsics = t(images, intrinsics)
        return images, intrinsics


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images, intrinsics):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors, intrinsics

class ArrayToTensor_Thermal(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""
    def __init__(self, min_temp, max_temp):
        # indoor : 10,40 / outdoor : 0,30
        self.Dmin = Celsius2Raw(min_temp)
        self.Dmax = Celsius2Raw(max_temp)

    def __call__(self, images, intrinsics):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            im[im<self.Dmin] = self.Dmin
            im[im>self.Dmax] = self.Dmax
            tensors.append((torch.from_numpy(im).float() - self.Dmin)/(self.Dmax - self.Dmin)) # thermal data clip into 30~50 degree clip
        return tensors, intrinsics

class TensorColorize(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""
    def __init__(self):
        import matplotlib.pyplot as plt
        self.cmap = plt.get_cmap('jet')
    
    def __call__(self, images, intrinsics):
        imgs_clr = []
        for im in images:
            im = im.squeeze().numpy()
            im = self.cmap(im)
            im = im[:,:,0:3]
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            imgs_clr.append(torch.from_numpy(im).float())
        return imgs_clr, intrinsics

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]
            output_intrinsics[:, 0, 2] = w - output_intrinsics[:, 0, 2]
        else:
            output_images = images
            output_intrinsics = intrinsics
        return output_images, output_intrinsics


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, ch = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[:,0] *= x_scaling
        output_intrinsics[:,1] *= y_scaling

        scaled_images = [cv2.resize(im, (scaled_w, scaled_h)) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]

        output_intrinsics[:, 0, 2] -= offset_x
        output_intrinsics[:, 1, 2] -= offset_y

        return cropped_images, output_intrinsics
