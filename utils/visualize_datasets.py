from imageio import imread 
import numpy as np 
import matplotlib.pyplot as plt 

path = "/home/vk/03/ThermalSfMLearner/collect some datasets/01.png"
# path = "/home/vk/03/ThermalSfMLearner/ProcessedData/indoor_aggresive_dark/Thermal/000007.png" 

image = imread(path) 

print(image) 

print("\033[92m[INFO]\033[00m Image Shape: ", image.shape)   

plt.imshow(image, cmap='gray') 
plt.show()


print("Image Shape: ", image.shape)  



