import cv2 
import numpy as np 
from flirpy.camera.lepton import Lepton

def display_live_feed():
    camera = Lepton() 

    while True:

        thermal_image = camera.grab().astype(np.float32)
        
        # print("\033[92m [INFO] \033[0m Thermal camera live feed started") 
        # print("\033[92m [INFO] \033[0m image:  ", thermal_image)  
        # print("\033[92m [INFO] \033[0m Shape : ", thermal_image.shape)  
        
        print("\033[92m [INFO] \033[0m Press 'ESC' to exit") 
        
        thermal_image = cv2.resize(thermal_image, (1080,720), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('thermal_image',thermal_image) 
       
       
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        
    camera.close() 
    

display_live_feed() 


