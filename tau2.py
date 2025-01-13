from flirpy.camera.lepton import Lepton
import numpy as np 

camera = Lepton()
image = camera.grab().astype(np.float32)   


print(image.shape) 
# image.save("./images/thermal_image.png")   

print(image)  

camera.close()  


