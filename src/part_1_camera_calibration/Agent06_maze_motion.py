import numpy as np
import cv2  
from pydobot.dobot import MODE_PTP
import time
import pydobot
import json
from camera_utilities import apply_affine, fit_affine, apply_homography, fit_homography
from robot_utilities import move_to_home, move_to_specific_position, get_current_pose

PORT = "COM6"
Z = -30

M = np.array([
    [-1.13736975e-02, -4.68272261e-01,  4.02584775e+02],
    [-4.77967150e-01,  5.47222325e-03,  1.38871876e+02]
], dtype=np.float64)

H = np.array([
    [-2.44594058e-02, -4.75669460e-01,  3.67247188e+02],
    [-4.34041615e-01,  5.08065338e-03,  1.20901686e+02],
    [-5.98330506e-05 ,-7.62411614e-05 , 1.00000000e+00]]
    , dtype=np.float64)

def move_robot_point(device,M,u,v):
    Xa, Ya = apply_affine(M, u, v) # Using Affine
    # Xa, Ya = apply_homography(H, u, v) # Using Homography
    print(f"Affine:  pixel({u:.3f}, {v:.3f}) -> robot({Xa:.6f}, {Ya:.6f})")
    move_to_specific_position(device, x=Xa, y=Ya, z=Z) # z = -45
    time.sleep(0.2)

def Agent06_RobotMove():
    device = pydobot.Dobot(port=PORT)
    device.speed(100, 100)
    move_to_home(device)
    time.sleep(2)

    data_path = "solution_path_points_unwarped.json"
    with open(data_path, 'r') as file:
        data = json.load(file)
    file.close()
    
    pix_coords = data['unwarped_path_pixels']
    pixel_coords = list(map(lambda x:tuple(x), pix_coords))

    for (u, v) in pixel_coords[:-2]:
        move_robot_point(device, M, u, v) 

    device.close() 
    
# if __name__ == "__main__":
#     main()
