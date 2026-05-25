from pydobot.dobot import MODE_PTP
import pydobot
import time

# Robot motion utility functions
def move_to_home(device):
    print("Homing the robot...")
    device.home()  
    time.sleep(2)
    (pose, joint) = device.get_pose()  
    print(f"pose: {pose}, j: {joint}")
  
def move_to_specific_position(device,x,y,z,r = 0.0):
    device.speed(100, 100)
    device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x = x, y = y , z = z , r= r)  
    time.sleep(0.7)

def get_current_pose(device):
    time.sleep(1)  
    print("current pose")
    (pose, joint) = device.get_pose()  # Get the current position and joint angles
    return pose, joint