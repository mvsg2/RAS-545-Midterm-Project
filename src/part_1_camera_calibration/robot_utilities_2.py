import pydobot
import time

# Robot motion utility functions

def move_to_home(device):
    #_______________________________________________________
    # Moving the robot to a home and get home position
    # Code ##################################################
    print("Homing the robot...")
    device.home()  # Home the robot to the origin position
    device.move_to(x = 240, y = 0 , z = 150 , r= 45)  # Move to position (x=250, y=0, z=50) with r=0
    # Code End ##############################################
    # ______________________________________________________

def move_to_specific_position(device,x,y,z,r = 0.0):
    #_______________________________________________________
    # Moving the robot to a specific position
    # Code ##################################################
    device.speed(50, 50)
    device.move_to(x = x, y = y , z = z , r= 45)  # Move to position (x=250, y=0, z=50) with r=0
    time.sleep(2)
    # Code End ##############################################
    # ______________________________________________________

def get_current_pose(device):
    #_______________________________________________________
    # Getting the current pose
    # Code ##################################################
    time.sleep(1)  
    print("current pose")
    x,y,z,r,j1,j2,j3,j4 = device.pose()  # Get the current position and joint angles
    print(f"pose: {x,y,z}")
    return (x,y,z)
    # Code End ##############################################
    # ______________________________________________________
# ________________________________________
