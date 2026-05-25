from Agent01_capture_image import *
from Agent02_maze_warp_from_json import *
from Agent03_maze_circles_and_grid_better import *
# from solve import *
from Agent04_solve_maze_agentic import *
from Agent05_unwrap_and_overlay_path import *
from Agent06_maze_motion import *
import time

def main():
    # start_color = input("What is the start color? (red/green): ").lower().strip()
    # if start_color not in ["red", "green"]:
    #     raise ValueError("Color given is invalid")
    # if start_color=="red":
    #     end_color = "green"
    # else:
    #     end_color = "red"

    Agent01_capture_image()
    time.sleep(1.2)
    Agent02_maze_warp_from_json()
    time.sleep(1.2)
    Agent03_maze_circles_and_grid_better()
    time.sleep(1.2)
    Agent04_maze_solver()
    time.sleep(1.2)
    Agent05_unwrap_and_path_overlay()
    time.sleep(1.2)
    Agent06_RobotMove()
    print("DONE AND DUSTED")

if __name__ == "__main__":
    main()