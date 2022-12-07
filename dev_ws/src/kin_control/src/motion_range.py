#!/usr/bin/env python3

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from pandas import Interval
import rospy
import rospkg
import numpy as np

from sensor_msgs.msg import JointState

### robotics
from general_robotics_toolbox import *
from ur5_robot_def import *

res=1
x_range=np.arange(-70,70,res)
y_range=np.arange(-70,70,res)
# x_range=np.arange(150,1200,res)
# y_range=np.arange(-300,750,res)
motion_plot=np.zeros((len(y_range),len(x_range)))
pos=None

fig, ax = plt.subplots()

## robot range
### define robot
rospack =rospkg.RosPack()
robot=robot_obj(rospack.get_path('kin_control')+'/config/UR5e_robot_default_config.yml')
COMPENSET=[2,1,0,3,4,5]
y_max=-9999999
y_min=9999999
z_max=-9999999
z_min=9999999

### robot range
# x: 455, 888  (range: 433)
# y: -346, 871 (rabge: 1217, half: 262.5)

### falcon range
# x: -55 59 (range 114)
# y: -56 56 (range 112)

def animate(i):

    ax.set_xticks=np.arange(0,70,10)
    ax.set_yticks=np.arange(0,70,10)
    ax.clear()
    ax.matshow(motion_plot)
    # print(pos)
    # print("===============")

def joint_cb(msg):

    global pos,y_min,y_max,z_min,z_max

    ## falcon plot
    x_index = np.argmax(x_range>msg.position[0])-1
    y_index = np.argmax(y_range>msg.position[1])
    global pos
    pos=np.round(msg.position,1)
    if pos[0]<y_min:
        y_min=pos[0]
    if pos[0]>y_max:
        y_max=pos[0]
    if pos[1]<z_min:
        z_min=pos[1]
    if pos[1]>z_max:
        z_max=pos[1]
    # print(y_min,y_max)
    # print(z_min,z_max)
    # print(msg.position)
    motion_plot[-y_index,x_index]=1
    
    ## robot plot
    # robot_q=np.array(msg.position)
    # robot_q=robot_q[COMPENSET]
    # robot_p=robot.fwd(robot_q).p*1000
    # pos=np.round(robot_p)
    # x_index = np.argmax(x_range>-pos[1])-1
    # y_index = np.argmax(y_range>pos[2])
    # # print(pos)
    # if pos[1]<y_min:
    #     y_min=pos[1]
    # if pos[1]>y_max:
    #     y_max=pos[1]
    # if pos[2]<z_min:
    #     z_min=pos[2]
    # if pos[2]>z_max:
    #     z_max=pos[2]
    # print(y_min,y_max)
    # print(z_min,z_max)
    # print(pos)
    # motion_plot[-y_index,x_index]=1
    

if __name__=='__main__':

    rospy.init_node("Motion_Range")

    sub_joint = rospy.Subscriber('falcon_cart_state',JointState,joint_cb,queue_size=1)
    # sub_joint = rospy.Subscriber('joint_states',JointState,joint_cb,queue_size=1)
    
    ani=animation.FuncAnimation(fig,animate,interval=10)
    plt.show()

    rospy.spin()


