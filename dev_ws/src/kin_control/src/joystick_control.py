#!/usr/bin/env python3

from matplotlib.pyplot import axis
import rospy
import rospkg
import numpy as np
from sensor_msgs.msg import Joy,JointState
from std_msgs.msg import Float64MultiArray

### robotics
from general_robotics_toolbox import *
from ur5_robot_def import *

MAX_OMG=0.5
MAX_VEL=0.05
DEAD_ZONE=0.01

class Joy2Vel(object):
    
    def __init__(self) -> None:

        ### define robot
        self.rospack =rospkg.RosPack()
        self.robot=robot_obj(self.rospack.get_path('kin_control')+'/config/UR5e_robot_default_config.yml')
        self.robot_q = None
        
        self.joint_vel_pub = rospy.Publisher('/joint_group_vel_controller/command',
        Float64MultiArray,queue_size=1)  
        self.vel_pub = rospy.Publisher('vel_cmd',Float64MultiArray,queue_size=1)    

        self.omega_th0 = 0
        self.vel_x = 0
        self.vel_y = 0
        self.sub_joy = rospy.Subscriber('/joy',Joy,self.joy_cb,queue_size=1)
        self.sub_states = rospy.Subscriber('/joint_states',JointState,self.joint_state_cb,
                            queue_size=1)

    def joy_cb(self,msg):

        self.omega_th0 = msg.axes[3]*MAX_OMG if np.fabs(msg.axes[3])>=DEAD_ZONE else 0
        # self.vel_x = msg.axes[-2]*-1*MAX_VEL if np.fabs(msg.axes[-2])>=DEAD_ZONE else 0
        # self.vel_y = msg.axes[-1]*MAX_VEL if np.fabs(msg.axes[-1])>=DEAD_ZONE else 0
        self.vel_x = msg.axes[0]*-1*MAX_VEL if np.fabs(msg.axes[0])>=DEAD_ZONE else 0
        self.vel_y = msg.axes[1]*MAX_VEL if np.fabs(msg.axes[1])>=DEAD_ZONE else 0

    def joint_state_cb(self,msg):

        self.robot_q=np.array(msg.position)

        robot_J = self.robot.jacobian(self.robot_q)
        # print(robot_J)
        vx=self.vel_x
        vy=self.vel_y
        joint_vel = np.matmul(np.linalg.pinv(robot_J),[0,0,0,0,vx,vy])
        joint_vel[5] += self.omega_th0*MAX_OMG
        # print(vx,vy)

        vel_msg = Float64MultiArray()
        vel_msg.data=joint_vel
        # print(joint_vel)
        self.joint_vel_pub.publish(vel_msg)
        v_msg=Float64MultiArray()
        v_msg.data=[vx,vy,self.omega_th0*MAX_OMG]
        self.vel_pub.publish(v_msg)

if __name__=='__main__':
    
    rospy.init_node('joy_to_vel')
    j2v=Joy2Vel()
    rospy.spin()