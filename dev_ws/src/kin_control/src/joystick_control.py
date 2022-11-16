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
COMPENSET=[2,1,0,3,4,5]
X_COMPENSET=-1
Y_COMPENSET=1

INNER_CONS=-459 # in x-direction
OUTER_CONS=np.radians(-31) # joint 3 can not smaller than

HOME_POS=np.radians([-269.944,-144.784,-101.074,-24.426,-1.873,3.642])
HOME_P=1
HOME_I=0.1
HOME_D=0.01

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

        ### go home button
        self.go_home_flag=False

    def joy_cb(self,msg):

        self.omega_th0 = msg.axes[3]*MAX_OMG if np.fabs(msg.axes[3])>=DEAD_ZONE else 0
        # self.vel_x = msg.axes[-2]*-1*MAX_VEL if np.fabs(msg.axes[-2])>=DEAD_ZONE else 0
        # self.vel_y = msg.axes[-1]*MAX_VEL if np.fabs(msg.axes[-1])>=DEAD_ZONE else 0
        self.vel_x = msg.axes[0]*-1*MAX_VEL if np.fabs(msg.axes[0])>=DEAD_ZONE else 0
        self.vel_y = msg.axes[1]*MAX_VEL if np.fabs(msg.axes[1])>=DEAD_ZONE else 0

        if msg.buttons[2] == 1:
            self.go_home_flag=True
        else:
            self.go_home_flag=False

    def joint_state_cb(self,msg):

        self.robot_q=np.array(msg.position)
        self.robot_q=self.robot_q[COMPENSET]
        # print(np.degrees(self.robot_q))
        # q_dum=np.degrees([self.robot_q])
        # print(self.robot.fwd(self.robot_q).p*1000)

        if self.go_home_flag:
            self.go_home()
            return

        robot_J = self.robot.jacobian(self.robot_q)
        # print(robot_J)
        vx=self.vel_x
        vy=self.vel_y
        ### inner hard constraints
        robot_p=self.robot.fwd(self.robot_q).p*1000
        if robot_p[1]>=INNER_CONS:
            if vx<0:
                vx=0
        ### outer hard constraints
        joint_vel = np.matmul(np.linalg.pinv(robot_J),[0,0,0,0,X_COMPENSET*vx,Y_COMPENSET*vy])
        # print(self.robot_q[2])
        if self.robot_q[2]>OUTER_CONS:
            if joint_vel[2]>0:
                joint_vel=np.zeros(joint_vel.shape)
        # print(joint_vel)
        
        joint_vel[5] += self.omega_th0*MAX_OMG
        # print(vx,vy)
        # print(joint_vel)

        vel_msg = Float64MultiArray()
        vel_msg.data=joint_vel
        # print(joint_vel)
        self.joint_vel_pub.publish(vel_msg)
        v_msg=Float64MultiArray()
        v_msg.data=[vx,vy,self.omega_th0*MAX_OMG]
        self.vel_pub.publish(v_msg)
    
    def go_home(self):

        pos_error = HOME_POS-self.robot_q
        joint_vel=pos_error*HOME_P
        joint_vel=np.clip(joint_vel,-0.3,0.3)
        # print(joint_vel)
        # print(np.degrees(self.robot_q))
        vel_msg = Float64MultiArray()
        vel_msg.data=joint_vel
        self.joint_vel_pub.publish(vel_msg)
        return

if __name__=='__main__':
    
    rospy.init_node('joy_to_vel')
    j2v=Joy2Vel()
    rospy.spin()