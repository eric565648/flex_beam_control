#!/usr/bin/env python3

from tkinter import X
import rospy
import rospkg
import numpy as np
from sensor_msgs.msg import Joy,JointState
from geometry_msgs.msg import Wrench
from std_msgs.msg import Float64MultiArray,Int32

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

### space mapping
falcon_x=[-40,40]
robot_x=[450,800]
falcon_y=[-40,40]
robot_y=[0,350]
potential_range=420
theta_compenset=1
robot_theta_upper=np.radians(50)
robot_theta_lower=np.radians(-20)
robot_theta_range=robot_theta_upper-robot_theta_lower

class Joy2Space(object):
    
    def __init__(self) -> None:

        ### define robot
        self.rospack =rospkg.RosPack()
        self.robot=robot_obj(self.rospack.get_path('kin_control')+'/config/UR5e_robot_default_config.yml')
        self.robot_q = None
        self.HOME_POS_CART=self.robot.fwd(HOME_POS).p*1000

        self.use_falcon=True
        
        self.joint_vel_pub = rospy.Publisher('/joint_group_vel_controller/command',
        Float64MultiArray,queue_size=1)
        self.joy_feedback_pub = rospy.Publisher('falcon_joy_feedback',Wrench,queue_size=1)  
        self.vel_pub = rospy.Publisher('vel_cmd',Float64MultiArray,queue_size=1)    

        self.target = np.array([0,0])
        if not self.use_falcon:
            self.joy_topic_name='/joy'
        else:
            self.joy_topic_name='/falcon_joy'
        
        self.sub_joy = rospy.Subscriber(self.joy_topic_name,Joy,self.joy_cb,queue_size=1)
        self.sub_knob= rospy.Subscriber('joy_knob',Int32,self.knob_cb,queue_size=1)
        self.sub_states = rospy.Subscriber('/joint_states',JointState,self.joint_state_cb,
                            queue_size=1)

        ### go home button
        self.go_home_flag=False
        self.first_set_home=False
        self.falcon_catch_robot=False

        ### anchoring angle position
        self.anchor_angle=True
        self.theta_scale=robot_theta_range/potential_range
        self.knob_theta_start=None
        self.robot_theta_start=None
        self.target_theta=None
        self.b3_release=True

        self.theta_buffer=[]
        self.theta_ave_window=3

        ## shutdown
        self.shuttingdown=False
    
    def knob_cb(self,msg):

        self.theta_buffer.append(msg.data)
        if len(self.theta_buffer)>self.theta_ave_window:
            self.theta_buffer.pop(0)

        if self.anchor_angle:
            self.knob_theta_start=np.mean(self.theta_buffer)
            self.robot_theta_start=self.robot_q[-1]
            print("Reset Knob Theta")
            self.anchor_angle=False

        if self.knob_theta_start is not None:
            self.target_theta=theta_compenset*(msg.data-self.knob_theta_start)*self.theta_scale+self.robot_theta_start

    def joy_cb(self,msg):
        # self.go_home_flag=True

        if not self.use_falcon:
            return
        else:
            self.target_x=((msg.axes[0]-falcon_x[0])/(falcon_x[0]-falcon_x[1]))*(robot_x[0]-robot_x[1])+robot_x[0]
            self.target_y=((msg.axes[1]-falcon_y[0])/(falcon_y[0]-falcon_y[1]))*(robot_y[0]-robot_y[1])+robot_y[0]

        if msg.buttons[2] == 1:
            self.go_home_flag=True
        else:
            self.go_home_flag=False
        
        if msg.buttons[3] == 0:
            self.b3_release=True
        if msg.buttons[3] == 1 and self.b3_release:
            self.anchor_angle=True
            self.b3_release=False

    def joint_state_cb(self,msg):

        if not self.use_falcon:
            return

        self.robot_q=np.array(msg.position)
        self.robot_q=self.robot_q[COMPENSET]
        # print(np.degrees(self.robot_q))
        # q_dum=np.degrees([self.robot_q])
        # print(self.robot.fwd(self.robot_q).p*1000)

        if not self.first_set_home:
            self.go_home()
            msg_wrench=Wrench()
            self.joy_feedback_pub.publish(msg_wrench)
            if np.linalg.norm(HOME_POS-self.robot_q)<0.02:
                print("Set home done")
                vel_msg = Float64MultiArray()
                vel_msg.data=np.zeros(len(self.robot_q))
                self.joint_vel_pub.publish(vel_msg)
                self.first_set_home=True
            return

        if self.go_home_flag:
            self.go_home()
            self.falcon_catch_robot=False
            return

        robot_J = self.robot.jacobian(self.robot_q)
        
        p_d = np.array([0,0,0,self.HOME_POS_CART[0],X_COMPENSET*self.target_x,Y_COMPENSET*self.target_y])
        # print(p_d)
        p_t = np.append([0,0,0],self.robot.fwd(self.robot_q).p*1000)
        # p_t[3]=0
        # print(p_t)

        if not self.falcon_catch_robot:
            left=False
            down=False
            error_init=p_d-p_t
            if error_init[4]*X_COMPENSET>0:
                left=True
            if error_init[5]*Y_COMPENSET>0:
                down=True
            left_right_str='Left' if left else 'Right'
            up_down_str='down' if down else 'up'
            print("Move",left_right_str,up_down_str)
            if np.linalg.norm((p_d-p_t))>10:
                return
            else:
                self.falcon_catch_robot=True
                print("Falcon catch robot")
                pass
        Kp=5
        if rospy.has_param('Kp_space'):
            Kp=rospy.get_param('Kp_space')
        vd=Kp*(p_d-p_t)
        if p_t[4]>=INNER_CONS:
            if vd[4]>0:
                vd[4]=0
        vd_constrain=2000
        vd_stop=vd_constrain
        if np.linalg.norm(vd)>vd_constrain:
            vd=vd/np.linalg.norm(vd)*vd_constrain
        if np.linalg.norm(vd)>vd_stop:
            vd=np.zeros(len(vd))
        vd*=1e-3
            
        ### outer hard constraints
        joint_vel = np.matmul(np.linalg.pinv(robot_J),vd)
        # print(self.robot_q[2])
        if self.robot_q[2]>OUTER_CONS:
            if joint_vel[2]>0:
                joint_vel=np.zeros(joint_vel.shape)
        # print(joint_vel)
        
        if self.target_theta is not None:
            
            Kp_theta=5
            if rospy.has_param('Kp_theta'):
                Kp_theta=rospy.get_param('Kp_theta')
            constraint_factor=0.1
            joint_vel[5] =Kp_theta*(self.target_theta-self.robot_q[-1])
            joint_vel[5]=max(-self.robot.joint_vel_limit[5]*constraint_factor,joint_vel[5])
            joint_vel[5]=min(self.robot.joint_vel_limit[5]*constraint_factor,joint_vel[5])
            if self.robot_q[-1]<robot_theta_lower:
                joint_vel[5]=max([0,joint_vel[5]])
            if self.robot_q[-1]>robot_theta_upper:
                joint_vel[5]=min([0,joint_vel[5]])

        ## joystickfeedback pub
        error_p=p_t-p_d
        dx=(error_p[4]*X_COMPENSET/(robot_x[0]-robot_x[1]))*(falcon_x[0]-falcon_x[1])
        dy=(error_p[5]*Y_COMPENSET/(robot_y[0]-robot_y[1]))*(falcon_y[0]-falcon_y[1])
        K_stiff=0.05
        if rospy.has_param('K_stiff'):
            K_stiff=rospy.get_param('K_stiff')
        msg_wrench=Wrench()
        msg_wrench.force.x=K_stiff*dx
        msg_wrench.force.y=K_stiff*dy
        

        if not self.shuttingdown:
            ## velocity pub
            vel_msg = Float64MultiArray()
            vel_msg.data=joint_vel
            # print(joint_vel)
            self.joint_vel_pub.publish(vel_msg)
            
            ## wrench pub
            self.joy_feedback_pub.publish(msg_wrench)
            
            v_msg=Float64MultiArray()
            v_msg.data=[X_COMPENSET*vd[4],Y_COMPENSET*vd[5],joint_vel[5]]
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
    
    def shutdown(self):
        
        print("Kinematic Control Shutdown")
        self.shuttingdown=True
        vel_msg = Float64MultiArray()
        self.joint_vel_pub.publish(vel_msg)
        msg_wrench=Wrench()
        self.joy_feedback_pub.publish(msg_wrench)

if __name__=='__main__':
    
    rospy.init_node('joy_to_space')
    j2s=Joy2Space()
    rospy.on_shutdown(j2s.shutdown)
    rospy.spin()