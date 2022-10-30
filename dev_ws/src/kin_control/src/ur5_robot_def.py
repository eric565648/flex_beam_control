#!/usr/bin/env python3

from general_robotics_toolbox import * 
from general_robotics_toolbox import tesseract as rox_tesseract
from general_robotics_toolbox import robotraconteur as rr_rox

import numpy as np
import yaml, copy,time
import pickle

def Rx(theta):
    return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
def Ry(theta):
    return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
def Rz(theta):
    return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

def Rp2H(p,R):
    ###return homogeneous transformation matrix from R and p
    return np.vstack((np.hstack((R,np.array([p]).T)),np.array([0,0,0,1])))

class Transform_all(object):
    def __init__(self, p_all, R_all):
        self.R_all=np.array(R_all)
        self.p_all=np.array(p_all)

class robot_obj(object):
    ###robot object class
    def __init__(self,def_path,tool_file_path='',base_transformation_file='',d=0,acc_dict_path=''):
        #def_path: robot 			definition yaml file, name must include robot vendor
        #tool_file_path: 			tool transformation to robot flange csv file
        #base_transformation_file: 	base transformation to world frame csv file
        #d: 						tool z extension
        #acc_dict_path: 			accleration profile

        with open(def_path, 'r') as f:
            robot = rr_rox.load_robot_info_yaml_to_robot(f)

        if len(tool_file_path)>0:
            tool_H=np.loadtxt(tool_file_path,delimiter=',')
            robot.R_tool=tool_H[:3,:3]
            robot.p_tool=tool_H[:3,-1]+np.dot(tool_H[:3,:3],np.array([0,0,d]))		

        if len(base_transformation_file)>0:
            self.base_H=np.loadtxt(base_transformation_file,delimiter=',')
        else:
            self.base_H=np.eye(4)

        if len(robot.joint_names)>6:	#redundant kinematic chain
            tesseract_robot = rox_tesseract.TesseractRobot(robot, "robot", invkin_solver="KDL")
        elif 'UR' in def_path:			#UR
            tesseract_robot = rox_tesseract.TesseractRobot(robot, "robot", invkin_solver="URInvKin")
        else:							#sepherical joint
            tesseract_robot = rox_tesseract.TesseractRobot(robot, "robot", invkin_solver="OPWInvKin")

        self.tesseract_robot=tesseract_robot


        ###set attributes
        self.upper_limit=robot.joint_upper_limit 
        self.lower_limit=robot.joint_lower_limit 
        self.joint_vel_limit=robot.joint_vel_limit 

    def fwd(self,q_all,world=False):
        ###robot forworld kinematics
        #q_all:			robot joint angles or list of robot joint angles
        #world:			bool, if want to get coordinate in world frame or robot base frame
        if q_all.ndim==1:
            q=q_all
            pose_temp=self.tesseract_robot.fwdkin(q)	

            if world:
                pose_temp.p=self.base_H[:3,:3]@pose_temp.p+self.base_H[:3,-1]
                pose_temp.R=self.base_H[:3,:3]@pose_temp.R
            return pose_temp
        else:
            pose_p_all=[]
            pose_R_all=[]
            for q in q_all:
                pose_temp=self.tesseract_robot.fwdkin(q)	
                if world:
                    pose_temp.p=self.base_H[:3,:3]@pose_temp.p+self.base_H[:3,-1]
                    pose_temp.R=self.base_H[:3,:3]@pose_temp.R

                pose_p_all.append(pose_temp.p)
                pose_R_all.append(pose_temp.R)

            return Transform_all(pose_p_all,pose_R_all)
    
    def jacobian(self,q):
        return self.tesseract_robot.jacobian(q)

    def inv(self,p,R,last_joints=[]):
        if len(last_joints)==0:
            last_joints=np.zeros(len(self.joint_vel_limit))
        return self.tesseract_robot.invkin(Transform(R,p),last_joints)