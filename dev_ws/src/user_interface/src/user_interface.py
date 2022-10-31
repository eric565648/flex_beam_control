#!/usr/bin/env python3

from copy import deepcopy
import numpy as np
from numpy.random import uniform
from pandas import DataFrame
# import pygame as pg
import rospy
import rospkg
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Joy

#### BEAM Parameters
BODY_RADIUS=3
BODY_COLOR=[255,255,0]
BASE_RADIUS=4
BASE_COLOR=[0,0,255]
TIP_RADIUS=1
TIP_COLOR=[0,255,0]
BEAM_THICKNESS=1
BEAM_COLOR=[255,255,255]

#### TARGET Parameters
TARGET_RADIUS=9
TARGET_COLOR=[0,0,255]
TARGET_TOTAL=6
TARGET_RANGE_UP=[[418,95],[440,140]]
TARGET_RANGE_DOWN=[[462,416],[482,453]]

# scale
SCALE_IMG = 1.5

## Robot Range Parameters
TOOL_RANGE=[[117,300],[140,342]]

## Cnt Down
CNT_DOWN_T = 5
IN_TARGET_T= 5
PUT_TEXT_O=[0,50]
MSG_IMG_SIZE=60

class GameInterface(object):
    def __init__(self) -> None:
        super().__init__()

        # pg.init()
        # self.pg_clock = pg.time.Clock()
        # self.pg_screen = pg.display.set_mode((700,800))

        self.background = np.ones((600,575,3))*0
        self.dots = None
        self.joy = None

        ### only for test
        self.joy=Joy()
        self.joy.buttons=np.array([0,0,0,0,0,0])

        ##### variables for interface
        self.stage = 0
        self.joy_previous = None # record previous joy commands
        self.cnt_down_st = None
        self.target_cnt = 0
        self.target_origin = None
        self.target_o_updown=None
        self.trial_st = None
        self.trial_duration=[]

        ## subscriber
        self.dot_pub = rospy.Subscriber('/dot',Float32MultiArray,self.dot_cb,queue_size=1)
        self.sub_joy = rospy.Subscriber('/joy',Joy,self.joy_cb,queue_size=1)
        
        ## publisher
        self.target_pup = rospy.Publisher('/current_target',Float32MultiArray,queue_size=1)
        # self.vel_constrain_pub = rospy.Publisher()

        ## timer for pygame
        self.pg_timer = rospy.Timer(rospy.Duration(1./100),self.timer_cb)

    def dot_cb(self,msg):
        self.dots = np.array(msg.data)
        self.dots = np.reshape(self.dots,(int(len(self.dots)/2),2))

        # print(img)
        # self.pg_screen.blit(pg.surfarray.make_surface(img),(0,0))
        # pg.display.update()

    def joy_cb(self,msg):
        self.joy = msg
    
    def timer_cb(self,event):

        if self.dots is None:
            return
        if self.joy is None:
            return
        if self.joy_previous is None:
            self.joy_previous = deepcopy(self.joy)

        dots = deepcopy(self.dots)
        img = deepcopy(self.background)
        msg_img = deepcopy(img[:MSG_IMG_SIZE,:,:])
        joy_state = deepcopy(self.joy)
        joy_state_prev = deepcopy(self.joy_previous)

        ##### stages
        if self.stage == 0: ## wait stage
            ## activate stage 
            if joy_state_prev.buttons[0]==0 and joy_state.buttons[0]==1:
                self.stage=1
            msg_img = cv2.putText(msg_img,"Push Button 1 to Start",PUT_TEXT_O,\
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        elif self.stage == 1: ## start count down stage
            if self.cnt_down_st is None:
                self.cnt_down_st=rospy.get_time()
            # count down time
            now_t = rospy.get_time()
            cnt_down = CNT_DOWN_T-(now_t-self.cnt_down_st)
            # put on img
            msg_img = cv2.putText(msg_img,str(round(cnt_down,2)),PUT_TEXT_O,\
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            
            if cnt_down<= 0:
                self.cnt_down_st=None
                self.trial_duration=[]
                self.target_cnt=0
                self.stage = 2 ## after count down, do to start stage

        elif self.stage == 2: ## start stage
            if self.cnt_down_st is None:
                self.cnt_down_st=rospy.get_time()
            duration = rospy.get_time()-self.cnt_down_st

            ### draw target zone
            # img = cv2.rectangle(img,TARGET_RANGE_UP[0],TARGET_RANGE_UP[1],(60,0,60),-1)
            # img = cv2.rectangle(img,TARGET_RANGE_DOWN[0],TARGET_RANGE_DOWN[1],(60,0,60),-1)

            ### draw target
            if self.target_origin is None:
                self.target_origin=np.array([uniform(TARGET_RANGE_UP[0][0],TARGET_RANGE_UP[1][0]),\
                                uniform(TARGET_RANGE_UP[0][1],TARGET_RANGE_UP[1][1])]).astype(int)
                self.target_o_updown=0
                self.trial_st=rospy.get_time()
            img = cv2.circle(img,self.target_origin,TARGET_RADIUS,TARGET_COLOR,-1)
            target_msg=Float32MultiArray()
            target_msg.data=self.target_origin
            self.target_pup.publish(target_msg)

            tip_o = np.array(deepcopy(dots[-1]))
            ### if tip in target
            if np.linalg.norm(tip_o-self.target_origin)<=TARGET_RADIUS or\
                joy_state.buttons[1]==1:
                # put cnt down on img
                msg_img = cv2.putText(msg_img,str(round(duration,2)),PUT_TEXT_O,\
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
                
                ## if cnt down enough time
                if duration >= IN_TARGET_T:
                    # record trial time
                    self.trial_duration.append(rospy.get_time()-self.trial_st)
                    # add target count
                    self.target_cnt+=1
                    print('target updaow:',self.target_o_updown)
                    if self.target_cnt<TARGET_TOTAL:
                        if self.target_o_updown==0:
                            self.target_origin=np.array([uniform(TARGET_RANGE_DOWN[0][0],TARGET_RANGE_DOWN[1][0]),\
                                uniform(TARGET_RANGE_DOWN[0][1],TARGET_RANGE_DOWN[1][1])]).astype(int)
                            self.target_o_updown=1
                            self.trial_st=rospy.get_time()
                        elif self.target_o_updown==1:
                            self.target_origin=np.array([uniform(TARGET_RANGE_UP[0][0],TARGET_RANGE_UP[1][0]),\
                                uniform(TARGET_RANGE_UP[0][1],TARGET_RANGE_UP[1][1])]).astype(int)
                            self.target_o_updown=0
                            self.trial_st=rospy.get_time()
                        self.cnt_down_st=None
                    else: # finish all targets
                        print(self.trial_duration)
                        self.target_origin=None
                        self.target_o_updown=None
                        self.target_cnt=None
                        self.cnt_down_st=None
                        self.stage=3
            else:
                self.cnt_down_st=None
                msg_img = cv2.putText(msg_img,"Move Green Tip to Target,"+str(TARGET_TOTAL-self.target_cnt)+" to go",PUT_TEXT_O,\
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            
            
        elif self.stage == 3:
            ### save duration file
            df=DataFrame({'duration':self.trial_duration})
            rospack = rospkg.RosPack()
            df.to_csv(rospack.get_path('user_interface')+'/data/test_'+str(rospy.get_rostime().nsecs)+'.csv')
            print("Total Duration:")
            print(self.trial_duration)
            self.stage=0

        ##### draw tool range
        # img = cv2.rectangle(img,TOOL_RANGE[0],TOOL_RANGE[1],(0,60,0),-1)

        ##### draw beam and dots
        for dot_i in range(len(dots)):
            dot=dots[dot_i].astype(int)
            if dot_i==0:
                # print("0:",dot)
                img = cv2.line(img,dot,dots[dot_i+1].astype(int),BEAM_COLOR,BEAM_THICKNESS)
                img = cv2.circle(img,dot,BASE_RADIUS,BASE_COLOR,-1)
            elif dot_i == len(self.dots)-1:
                # print("4:",dot)
                img = cv2.circle(img,dot,TIP_RADIUS,TIP_COLOR,-1)
            else:
                img = cv2.line(img,dot,dots[dot_i+1].astype(int),BEAM_COLOR,BEAM_THICKNESS)
                img = cv2.circle(img,dot,BODY_RADIUS,BODY_COLOR,-1)

        ### concate msg img
        img = np.vstack((img,msg_img))

        ## resize img
        width = int(img.shape[1] * SCALE_IMG)
        height = int(img.shape[0] * SCALE_IMG)
        img = cv2.resize(img,(width, height),interpolation=cv2.INTER_AREA)
        
        cv2.imshow("process img",img)
        cv2.waitKey(1)

        ##### record previous joy
        self.joy_previous = deepcopy(joy_state)

if __name__=='__main__':

    rospy.init_node('game_interface')
    gi = GameInterface()
    rospy.spin()
