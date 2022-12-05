#!/usr/bin/env python3

import numpy as np
from copy import deepcopy
import rospy
import message_filters
from sensor_msgs.msg import Image,CameraInfo
from std_msgs.msg import Float32MultiArray
import cv2
from cv_bridge import CvBridge,CvBridgeError
import time

TOTAL_BLOB=5
ANC_ORIGIN=np.array([0,571])
COLOR_CODE=[[0,0,255],[255,0,0],[255,0,0],[255,0,0],[0,255,0]]

class KalmanFilter(object):
    def __init__(self) -> None:
        super().__init__()

        ## state
        ## px,py,vx,vy
        self.state=None
        self.trans_mat=None
        self.H_mat = np.array([[1,0,0,0],
                            [0,1,0,0]])

        ## variance of measurement
        ## unit: pixel, pixel/sec
        self.pos_var = 1
        self.vel_var = 5
        self.cov_mat = np.identity(4)
        self.cov_mat[0,0]*=self.pos_var
        self.cov_mat[1,1]*=self.pos_var
        self.cov_mat[2,2]*=self.vel_var
        self.cov_mat[3,3]*=self.vel_var

        ## noise
        self.Q_mat = np.identity(4)*100
        self.R_mat = deepcopy(self.cov_mat[:2,:2])
    
    def init_measurement(self,pos,vel,dt):
        self.state = np.array([pos[0],pos[1],vel[0],vel[1]])
        self.trans_mat = np.identity(4)
        self.trans_mat[0,2] = dt
        self.trans_mat[1,3] = dt
        self.Q_mat = self.Q_mat*dt
    
    def predict_update(self,pose_obs):
        
        ## predict
        mu_pred = np.matmul(self.trans_mat,self.state)
        cov_pred = np.matmul(self.trans_mat,\
                np.matmul(self.cov_mat.T,self.trans_mat.T))+self.Q_mat
        
        ## update
        if pose_obs is None:
            self.state=np.squeeze(mu_pred)
            self.cov_mat=np.squeeze(cov_pred)
        else:
            pose_obs=np.squeeze(np.array(pose_obs))
            HCH_R=np.matmul(self.H_mat,np.matmul(cov_pred,self.H_mat.T))+self.R_mat
            kalman_gain = np.matmul(cov_pred,np.matmul(self.H_mat.T,np.linalg.pinv(HCH_R)))
            # kalman_gain = np.matmul(np.linalg.pinv(HCH_R),np.matmul(self.H_mat,cov_pred))
            innovation = pose_obs-np.matmul(self.H_mat,mu_pred)
            mu_update = mu_pred+np.matmul(kalman_gain,innovation)
            # cov_update = cov_pred-np.matmul(kalman_gain,
            #                     np.matmul(HCH_R,kalman_gain.T))
            cov_update = np.matmul(np.identity(kalman_gain.shape[0])-np.matmul(kalman_gain,self.H_mat),\
                                cov_pred)
            
            self.state=np.squeeze(mu_update)
            self.cov_mat=np.squeeze(cov_update)

        return self.state[:2]

class IRDetection(object):
    def __init__(self) -> None:
        super().__init__()

        self.bridge = CvBridge()

        ## kalman filter
        self.DOT_NUM=5
        self.kalman_filters=[KalmanFilter() for i in range(self.DOT_NUM)]

        ## publisher
        self.dot_pub = rospy.Publisher('/dot',Float32MultiArray,queue_size=1)
        # self.ir_enhance_pub = rospy.Publisher('/ir/enhance')

        ## subscriber
        # self.ir_sub=rospy.Subscriber('/ir/image_raw',Image,self.img_cb,queue_size=1)
        self.ir_sub=message_filters.Subscriber('/ir/image_raw',Image)
        self.depth_sub=message_filters.Subscriber('/depth/image_raw',Image)
        self.depth_info_sub=message_filters.Subscriber('/depth/camera_info',CameraInfo)
        self.ts_img = message_filters.TimeSynchronizer([self.ir_sub,self.depth_sub,self.depth_info_sub],1)
        self.ts_img.registerCallback(self.img_cb)
    
    def img_cb(self,ir_img,depth_img,depth_info):

        try:
            self.img = self.bridge.imgmsg_to_cv2(ir_img)
            self.depth_img=self.bridge.imgmsg_to_cv2(depth_img)
            self.depth_info=depth_info
        except CvBridgeError as e:
            print(e)

        self.detect_obj()
    
    def detect_obj(self):

        st=time.time()

        ## normalize img
        img = deepcopy(self.img.astype(float))
        img = img/np.max(img)

        ## thresholding to get tapes
        THRES=0.35
        ret,img = cv2.threshold(img,THRES,1,cv2.THRESH_BINARY)
        img = img.astype(np.uint8)*255

        ## find contour
        contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        ## draw contour rect
        boxes = []
        origins = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            # print(rect)
            origins.append(rect[0])
            box = cv2.boxPoints(rect)
            boxes.append(np.int0(box))
            # print('----------')
        # print("==================")
        origins=np.array(origins)
        boxes=np.array(boxes)
        # print(origins)
        # return

        ## find corresponding blobs
        min_blob_id = np.argmin(np.linalg.norm(origins-ANC_ORIGIN,2,axis=-1))
        sort_id = np.argsort(np.linalg.norm(origins-origins[min_blob_id],2,axis=1))
        origins=origins[sort_id]
        boxes=boxes[sort_id]

        # for box_i in range(len(boxes)):
        #     try:
        #         if box_i < len(boxes):
        #             c = COLOR_CODE[box_i]
        #         else:
        #             c=COLOR_CODE[-1]
        #     except IndexError:
        #         c=COLOR_CODE[1]
        #     img=cv2.drawContours(img,[boxes[box_i]],0,c,2)
        # cv2.imshow("process img",img)
        # cv2.waitKey(1)

        ### add kalman filer
        # 0.initialization
        if self.kalman_filters[0].state is None:
            for i in range(self.DOT_NUM):
                self.kalman_filters[i].init_measurement(origins[i],[0,0],1/30.)
            return
        else:
            # 1. data association based on distance
            origin_assocated={}
            for i in range(self.DOT_NUM):
                origin_assocated[i]=[]
            for origin in origins:
                all_dist=[]
                for i in range(self.DOT_NUM):
                    all_dist.append(np.linalg.norm(origin-self.kalman_filters[i].state[:2]))
                origin_assocated[np.argmin(all_dist)].append(origin)
            # 2. Do Kalman filter
            origins_filtered=[]
            for i in range(self.DOT_NUM):
                if len(origin_assocated[i])<1:
                    self.kalman_filters[i].predict_update(None)
                elif len(origin_assocated[i])>1:
                    origin_assocated_i_ave=np.mean(np.array(origin_assocated[i]),axis=0)
                    self.kalman_filters[i].predict_update(origin_assocated_i_ave)
                else:
                    self.kalman_filters[i].predict_update(origin_assocated[i])
                origins_filtered.append(self.kalman_filters[i].state[:2])
                
        
        ## draw contour
        img = np.stack((img,)*3,axis=-1)
        # img=cv2.drawContours(img,contours,-1,(0,0,255),1)
        # for box_i in range(len(boxes)):
        #     try:
        #         if box_i < len(boxes):
        #             c = COLOR_CODE[box_i]
        #         else:
        #             c=COLOR_CODE[-1]
        #     except IndexError:
        #         print(len(boxes))   
        #         print(box_i)
        #         c=COLOR_CODE[1]
        #     img=cv2.drawContours(img,[boxes[box_i]],0,c,2)
        # for origin in origins:
        #     img = cv2.circle(img,np.array(origin).astype(int),3,[0,0,255],-1)
        # for origin in origins_filtered:
        #     img = cv2.circle(img,np.array(origin).astype(int),3,[0,255,0],-1)
        
         
        ### no kalman filters
        # dot_msg = Float32MultiArray()
        # dot_msg.data=np.reshape(origins,(-1,))
        # self.dot_pub.publish(dot_msg)

        ## with kalman filters
        dot_msg = Float32MultiArray()
        dot_msg.data=np.reshape(origins_filtered,(-1,))
        self.dot_pub.publish(dot_msg)

        ## for debug
        # cv2.imshow("origin img",self.img)
        # cv2.imshow("process img",img)
        # cv2.waitKey(1)

        # print(time.time()-st)

if __name__=='__main__':
    
    rospy.init_node('ir_detection')
    ird = IRDetection()
    rospy.spin()