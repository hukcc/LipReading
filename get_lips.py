#coding: utf-8
import sys
import dlib
import cv2
import numpy as np
import os

def file_name(file_dir):   
        L=[]   
        for root, dirs, files in os.walk(file_dir):  
            for file in files:  
                if os.path.splitext(file)[1] == '.mpg':  
                    L.append(os.path.join(root, file))  
        return L  

detector = dlib.get_frontal_face_detector() #获取人脸分类器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
cnt = 0 #count videos
while(1):
    videos_path = file_name("s1")
    if cnt >= len(videos_path) :
        print ("no more videos!")
        break
    cap = cv2.VideoCapture(videos_path[cnt])
    cnt = cnt + 1
    pics_num = 0    #count frames
########################################################To write lip images as video#######################################
    
    # fps = cap.get(cv2.CAP_PROP_FPS)  
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # video_saver = cv2.VideoWriter("test",cv2.cv2.CAP_PROP_FOURCC('M', 'J', 'P', 'G'),fps,size)
    #still have some problem on this
############################################################################################################################
    while(1):
        succ,img = cap.read()
        if img is None:
            break
        rects = detector(img, 0)
        #pics_num = 0
        for i in range(len(rects)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
            top = 100000 
            bottle = 0 
            left = 10000  
            right = 0
            for idx, point in enumerate(landmarks):
                if idx > 48 :
                    pos = (point[0,0] , point[0,1])
                    if point[0,0] < left :
                        left   = point[0,0]
                    if point[0,0] > right :
                        right  = point[0,0] 
                    if point[0,1] < top :
                        top    = point[0,1]
                    if point[0,1] > bottle :
                        bottle = point[0,1]
                    cv2.circle(img, pos, 1, color=(0, 255, 0))
            w_offset = (right-left)//4
            h_offset = (bottle-top)//2
            cv2.rectangle(img,(left-w_offset,top-h_offset),(right+w_offset,bottle+h_offset),(0,0,255),3)
            liproi = img[top-h_offset:bottle+h_offset,left-w_offset:right+w_offset]
            #print ((left-w_offset,top-h_offset),(right+w_offset,bottle+h_offset))
        
        #video_saver.write(img)           #write video
        cv2.imwrite("pics/"+str(cnt)+"/"+str(pics_num)+".jpg",liproi) 
        print (cnt , pics_num)
        pics_num = pics_num + 1
        cv2.imshow("", img)
        cv2.imshow("lip",liproi)
        cv2.waitKey(1)
