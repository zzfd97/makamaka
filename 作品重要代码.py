#头文件

from pynq import Overlay 
from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *
#引入bit设置像素分辨率
base = BaseOverlay("base.bit")
Mode = VideoMode(640,480,24)
#lib = MicroblazeLibrary(base.ARDUINO, ['uart'])
#device = lib.uart_open(0,1)
#hdmi_in = base.video.hdmi_in
hdmi_out = base.video.hdmi_out
#hdmi配置
hdmi_out.configure(Mode,PIXEL_BGR)

hdmi_out.start()
#导入库
import cv2
import imutils
import numpy as np
import time
import math
#--------相机----------
cap = cv2.VideoCapture(0)
if (cap.isOpened()== False): 
  print("Error opening video0 stream or file")
cap1 = cv2.VideoCapture(1)
if (cap1.isOpened()== False): 
  print("Error opening video1 stream or file")



#------------------------------------红绿灯配置----------------------------

#颜色阈值

#red1
Lower = np.array([0, 128,46])
Upper = np.array([200, 255, 255])
#green
Lower1 = np.array([35, 128, 46])
Upper1 = np.array([77, 255, 255])
#yellow
Lower2 = np.array([15, 128, 46])
Upper2 = np.array([34, 255, 255])


#--------------光流配置-------------------------
flag=1
cnt=0
frameNum = 0 

track_len = 10
detect_interval = 5
tracks = []
frame_idx = 0
fps_counter = 0
start_time = time.time()
fps=0
filter_counter = 0


feature_params = dict( maxCorners = 2,
                       qualityLevel = 0.5,
                       minDistance = 7,
                       blockSize = 7 )


lk_params = dict( winSize  = (11,11),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),minEigThreshold = 1e-4)

color = np.random.randint(0,255,(100,3))


#led_ip = ol.ip_dict['gpio_leds']
#switches_ip = ol.ip_dict['gpio_switches']
#leds = AxiGPIO(led_ip).channel1
#switches = AxiGPIO(switches_ip).channel1



#读一下图像
frame5=cv2.imread("2.jpg",1)
frame5=cv2.resize(frame5,(640,480))
while(1):
#    _ret, frame5 = cap1.read()
#    frame_r=frame5.copy()
#    frame5=cv2.resize(frame5,(640,480))

    #把RGB换成HSV 制作掩模处理，形态学滤波操作
    HSV = cv2.cvtColor(frame5, cv2.COLOR_BGR2HSV)
    HSV1=HSV.copy()
    HSV2=HSV.copy()

    mask = cv2.inRange(HSV, Lower, Upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


    mask1 = cv2.inRange(HSV1, Lower1, Upper1)
    mask1 = cv2.erode(mask1, None, iterations=2)
    mask1 = cv2.dilate(mask1, None, iterations=2)

    mask2 = cv2.inRange(HSV2, Lower2, Upper2)
    mask2 = cv2.erode(mask2, None, iterations=2)
    mask2 = cv2.dilate(mask2, None, iterations=2)


    #寻找颜色边框 参数别改
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # 遍历结构体
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]


    #寻找颜色边框 参数别改
    cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # 遍历结构体
    cnts1 = cnts1[0] if imutils.is_cv2() else cnts1[1]


    #寻找颜色边框 参数别改
    cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # 遍历结构体
    cnts2 = cnts2[0] if imutils.is_cv2() else cnts2[1]


#--------------------------------------------------




    #画图操作红色
    if len(cnts) > 0:
        print(1)
        c = max(cnts, key=cv2.contourArea)
        #max(lis, key = lambda x: x[1])
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(frame5, [box], 0, (255, 255, 255), 2)
        #对应编号 颜色 粗细
  

        while True:
            _ret, frame = cap.read()
            print(2)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
            if len(tracks) > 0:
                img0, img1 = prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                tracks = new_tracks
                cnt = len(tracks)
                speed=np.zeros((cnt,3))
                for i in range(0,len(tracks)):
                    try:
                        speed[i][0] = tracks[i][1][0] - tracks[i][0][0]
                        speed[i][1] = tracks[i][1][1] - tracks[i][0][1]
                        speed[i][2] = math.sqrt(speed[i][0]*speed[i][0] + speed[i][1]*speed[i][1])
                    except:
                        print('1')
                speed = speed[np.lexsort(-speed.T)]
                if(fps !=None and len(tracks) !=0):
                    if( len(tracks)> 3):
    #                        print(len(self.tracks))
    #                        print(len(speed)) 
                        for i in range(0,len(tracks)-1):
                            try:
    #                            print (speed[i][2] ,speed[i+1][2] ) 
                                if(speed[i][2]*0.9 >= speed[i+2][2]):
    #                                    speed = np.delete(speed, 0, 0)
                                    filter_counter = filter_counter + 1
                                else:
                                    break
                            except:
                                print('2')
    #                        speed = speed[speed[:,2] >= ( max(speed[:,2])) ]
                        speed = speed[speed[:,2] <= (speed[filter_counter,2]) ]
                        x = int(speed[0,0] * fps * 10)
                        y = int(speed[0,1] * fps * 10)
                        filter_counter=0
                    else:
    #                        speed = speed[speed[:,2] == ( max(speed[:,2])) ]
                        x = int(np.mean(speed[:,0]) * fps * 10)
                        y = int(np.mean(speed[:,1]) * fps * 10)
                    print("speedX: %.1f ,speedY: %.1f"% (x,y))
                cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
                if(True):
                    print(3)
                    frameNum += 1
                    tempframe = frame  
                    tempframe_temp= frame
                    if(frameNum==1):
                        previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)

                    if(frameNum>=2):
                        currentframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)        
                        currentframe = cv2.absdiff(currentframe,previousframe) 

                    #阈值化处理
                        ret, threshold_frame = cv2.threshold(currentframe, 20, 255, cv2.THRESH_BINARY)  #降噪过滤突兀点

                    #腐蚀与膨胀操作(卷积核大小大概这么着把)
                        kernel_erosion= cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        kernel_dilate=cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

                        threshold_frame=cv2.erode(threshold_frame,kernel_erosion)
                        threshold_frame=cv2.dilate(threshold_frame,kernel_dilate)

                    #显示膨胀腐蚀之后的图像
                        #cv2.imshow('threshold',threshold_frame)
                    #轮廓检测，准备框选
                        #if(x!=0):
                        cnts= cv2.findContours(threshold_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

                        cnts=cnts[0] if imutils.is_cv2() else cnts[1]
                        if len(cnts)>0:
                            c=max(cnts,key=cv2.contourArea)

                            rect=cv2.minAreaRect(c)
                            box=np.int0(cv2.boxPoints(rect))

                            cv2.drawContours(vis,[box],0,(0,255,0),3)   #画出轮廓
                        ''' mask = 0xffffffff
                            leds.write(0xf, mask)
                            switches.read()
                            
                            switches.setdirection("in")
                            switches.setlength(3)
                            switches.read()'''
                    previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)


    #            else:
    #                cnt=cnt+1
    #                if(cnt==3):
    #                    frameNum = 0 
    #                    cnt=0


            if frame_idx % detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        tracks.append([(x, y)])
                    if(len(tracks)>20):
                        tracks = np.delete(tracks, [0,1], axis=0)
                

            frame_idx += 1
            prev_gray = frame_gray
            outframe=hdmi_out.newframe()
            #outframe[:,:,:]=vis[:,:,:]
            
            outframe[:,:,:]=vis[:,:,:]
            hdmi_out.writeframe(outframe) 

            fps_counter=fps_counter+1
            if (time.time() - start_time) > 1:
                fps=fps_counter / (time.time() - start_time)
                print("FPS: %.1f"%(fps))
                fps_counter = 0
                start_time = time.time() 
