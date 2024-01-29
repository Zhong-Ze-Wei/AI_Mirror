'''
这是主页，进入软件
HandTrackingModule是全局依赖，用于操控鼠标
face_module是脸部依赖，用于选择视频
videos_choose是视频选择，用于直接进入训练 bigdata
keep_sport是运动模块，用于索引跟踪 Blazepose
'''

# -*- coding: utf-8 -*-

import cv2
import hand_module as htm
import autopy
import numpy as np
import time
import face_module as fm
import pyautogui
import videos_choose as vs

##########################################################################################
# 参数设置：

wCam, hCam = 1080, 720 # 视频宽和高
frameR = 100# 识别范围，为视角内100单位
smoothening = 3# 鼠标灵敏度 越小越快 初始5
dis_choose = 40 # 鼠标点击确认范围 越大越灵敏
cap = cv2.VideoCapture(0)  # 若使用笔记本自带摄像头则编号为0  若使用外接摄像头 则更改为1或其他编号
w0, h0 = 500, 200  # 选项大小
w1, h1 = 750, 150  # 选项1位置
w2, h2 = 750, 450  # 选项2位置
hand_model = 5  # 手部伸出动作方法模式 1：大拇指看远指骨，其他看近指骨  2：都看近指骨  3：都看远指骨 4:大拇指角度 5：全角度
# 按比例设置
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
#参数写入完成
#########################################################################################

# 其他设置初始化与读取

# 读取page页面
page = cv2.imread("./data/main/bg.jpg", cv2.IMREAD_UNCHANGED)
last_gesture_time = 0
detector = htm.handDetector()  # 类的元素
wScr, hScr = autopy.screen.size()  # 得到整个电脑的区域 2560.0 1080.0 方便映射鼠标

# 函数方法 转换页面：
def button1_window(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and x > w1 and y > h1 and x < w0+w1 and y < h0+h1:
        cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()  # 关闭当前窗口
        fm.face()
        cv2.waitKey(0)

    if event == cv2.EVENT_LBUTTONDOWN and x > w2 and y > h2 and x < w0+w2 and y < h0+h2:
        cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()  # 关闭当前窗口
        vs.videos()
        cv2.waitKey(0)


while True:
    success, img = cap.read()
    # 页面翻转
    img = cv2.flip(img,1)
    # 检测手部 得到手指关键点坐标
    img = detector.findHands(img)
    # 查看时间戳
    current_time = time.time()

    lmList = detector.findPosition(img, draw=False)
    # 判断五个手指是否伸出，伸出则白色
    if len(lmList) != 0:
        # 返回一个列表，共五位，分别为五个手指是否伸出的true or false，参数方法1，2，3，4
        fingers = detector.fingersUp(hand_model) # 返回一个列表，共五位，分别为五个手指是否伸出的true or false
        # print(fingers)
        for i in range(0,5,1):
            if fingers[i]== 1:
                cv2.circle(img, (lmList[(i+1)*4][1],lmList[(i+1)*4][2]), 10, (255, 255, 255), cv2.FILLED)
                cv2.putText(img, 'model:'+ str(int(hand_model)), (15, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


        # 动作1：移动鼠标
        if fingers==[0,1,0,0,0] or fingers==[1,1,0,0,0]:  #如果食指出其他手指不出
            # 坐标转换： 将食指在窗口坐标转换为鼠标在桌面的坐标
            # 鼠标坐标 参数中前者是食指坐标，后者是鼠标坐标
            x8 = np.interp(lmList[8][1], (frameR, wCam - frameR), (0, wScr))
            y8 = np.interp(lmList[8][2], (frameR, hCam - frameR), (0, hScr))
            # smoothening values 鼠标移动灵敏度，越小越高
            clocX = plocX + (x8 - plocX) / smoothening
            clocY = plocY + (y8 - plocY) / smoothening
            # 移动鼠标
            autopy.mouse.move(clocX, clocY)
            cv2.putText(img, 'Moving', (15, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.circle(img, (lmList[8][1], lmList[8][2]), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 动作2：点击鼠标
        if fingers==[0,1,1,0,0]or fingers==[1,1,1,0,0]: # 若只是食指和中指都伸出 则检测指头距离 距离够短则对应鼠标点击
            length, img, pointInfo = detector.findDistance(8, 12, img)
            if length < dis_choose:
                cv2.circle(img, (pointInfo[4], pointInfo[5]),15, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, 'Click', (15, 160), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

                if current_time - last_gesture_time >= 0.5:
                    last_gesture_time = current_time  # 还原时间
                    print("点击鼠标")
                    autopy.mouse.click()

        # 动作3、4：向上向下          若只有大拇指伸出，则判断方向con
        if fingers == [1,0,0,0,0]:
            x4=lmList[4][1]
            y4=lmList[4][2]
            x3=lmList[3][1]
            y3=lmList[3][2]
            # 向上
            print(current_time - last_gesture_time)
            if x4>x3 :
                cv2.putText(img, 'up', (15, 160), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                cv2.circle(img, (lmList[4][1], lmList[4][2]), 15, (255, 0, 255), cv2.FILLED)
                if current_time - last_gesture_time >=  0.5:
                    last_gesture_time = current_time #还原时间
                    pyautogui.press("up")
                    print("写入up")
            # 向下
            elif x4<x3:
                cv2.putText(img, 'down', (15, 160), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                cv2.circle(img, (lmList[4][1], lmList[4][2]), 15, (255, 0, 255), cv2.FILLED)
                if current_time - last_gesture_time >=  0.5:
                    last_gesture_time = current_time  #还原时间
                    pyautogui.press("down")
                    print("写入down")

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'fps:{int(fps)}', [15, 25],
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    page = cv2.resize(page, (img.shape[1], img.shape[0]))
    addimg = cv2.addWeighted(img, 0.1, page, 0.8, 0)


    cv2.imshow("Image", addimg)

    cv2.setMouseCallback('Image', button1_window)

    if cv2.waitKey(10) & 0xFF == 27:  # 每帧滞留15毫秒后消失，ESC键退出
        break

