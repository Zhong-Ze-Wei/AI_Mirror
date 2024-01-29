import cv2  # 导入OpenCV库
import mediapipe as mp  # 导入Mediapipe库
import time  # 导入time库，用于计时
import math  # 导入math库，用于数学运算

# -*- coding: utf-8 -*-

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.8, trackCon=0.8):
        self.mode = mode  # 是否进行跟踪模式
        self.maxHands = maxHands  # 最大检测手的数量
        self.detectionCon = detectionCon  # 检测的置信度阈值
        self.trackCon = trackCon  # 跟踪的置信度阈值
        self.mpHands = mp.solutions.hands  # 创建一个Hand对象
        self.hands = self.mpHands.Hands(
            static_image_mode=mode, # 如果是图片则为True，否则为False
            max_num_hands=maxHands, # 检测的最大手的数量
            min_detection_confidence=detectionCon, # 最小检测置信度
            min_tracking_confidence=trackCon# 最小跟踪置信度
        )
        self.mpDraw = mp.solutions.drawing_utils  # 创建一个DrawingUtils对象
        self.tipIds = [4, 8, 12, 16, 20]  # 指尖的id号，用于计算手指的状态

    #检测图像中的手，并可视化检测结果，并画出连接线
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像从BGR格式转换为RGB格式
        self.results = self.hands.process(imgRGB)  # 对RGB图像进行手部检测

        # print(self.results.multi_handedness)  # 获取检测结果中的左右手标签并打印

        if self.results.multi_hand_landmarks:  # 如果检测到了手
            for handLms in self.results.multi_hand_landmarks:  # 对于每一只检测到的手
                if draw:  # 如果需要可视化检测结果
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  # 绘制手部关键点和连接线
        return img  # 返回绘制了关键点和连接线的图像


    #检测图像中所有手指坐标，并画出点
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList

    # 检测手指是否伸出 共四个方法
    def fingersUp(self,num):
        fingers = []

        def angle(A, B, C):
            """计算三个点 A, B, C 所在的角度，单位为度"""
            vec1 = [B[0] - A[0], B[1] - A[1]]
            vec2 = [B[0] - C[0], B[1] - C[1]]
            norm1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
            norm2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)
            if norm1 == 0 or norm2 == 0:
                return 0  # 当向量模为0时，直接返回0度
            cos_theta = sum([vec1[i] * vec2[i] for i in range(2)]) / (norm1 * norm2)
            try:
                theta = math.degrees(math.acos(cos_theta))
            except ValueError:
                # 当acos参数不在[-1, 1]范围内时，设角度为默认值（90度）
                theta = 170
            return theta

        # 方法1，大拇指指尖在指尖在远指骨间关节下方，其他指尖在近指骨关节下方。
        if num==1:
            # 检测大拇指是否伸出 单独列出是因为大拇指只有三个关键点，而其他手指有四个关键点，只考虑手向上的情况
            if self.lmList[self.tipIds[0]][2] < self.lmList[self.tipIds[0] - 1][2]:
                fingers.append(1)
            else:
                fingers.append(0)
            # 检测其他手指是否伸出
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # 统计伸出的手指数
            # totalFingers = fingers.count(1)
            return fingers

        # 方法2，所有指尖在近指骨关节下方。
        if num==2:
            for id in range(0, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 1][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # 统计伸出的手指数
            # totalFingers = fingers.count(1)
            return fingers

        # 方法3，所有指尖在远指骨关节下方。
        if num == 3:
            for id in range(0, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 1][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # 统计伸出的手指数
            # totalFingers = fingers.count(1)
            return fingers

        # 方法4，大拇指根据角度判断，其他指尖在近指骨关节下方。
        if num==4:
            # 检测大拇指是否伸出 单独列出是因为大拇指只有三个关键点，而其他手指有四个关键点，只考虑手向上的情况
            a = self.lmList[self.tipIds[0] - 0][1:]
            b = self.lmList[self.tipIds[0] - 1][1:]
            c = self.lmList[self.tipIds[0] - 2][1:]
            d = self.lmList[self.tipIds[0] - 3][1:]
            theta1 = angle(a, b, c)
            theta2 = angle(b, c, d)
            if (theta1>155 or theta1<25) and (theta2 > 155 or theta1<25) :
                fingers.append(1)
            else:
                fingers.append(0)

            # 检测其他手指是否伸出
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # 统计伸出的手指数
            # totalFingers = fingers.count(1)
            return fingers

        # 方法5，全部根据角度判断
        if num==5:
            # 检测其他手指是否伸出
            for id in range(0, 5):
                a = self.lmList[self.tipIds[id] - 0][1:]
                b = self.lmList[self.tipIds[id] - 1][1:]
                c = self.lmList[self.tipIds[id] - 2][1:]
                d = self.lmList[self.tipIds[id] - 3][1:]
                theta1 = angle(a, b, c)
                theta2 = angle(b, c, d)
                if id==1:
                    if (theta1 > 150 or theta1 < 30) and (theta2 > 150 or theta1 < 30) :
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    if (theta1 > 160 or theta1 < 20) and (theta2 > 160 or theta1 < 20) and self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
            # 统计伸出的手指数
            # totalFingers = fingers.count(1)
            return fingers

    # 检测两个手指p1p2之间的距离，并且生成图像
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            # 在图像上绘制两个手指之间的线条和手指关键点
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            # 计算两个手指之间的距离
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
