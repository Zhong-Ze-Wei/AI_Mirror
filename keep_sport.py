"""
运动模式主体
videos是采用的视频，用文件名读取
model为使用的模式，上下或左右
刚刚进入时不会开始运动，而是先进行调节，双手合十才开始运动
"""


import cv2
import mediapipe as mp
import math
import time

def length(a,b):
    return int(math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2))

def sport(videos,model): # num为是使用的视频,model为使用的模式，1为并行（左右），2为并列（上下）
    ########################初始化监测模块####################################
    # 初始化MediaPipe的Pose模块
    mp_pose = mp.solutions.pose
    # 视频流人体监测
    pose1 = mp_pose.Pose(min_detection_confidence=0.7,min_tracking_confidence=0.7,
                        static_image_mode=False)
    # 本地流人体监测
    pose2 = mp_pose.Pose(min_detection_confidence=0.7,min_tracking_confidence=0.7,
                        static_image_mode=False)
    mp_drawing = mp.solutions.drawing_utils
    #########################参数写入#############################
    # 选择播放视频
    number=str(videos)
    # 缩放比例 （缩小倍数,越小运行速度越快）
    rate =2
    # 本地视频比例
    rate2=1
    # 运动距离长度和
    score2_1=0
    score2_2=0
    # 动作成绩
    old_position1=[]
    old_position2=[]
    # 视频目录地址
    video_path=('data/videos/')
    # 初始化视频开始情况
    start=False
    # 初始化时间参数
    start_time=time.time()
    #########################读取信息#####################################
    # 初始化摄像头和本地视频
    cap = cv2.VideoCapture(0)
    video = cv2.VideoCapture(video_path+number+'_.mp4')
    #video = cv2.VideoCapture('data/videos/2.mp4')
    if start==False:
        ret_local, frame2 = video.read()
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)


    while True:
        try:
            # 读取摄像头视频帧
            ret, frame1 = cap.read()
            # 读取本地视频帧
            if start:
                ret_local, frame2 = video.read()

            if not ret_local or not ret:
                # 将本地视频指针设置到开头，以便下一次播放
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if start:
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            # 转换视频
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame1 = cv2.flip(frame1, 1)
            h, w = frame1.shape[:2]
            h2, w2 = frame2.shape[:2]
            frame1 = cv2.resize(frame1, (int(w//rate), int(h//rate))) # 将摄像头画面大小缩放
            if model==1:     # 并行
                frame2 = cv2.resize(frame2, (int(h*w2//h2//rate), int(h//rate))) # 将本地流与视频流对齐
            elif model==2:  # 并列
                frame2 = cv2.resize(frame2, (int(w//rate), int(h*h2//w2//rate))) # 将本地流与视频流对齐

            # 对左侧图像进行肢体检测
            results1 = pose1.process(frame1)
            if results1.pose_landmarks:
                point1=[]
                for id, lm in enumerate(results1.pose_landmarks.landmark):
                    h, w, c = frame1.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    point1.append([cx,cy])
                    if id==0: # 头部标记
                        cv2.circle(frame1, (cx, cy), 10, (255, 0, 0), -1)
                    if id in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]: #肢体点标记
                        cv2.circle(frame1, (cx, cy), 5, (255, 0, 0), -1)
                point1.append([(point1[23][0]+point1[24][0]+point1[11][0]+point1[12][0])//4,
                               (point1[23][1]+point1[24][1]+point1[11][1]+point1[12][1])//4])  #添加居中点
                cv2.circle(frame1, (point1[33][0], point1[33][1]), 10, (255, 0, 0), 2)

                if len(old_position1)!=0 and len(point1)!=0:
                    for i in [11,12,13,14,15,16,23,24,25,26,27,28]:
                        score2_1+=length(point1[i],old_position1[i])
                    old_position1 = point1
                else:
                    old_position1 = point1

            # 设置还在预备阶段时候的调节
            if not start: #还在预备阶段
                stop_time = time.time()
                # 操作1：左手升起放大
                if point1[15][1]>point1[0][1] and point1[16][1]<point1[0][1] :
                    cv2.circle(frame1, (point1[16][0], point1[16][1]), 10, (255, 255, 0), -1)
                    if stop_time-start_time>=0.4:
                        if rate2<10:
                            rate2+=0.1
                            print("放大",rate2)
                            start_time=time.time()
                # 操作2：右手升起减小
                if point1[16][1]>point1[0][1] and point1[15][1]<point1[0][1] :
                    cv2.circle(frame1, (point1[15][0], point1[15][1]), 10, (255, 255, 0), -1)
                    if stop_time-start_time>=0.4:
                        if rate2 > 0.5:
                            rate2-=0.1
                            print("缩小",rate2)
                            start_time=time.time()
                # 操作三：双手合并开始运动
                if abs(point1[19][1]-point1[20][1])<10 and abs(point1[19][0]-point1[20][0])<10:
                    time.sleep(1)
                    start=True

            # 对右侧图像进行肢体检测
            results2 = pose2.process(frame2)
            if results2.pose_landmarks and results1.pose_landmarks:
                point2=[]
                add_x = 0
                add_y = 0
                for id2, lm2 in enumerate(results2.pose_landmarks.landmark): # 第一次循环用来求距离，计算偏移程度
                    if id2 in[11,12,23,24]:
                        h, w, c = frame2.shape
                        cx, cy = int(lm2.x * w), int(lm2.y * h)
                        add_x += cx
                        add_y += cy
                change_x = point1[33][0] - add_x // 4
                change_y = point1[33][1] - add_y // 4

                for id2, lm2 in enumerate(results2.pose_landmarks.landmark): # 第二次循环计算偏移坐标
                    h, w, c = frame2.shape
                    cx, cy = int(lm2.x * w)+change_x, int(lm2.y * h)+change_y
                    cx = int(point1[33][0] + (cx - point1[33][0]) * rate2)
                    cy = int(point1[33][1] + (cy - point1[33][1]) * rate2)
                    print(cx,cy)
                    point2.append([cx,cy])

                if len(old_position2)!=0:
                    for i in [11,12,13,14,15,16,23,24,25,26,27,28]:
                        score2_2+=length(point2[i],old_position2[i])
                    old_position2=point2
                else:
                    old_position2 = point2
                # 绘制右侧图像的关键点和骨骼框架
                try:
                    # 手部五条线
                    cv2.line(frame1, (point2[11][0], point2[11][1]), (point2[12][0], point2[12][1]), (0, 0, 255), 2)
                    cv2.line(frame1, (point2[11][0], point2[11][1]), (point2[13][0], point2[13][1]), (0, 0, 255), 2)
                    cv2.line(frame1, (point2[13][0], point2[13][1]), (point2[15][0], point2[15][1]), (0, 0, 255), 2)
                    cv2.line(frame1, (point2[12][0], point2[12][1]), (point2[14][0], point2[14][1]), (0, 0, 255), 2)
                    cv2.line(frame1, (point2[14][0], point2[14][1]), (point2[16][0], point2[16][1]), (0, 0, 255), 2)
                    #肢体三条线
                    cv2.line(frame1, (point2[11][0], point2[11][1]), (point2[23][0], point2[23][1]), (0, 0, 255), 2)
                    cv2.line(frame1, (point2[12][0], point2[12][1]), (point2[24][0], point2[24][1]), (0, 0, 255), 2)
                    cv2.line(frame1, (point2[24][0], point2[24][1]), (point2[23][0], point2[23][1]), (255, 0, 0), 2)
                    #腿部六条线
                    cv2.line(frame1, (point2[24][0], point2[24][1]), (point2[26][0], point2[26][1]), (0, 0, 255), 2)
                    cv2.line(frame1, (point2[26][0], point2[26][1]), (point2[28][0], point2[28][1]), (0, 0, 255), 2)
                    cv2.line(frame1, (point2[28][0], point2[28][1]), (point2[32][0], point2[32][1]), (0, 0, 255), 2)
                    cv2.line(frame1, (point2[23][0], point2[23][1]), (point2[25][0], point2[25][1]), (0, 0, 255), 2)
                    cv2.line(frame1, (point2[25][0], point2[25][1]), (point2[27][0], point2[27][1]), (0, 0, 255), 2)
                    cv2.line(frame1, (point2[27][0], point2[27][1]), (point2[31][0], point2[31][1]), (0, 0, 255), 2)
                except:
                    print("无法划线")

            # 计算实时的准确度
            # 动作成绩初始化
            score1 = 0
            if start:
                for i in [11,12,13,14,15,16,23,24,25,26,27,28]:
                    score1+=length(point1[i],point2[i])
                if score1 <= 100:
                    level = 'great'
                elif score1<=150 and score1>100:
                    level='great'
                elif score1>150 and score1<=250:
                    level='good'
                elif score1 > 250 and score1 <= 350:
                    level = 'not good'
                elif score1 > 350:
                    level = 'bad'

            # 将左右两个图像拼接在一起
            if model==1:# 并行
                frame = cv2.hconcat([frame1, frame2])
            elif model==2: # 并列
                frame = cv2.vconcat([frame1, frame2])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame,(frame.shape[1]*rate,frame.shape[0]*rate))

             # 计算结果：根据score输出level
            if start:
                cv2.putText(frame, 'Level:'+level , (15, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            #cv2.putText(frame, 'Calorie:200KJ  100%' , (15, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if start:
                cv2.putText(frame, 'Calorie: '+str( "{:.3f}".format((score2_1/400000)))+'KJ', (15, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            cv2.imshow('Pose Detection', frame)

            # 等待按下ESC键退出程序
            if cv2.waitKey(1) == ord('q'):
                break
        except:
            print("发生某些错误")


    # 释放资源
    cap.release()
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 调用main函数，程序从这里开始运行
    sport('1',1)
