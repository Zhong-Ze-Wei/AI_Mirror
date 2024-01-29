"""
这个是手动选择的界面，用于直接选择视频进行运动，需要导入手部算法
"""
import cv2
import hand_module as htm
import autopy
import numpy as np
import time
import os
import keep_sport as ks
def videos():
    ##############################参数设置：##############################################
    # 路径参数
    img_dir = 'data/videos/img'
    # 获取图片文件名列表
    img_files = os.listdir(img_dir)
    # 初始页数
    num = 1
    # 每页显示的图片数量
    per_page = 4
    # 图片尺寸
    img_size = (300, 300)
    # 图片间隔
    spacing = 50
    # 视频宽和高
    wCam, hCam = 1080, 720
    # 识别范围，为视角内100单位
    frameR = 100
    # 鼠标灵敏度 越小越快 初始5
    smoothening = 3
    # 鼠标点击确认范围 越大越灵敏
    dis_choose = 40
    # 若使用笔记本自带摄像头则编号为0  若使用外接摄像头 则更改为1或其他编号
    cap = cv2.VideoCapture(0)
    # 手部伸出动作方法模式 1：大拇指看远指骨，其他看近指骨  2：都看近指骨  3：都看远指骨 4:大拇指角度 5：全角度
    hand_model = 5
    # 按比例设置
    cap.set(3, wCam)
    cap.set(4, hCam)
    # 初始化坐标点
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0
    # 初始化时间
    last_gesture_time = 0
    # 初始化手部方法
    detector = htm.handDetector()  # 类的元素
    # 得到电脑大小用于映射鼠标
    wScr, hScr = autopy.screen.size()  # 得到整个电脑的区域 2560.0 1080.0 方便映射鼠标
    ################################参数写入完成############################################

    def choose(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and x > 0 and y > 0 and x < wCam // 2 and y < hCam // 2:
            cv2.waitKey(1)
            cv2.destroyAllWindows()  # 关闭当前窗口
            ks.sport(str(num * 4 - 3), 1)
            cv2.waitKey(0)
        elif event == cv2.EVENT_LBUTTONDOWN and x > wCam // 2 and y > 0 and x < wCam and y < hCam // 2:
            cv2.waitKey(1)
            cv2.destroyAllWindows()  # 关闭当前窗口
            ks.sport(str(num * 4 - 2), 1)
            cv2.waitKey(0)
        elif event == cv2.EVENT_LBUTTONDOWN and x > 0 and y > hCam // 2 and x < wCam // 2 and y < hCam:
            cv2.waitKey(1)
            cv2.destroyAllWindows()  # 关闭当前窗口
            ks.sport(str(num * 4 - 1), 1)
            cv2.waitKey(0)
        elif event == cv2.EVENT_LBUTTONDOWN and x > wCam // 2 and y > hCam // 2 and x < wCam and y < hCam:
            cv2.waitKey(1)
            cv2.destroyAllWindows()  # 关闭当前窗口
            ks.sport(str(num * 4), 1)
            cv2.waitKey(0)

    # 打开第一页的图片
    def show(page):
        page = page
        start_index = (page - 1) * per_page  # 开始参数 0
        end_index = start_index + per_page  # 结尾参数 4
        cur_img_files = img_files[start_index:end_index]
        # 创建一个大图，用来放置四张小图
        big_img = np.zeros((img_size[0] * 2 + spacing * 3, img_size[1] * 2 + spacing * 3, 3), dtype=np.uint8)
        # 遍历当前页的图片，将它们放到大图上
        for i, img_file in enumerate(cur_img_files):
            # 拼接图片的完整路径
            img_path = os.path.join(img_dir, img_file)
            # 读取图片
            img = cv2.imread(img_path)
            # 调整图片大小
            img = cv2.resize(img, img_size)
            # 计算该图片在大图中的位置
            row = i // 2
            col = i % 2
            x = col * (img_size[1] + spacing) + spacing
            y = row * (img_size[0] + spacing) + spacing
            # 将该图片放到大图对应的位置上 #这时得到四张初试图片 big_img
            big_img[y:y + img_size[0], x:x + img_size[1], :] = img
        return big_img
    # 等待用户按下按键
    while True:
        page=show(num)
        success, img = cap.read()
        # 页面翻转
        img = cv2.flip(img,1)
        # 检测手部 得到手指关键点坐标
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        # 查看时间戳
        current_time = time.time()

        # 判断五个手指是否伸出，伸出则白色
        if len(lmList) != 0:
            # 返回一个列表，共五位，分别为五个手指是否伸出的true or false，参数方法1，2，3，4
            fingers = detector.fingersUp(hand_model) # 返回一个列表，共五位，分别为五个手指是否伸出的true or false
            # print(fingers)
            for i in range(0,5,1):
                if fingers[i]== 1:
                    cv2.circle(img, (lmList[(i+1)*4][1],lmList[(i+1)*4][2]), 10, (255, 255, 255), cv2.FILLED)

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

            # 动作3、4：向上向下          若只有大拇指伸出，则判断方向
            if fingers == [1,0,0,0,0]:
                x4=lmList[4][1]
                y4=lmList[4][2]
                x3=lmList[3][1]
                y3=lmList[3][2]
                # 向上
                print(current_time - last_gesture_time)
                if y4<y3 :
                    cv2.putText(img, 'up', (15, 160), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                    cv2.circle(img, (lmList[4][1], lmList[4][2]), 15, (255, 0, 255), cv2.FILLED)
                    if current_time - last_gesture_time >=  0.5: # 时间戳0.5秒保证准确性
                        last_gesture_time = current_time #还原时间
                        ############################向上一页################################
                        if num>1:
                            num-=1
                            page = show(num)
                # 向下
                elif y4>y3:
                    cv2.putText(img, 'down', (15, 160), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                    cv2.circle(img, (lmList[4][1], lmList[4][2]), 15, (255, 0, 255), cv2.FILLED)
                    if current_time - last_gesture_time >=  0.5:# 时间戳0.5秒保证准确性
                        last_gesture_time = current_time  #还原时间
                        ###########################向下一页#################################
                        if num<2:
                            num+=1
                            page = show(num)
                        print("向下一页")

        page = cv2.resize(page, (img.shape[1], img.shape[0]))  # 把背景图做到和手势页面一样大小
        addimg = cv2.addWeighted(img, 0.1, page, 0.9, 0)  # 把背景图覆盖到手势页面上

        cv2.imshow("img", addimg)
        cv2.setMouseCallback('img', choose)#设置写入的鼠标方法

        # 如果用户按下"退出"按键
        if cv2.waitKey(1) == ord('q'):
            break

    # 关闭所有窗口
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 调用main函数，程序从这里开始运行
    videos()