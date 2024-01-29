import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keep_sport
import time
import pymysql
import cv2

# 建立数据库连接
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='123456',
    database='gym'
)
def face():
    # 初始化 Mediapipe 的人脸检测器
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.95)

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    # 加载模型
    model = load_model('model/inference_model_0.993.h5')

    # 模型一一对比方法
    def sim(emb1,emb2):
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    # 初始化参数：
    name=[]
    start_time = time.time()

    # 模型总体对比方法
    def face_choose(emb):
        score = 0
        c = conn.cursor()
        print(c)
        c.execute('SELECT emb FROM face')
        rows = c.fetchall()
        emb_list = [np.frombuffer(row[0], dtype=np.float32) for row in rows]
        for i in emb_list:
            if sim(emb, i)>score:
                score=sim(emb, i)
                emb_great=i
                print(score)
        with conn.cursor() as cursor:
            sql = "SELECT img_path FROM face WHERE emb =%s;"
            cursor.execute(sql, emb_great.tostring())
            result = cursor.fetchone()
        return [result ,score]

    # 进入新方法

    time.sleep(2)
    while True:
        try:
            # 读取摄像头的帧
            ret, frame = cap.read()
            # 对帧进行预处理
            frame = cv2.flip(frame, 1) # 水平翻转帧，使镜像效果更好
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 将BGR图像转换为RGB图像，以便于后续的处理

            # 利用 Mediapipe 的人脸检测器检测人脸
            results = face_detection.process(rgb_frame)

            # 如果检测到人脸，则提取人脸图像并显示出来
            if results.detections:
                for detection in results.detections:
                    # 获取人脸框的位置
                    bbox = detection.location_data.relative_bounding_box
                    x, y, w, h = int(bbox.xmin * frame.shape[1]), int(bbox.ymin * frame.shape[0]), \
                                 int(bbox.width * frame.shape[1]), int(bbox.height * frame.shape[0])
                    # 提取人脸图像
                    face_img = frame[y:y+h, x:x+w, :]
                    face_img = cv2.resize(face_img, (96, 112))
                    # 提取人脸特征向量
                    emb = model.predict(np.expand_dims(face_img, axis=0))[0]
                    # 读取实际人脸
                    user_num=face_choose(emb)[0][0]
                    user_score=face_choose(emb)[1]
                    print('user_num',user_num)
                    print('user_score',user_score)
                    if user_num not in name:
                        name.append(user_num)
                    end_time = time.time()
                    if end_time-start_time>1 :
                        if len(name)==1:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, 'user:'+str(name[0]), (int(x+w/2), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3) #在人脸框上输出名字
                        elif len(name)!=1:
                            start_time= time.time()
                            name=[]
                    if end_time - start_time > 6:
                        if len(name)==1:
                            if name[0]=='1':
                                print("已经识别出人物1")
                                cv2.waitKey(1)
                                cap.release()
                                cv2.destroyAllWindows()  # 关闭当前窗口
                                keep_sport.sport('1',1)
                                cv2.waitKey(0)
                                break
                            if name[0]=='2':
                                print("已经识别出人物2")
                                cv2.waitKey(1)
                                cap.release()
                                cv2.destroyAllWindows()  # 关闭当前窗口
                                keep_sport.sport('2',1)
                                cv2.waitKey(0)
                                break
                # 在原始帧上显示检测结果
            cv2.imshow("Frame", frame)

            # 按下 q 键退出循环
            if cv2.waitKey(1) == ord('q'):
                break
        except:
            continue
if __name__ == "__main__":
    # 调用main函数，程序从这里开始运行
    face()