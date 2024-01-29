import cv2
import mediapipe as mp

mpPose = mp.solutions.pose  # 检测人的手
pose_mode = mpPose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)  # 模式参数设置
mpDraw = mp.solutions.drawing_utils  # 绘图

cap = cv2.VideoCapture(0)
biaoji = 0
i = 0

while True:
    success,img = cap.read()
    img = cv2.flip(img,1)
    results = pose_mode.process(img)
    if results.pose_landmarks:
        point23_25 = []
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx,cy = int(lm.x*w),int(lm.y*h)
            cv2.circle(img,(cx,cy),10,(255,0,0),-1)
            if id in [23,25]:
                point23_25.append([cx,cy])
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        cv2.line(img,(point23_25[0][0],point23_25[0][1]),(point23_25[1][0],point23_25[1][1]),(0,0,255),5)
        if point23_25[0][1]>point23_25[1][1]:
            if biaoji == 1:
                i += 1
                biaoji = 0
                cv2.putText(img,"Leg up--{}".format(i),(10,50),cv2.FONT_HERSHEY_PLAIN, 3,(0,0,255),3)
        else:
            biaoji = 1
            cv2.putText(img,"Leg down--{}".format(i),(10,450),cv2.FONT_HERSHEY_PLAIN, 3,(0,0,255),3)
    cv2.imshow("img",img)
    if cv2.waitKey(1)&0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
