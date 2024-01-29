"""
通过这个代码，我们可以得到照片的人脸部分并且返回成同样的大小96x112（需要的输入大小）
"""
import os
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 定义需要读取的目录和保存结果的目录
input_dir = "../data/face_data/01"
output_dir = "../data/face/1"

# 定义需要统一的人脸图片大小
face_size = (96, 112)

# 初始化人脸检测器
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:

    # 遍历目录中的所有图片文件
    for filename in os.listdir(input_dir):
        if not filename.endswith(".jpg") and not filename.endswith(".jpeg") and not filename.endswith(".png"):
            continue
        # 读取图片文件
        image = cv2.imread(os.path.join(input_dir, filename))
        # 调整图片大小
        image = cv2.resize(image, (640, 480))
        # 将图片转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 检测人脸
        results = face_detection.process(image)
        #print(results.detections.location_data)
        # 遍历检测到的所有人脸
        if results.detections is not None:
            for detection in results.detections:
            # 获取人脸在图片中的位置
                bounding_box = detection.location_data.relative_bounding_box
                x, y, w, h = int(bounding_box.xmin * image.shape[1]), int(bounding_box.ymin * image.shape[0]), \
                            int(bounding_box.width * image.shape[1]), int(bounding_box.height * image.shape[0])

                # 截取人脸图像
                face_image = image[y:y+h, x:x+w]
                # 调整人脸图像大小
                face_image = cv2.resize(face_image, face_size)
                # 保存人脸图像
                face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, filename), face_image)
print('已经全部转换完成')
