import cv2

name='6'
in_path='../data/videos/'+name+'.mp4'
out_photo='../data/videos/img/'+name+'.jpg'
out_video='../data/videos/'+name+'_.mp4'
# 读取视频
cap = cv2.VideoCapture(in_path)

# 截取第一帧图像
def extract_first_frame(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        frame=cv2.resize(frame,(480,640))
        cv2.imwrite(output_path, frame)
    else:
        print("Failed")
    cap.release()
extract_first_frame(in_path,out_photo)

# 获取原始帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# 设置目标帧率
target_fps = 15

# 计算出每个目标帧之间的帧间隔
frame_interval = int(round(fps / target_fps))

# 打开输出视频文件
out = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (int(cap.get(3)), int(cap.get(4))))

# 迭代读取每个帧并保存每个第 n 帧，其中 n 是计算出的帧间隔
frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_num % frame_interval == 0:
        out.write(frame)
    frame_num += 1

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
