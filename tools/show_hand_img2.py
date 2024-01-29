import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.rcParams['font.family'] = 'SimHei'  # 使用中文字体

path='../data/hand_model_img/'
# 修改num为1，2，3 分别对应模式123
num=3
# 图片路径列表
if num==1:
    image_paths = ["2-1.jpg", "2-2.jpg", "3-1.jpg", "3-2.jpg"]
    labels = ['模式2的手掌张开', '模式2的竖大拇指', '模式3的手掌张开', '模式3的竖大拇指']
    text='大拇指在两种伸展状态下的图像展示'
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5))
elif num==2:
    image_paths = ["2-3.jpg", "2-4.jpg","2-5.jpg", "2-6.jpg", "3-3.jpg", "3-4.jpg","3-5.jpg", "3-6.jpg"]
    labels = ['模式2的比yeah', '模式2的食指伸出', '模式2的握拳', '模式2的五指抓状','模式3的比yeah', '模式3的食指伸出', '模式3的握拳', '模式3的五指抓状']
    text='大拇指在四种收缩状态下的图像展示'
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 5))
elif num == 3:
    image_paths = ["click.jpg", "moving.jpg","up.jpg", "down.jpg"]
    labels = ['点击', '移动', '向上', '向下']
    text='手指的四个模式识别情况'
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5))

# 将每张图片依次显示在对应的子图上
for i, ax in enumerate(axes.flat):
    # 判断图片路径是否存在
    if i < len(image_paths):
        img = mpimg.imread(path+image_paths[i])
        ax.imshow(img)
        ax.set_title(labels[i])
        ax.set_axis_off()  # 关闭坐标轴
    else:
        break
plt.suptitle(text, fontsize=10)
plt.tight_layout()
plt.show()