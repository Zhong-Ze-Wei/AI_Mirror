import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.rcParams['font.family'] = 'SimHei'  # 使用中文字体

path='../data/hand_model_img/'
# 修改num为1，2，3,4 分别对应模式1234
num=5
# 图片路径列表
image_paths = ["-1.jpg", "-2.jpg", "-3.jpg", "-4.jpg",
               "-5.jpg", "-6.jpg", "-7.jpg", "-8.jpg",]
labels = ['手掌完全张开朝上', '竖着大拇指', '比yeah', '食指向上伸出', '握拳', '五指抓状', 'OK的姿势', '摇滚手势']

# 创建一个2行4列的子图
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 5))

# 将每张图片依次显示在对应的子图上
for i, ax in enumerate(axes.flat):
    # 判断图片路径是否存在
    if i < len(image_paths):
        img = mpimg.imread(path+str(num)+image_paths[i])
        ax.imshow(img)
        ax.set_title(labels[i])
        ax.set_axis_off()  # 关闭坐标轴
    else:
        break
plt.suptitle('模式'+str(num)+'的八种情况', fontsize=10)
plt.tight_layout()
plt.show()