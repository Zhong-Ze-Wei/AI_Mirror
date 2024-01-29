import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 展示截图用

plt.rcParams['font.family'] = 'SimHei'  # 使用中文字体

path='../data/hand_img_show/'
# 修改num为1，2，3 分别对应模式123
num=1
image_paths = ["1.jpg", "2.jpg"]
labels = ['情况1', '情况2']

# 创建一个3行2列的子图
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# 将每张图片依次显示在对应的子图上
for i, ax in enumerate(axes.flat):
    # 判断图片路径是否存在
    if i < len(image_paths):
        img = mpimg.imread(path+image_paths[i])
        ax.imshow(img)
        ax.set_title(labels[i],fontsize=8)
        ax.set_axis_off()  # 关闭坐标轴
    else:
        break
plt.suptitle('手部关键点识别展示', fontsize=10)
plt.tight_layout()
plt.show()