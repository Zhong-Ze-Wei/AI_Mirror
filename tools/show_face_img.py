import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.rcParams['font.family'] = 'SimHei'  # 使用中文字体

path='../data/face/01/'
# 修改num为1，2，3,4 分别对应模式1234
num=1
# 图片路径列表
image_paths = ["003.jpg", "004.jpg",
               "005.jpg", "006.jpg"]

# 创建一个2行4列的子图
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5, 7))

# 将每张图片依次显示在对应的子图上
for i, ax in enumerate(axes.flat):
    # 判断图片路径是否存在
    if i < len(image_paths):
        img = mpimg.imread(path+"0"+str(num)+image_paths[i])
        ax.imshow(img)
        ax.set_axis_off()  # 关闭坐标轴
    else:
        break
plt.suptitle('user:01'+'的照片展示', fontsize=20)
plt.tight_layout()
plt.show()