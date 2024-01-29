"""
通过这个代码，我们可以读取人脸区域图片并将特征向量保存到本地数据库中（调用模型）.运行主程序前需要先运行这个模块，初始化本地图片
"""
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
import pymysql

# 建立数据库连接
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='123456',
    database='gym'
)

# 载入模型
model = load_model('../model/inference_model_0.993.h5')

# 函数输入图片路径和模型对象，返回该图片的特征向量emb。
def get_emb(img_path):
    # 加载图片并进行预处理
    img = Image.open(img_path)
    # 提取人脸特征向量
    emb = model.predict(np.expand_dims(img, axis=0))[0]
    # 返回特征向量
    return emb

# 遍历文件夹下所有图片
folder_path = '../data/face/'
for file_name1 in os.listdir(folder_path): # 分别读取到文件夹1，2，3，代表user1，2，3
    # 获取文件路径
    img_file = os.path.join(folder_path, file_name1) #完整路径名‘../data/face/1’
    for file_name2 in os.listdir(img_file):
        if file_name2.endswith('.jpg'):
            # 获取图片路径
            img_path = os.path.join(img_file, file_name2)
            # 获取特征向量
            emb = get_emb(img_path)
            # 保存到mysql
            with conn.cursor() as cursor:
                sql = "INSERT INTO face(img_path, emb) VALUES (%s, %s)"
                cursor.execute(sql, (file_name1, emb.tostring()))
            conn.commit()

print('运行成功')
