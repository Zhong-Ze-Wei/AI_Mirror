import os

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置log level，2代表只显示error信息

import pymysql

# 建立数据库连接
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='1234',
    database='gym'
)

# 加载模型
model = load_model('../model/inference_model_0.993.h5')

# 加载测试图片并进行预处理
img1 = Image.open('../data/face/1/1.jpg').resize((96, 112))
img2 = Image.open('../data/face/1/2.jpg').resize((96, 112))

# 提取人脸特征向量
emb1 = model.predict(np.expand_dims(img1, axis=0))[0]
emb2 = model.predict(np.expand_dims(img2, axis=0))[0]

# 计算余弦相似度
def sim(emb1,emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# 尝试：
c = conn.cursor()
c.execute('SELECT emb FROM face')
rows = c.fetchall()
emb_list = [np.frombuffer(row[0]) for row in rows]
for i in emb_list:
    print(i)
    print(emb1)
    score=sim(i,emb1)
    print(score)


# print("Similarity score: ", sim(emb1,emb2))
