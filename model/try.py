import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# 加载模型
model = load_model('../model/inference_model_0.993.h5')

# 加载测试图片并进行预处理
img1 = Image.open('../data/face/1/01003.jpg').resize((96, 112))
img2 = Image.open('../data/face/1/01004.jpg').resize((96, 112))

# 提取人脸特征向量
emb1 = model.predict(np.expand_dims(img1, axis=0))[0]
emb2 = model.predict(np.expand_dims(img2, axis=0))[0]

# 计算余弦相似度
similarity_score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

print("Similarity score: ", similarity_score)
