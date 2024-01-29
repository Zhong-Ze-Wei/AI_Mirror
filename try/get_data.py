import numpy as np
import os
import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置log level，2代表只显示error信息

import pymysql

# 建立数据库连接
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='1234',
    database='gym'
)


c = conn.cursor()
c.execute('SELECT emb FROM face')
rows = c.fetchall()
emb_list = [np.frombuffer(row[0]) for row in rows]
for i in emb_list:
    print(i)