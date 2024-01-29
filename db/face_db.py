# 在运行本项目前，需要建立数据库gym
import mysql.connector

# 连接数据库
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="123456"
)

# 创建数据库
cursor = db.cursor()
cursor.execute("CREATE DATABASE IF NOT EXISTS gym")

# 连接“健身”数据库
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="123456",
    database="gym"
)

# 创建“names”表
cursor = db.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS names (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255))")

# 创建“videos”表
cursor = db.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS videos (id INT AUTO_INCREMENT PRIMARY KEY, title VARCHAR(255), url VARCHAR(255))")

# 创建表格
mycursor = db.cursor()
mycursor.execute("CREATE TABLE IF NOT EXISTS face (id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,img_path VARCHAR(255) NOT NULL,emb BLOB NOT NULL)")

print("ok")