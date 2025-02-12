import cv2
import numpy as np
import os

# 六个摄像头的RTSP流地址
RTSP_URLS = [
    'rtsp://admin:xsznaf888@192.168.1.64/LiveMedia/ch1/Media1',
    'rtsp://admin:xsznaf888@192.168.1.65/LiveMedia/ch1/Media1',
    'rtsp://admin:xsznaf888@192.168.1.66/LiveMedia/ch1/Media1',
    'rtsp://admin:xsznaf888@192.168.1.67/LiveMedia/ch1/Media1',
    'rtsp://admin:xsznaf888@192.168.1.68/LiveMedia/ch1/Media1',
    'rtsp://admin:xsznaf888@192.168.1.69/LiveMedia/ch1/Media1'
]

# 创建保存图像的文件夹
output_dir = "captured_images"
os.makedirs(output_dir, exist_ok=True)

# 打开每个RTSP流
caps = [cv2.VideoCapture(url) for url in RTSP_URLS]

# 检查每个摄像头是否成功打开
for idx, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"无法连接到摄像头 {idx+1}")
        continue

# 从每个摄像头获取图像并保存
for idx, cap in enumerate(caps):
    ret, frame = cap.read()
    if ret:
        # 保存图像到文件
        img_filename = os.path.join(output_dir, f"camera_{idx+1}.jpg")
        cv2.imwrite(img_filename, frame)
        print(f"摄像头 {idx+1} 图像已保存: {img_filename}")
    else:
        print(f"从摄像头 {idx+1} 获取图像失败")

# 释放摄像头
for cap in caps:
    cap.release()

# 打印完成消息
print("所有图像已保存完毕")
