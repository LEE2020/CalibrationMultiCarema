import cv2
import numpy as np

# 读取图像
img1 = cv2.imread('./captured_images/camera_1.jpg', cv2.IMREAD_GRAYSCALE)  # 查询图像
img2 = cv2.imread('./captured_images/camera_5.jpg', cv2.IMREAD_GRAYSCALE)  # 训练图像
# 图像增强：使用 CLAHE 提高对比度
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
img1 = clahe.apply(img1)
img2 = clahe.apply(img2)

# 使用 Canny 边缘检测提取边缘区域
edges1 = cv2.Canny(img1, 50, 150)
edges2 = cv2.Canny(img2, 50, 150)

# 显示边缘检测结果
cv2.imshow('Edges Image 1', edges1)
cv2.imshow('Edges Image 2', edges2)

# 计算图像的特征点和描述符（使用ORB）
orb = cv2.ORB_create(nfeatures=500)  # 设置较多特征点
kp1, des1 = orb.detectAndCompute(img1, edges1)  # 使用边缘图像来计算特征点
kp2, des2 = orb.detectAndCompute(img2, edges2)  # 使用边缘图像来计算特征点
des1 = des1.astype(np.float32)
des2 = des2.astype(np.float32)
# 创建 FLANN 匹配器
index_params = dict(algorithm=1, trees=10)
search_params = dict(checks=50)  # 设置搜索的检查次数

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# 通过比率测试过滤匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:  # 比率测试
        good_matches.append(m)

# 可视化匹配点
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("FLANN Matches", img_matches)

# 计算变换矩阵并进行图像拼接
if len(good_matches) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 使用RANSAC算法估算单应性矩阵（透视变换矩阵）
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 将第二张图像映射到第一张图像的坐标系
    img2_warped = cv2.warpPerspective(img2, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))

    # 拼接两张图像
    result = np.copy(img2_warped)
    result[0:img1.shape[0], 0:img1.shape[1]] = img1

    # 显示拼接后的结果
    cv2.imshow('Stitched Image', result)
else:
    print("匹配点太少，无法进行拼接")

# 等待按键退出
cv2.waitKey(0)
cv2.destroyAllWindows()