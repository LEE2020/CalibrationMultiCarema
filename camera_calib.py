import cv2
import numpy as np
import os

# ================== 配置 ==================
CHECKERBOARD = (10, 10)  # 内角点数量 (rows x cols)
SQUARE_SIZE = 7  # 每个格子的实际大小（单位：毫米）
MIN_VALID_IMAGES = 30 # 每个摄像头至少有效图像数量
OUTPUT_DIR = "calibration_results"  # 保存标定结果的文件夹
RTSP_URLS = [
   # 'rtsp://admin:xsznaf888@192.168.1.64/LiveMedia/ch1/Media1',
    'rtsp://admin:xsznaf888@192.168.1.65/LiveMedia/ch1/Media1',
    #'rtsp://admin:xsznaf888@192.168.1.66/LiveMedia/ch1/Media1',
    #'rtsp://admin:xsznaf888@192.168.1.67/LiveMedia/ch1/Media1',
    #'rtsp://admin:xsznaf888@192.168.1.68/LiveMedia/ch1/Media1',
    #'rtsp://admin:xsznaf888@192.168.1.69/LiveMedia/ch1/Media1'
]
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ================== 准备3D对象点 ==================
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# ================== 模块化函数 ==================
def capture_chessboard_images(url, checkerboard, min_images):
    """采集RTSP摄像头的棋盘格图像"""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"无法打开摄像头：{url}")
        return None, None

    obj_points = []
    img_points = []
    valid_images = 0

    while valid_images < min_images:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频流帧")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

        if ret:
            # 提高角点精度，使用cornerSubPix
            corners_refined = cv2.cornerSubPix(gray, corners, (10, 10), (-1, -1), CRITERIA)
            obj_points.append(objp)
            img_points.append(corners_refined)

            # 绘制角点并显示
            cv2.drawChessboardCorners(frame, checkerboard, corners_refined, ret)
            cv2.imshow("Chessboard Detection", frame)
            valid_images += 1

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return obj_points, img_points

def calibrate_camera(obj_points, img_points, image_size):
    """执行相机标定"""
    # 使用CALIB_RATIONAL_MODEL来增强畸变模型
    #flags = cv2.CALIB_RATIONAL_MODEL  # 使用理性模型
    flags = cv2.CALIB_ZERO_TANGENT_DIST
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None, flags=flags
    )
    if ret:
        print("标定成功！")
        print(f"相机内参矩阵：\n{camera_matrix}")
        print(f"畸变系数：\n{dist_coeffs}")
        # 获取外参：旋转矩阵和平移向量
        print(f"旋转矩阵长度：",len(rvecs))
        print(f"平移矩阵长度L：",len(tvecs))
        rvecs_ = np.mean(rvecs, axis=0);
        tvecs_ = np.mean(tvecs, axis=0)
        print(f"旋转向量 (rvec): \n{rvecs_}")
        print(f"平移向量 (tvec): \n{tvecs_}")
        #for i in range(len(rvecs)):
        #    print(f"摄像头 {i + 1} 外参：")
        #    print(f"旋转向量 (rvec): \n{rvecs[i]}")
        #    print(f"平移向量 (tvec): \n{tvecs[i]}")


    else:
        print("标定失败，请检查输入数据")
    return ret, camera_matrix, dist_coeffs,rvecs_,tvecs_

def save_calibration_results(output_dir, camera_matrix, dist_coeffs,rvecs_,tvecs_):
    """保存标定结果"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "camera_calib_1.npz")
    np.savez(filepath, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,rvecs = rvecs_, tvecs = tvecs_)

    print(f"标定结果已保存到 {filepath}")

# ================== 主流程 ==================
if __name__ == "__main__":
    all_obj_points = []
    all_img_points = []
    image_size = None

    for idx, url in enumerate(RTSP_URLS):
        print(f"采集摄像头 {idx + 1} 的图像...")
        obj_points, img_points = capture_chessboard_images(url, CHECKERBOARD, MIN_VALID_IMAGES)

        if obj_points and img_points:
            all_obj_points.extend(obj_points)
            all_img_points.extend(img_points)
            if image_size is None:
                # 假设所有摄像头分辨率相同，取第一个有效帧大小
                ret, frame = cv2.VideoCapture(url).read()
                image_size = frame.shape[:2][::-1]

    if all_obj_points and all_img_points:
        print("开始标定摄像头...")
        ret, camera_matrix, dist_coeffs,rvecs_,tvecs_ = calibrate_camera(all_obj_points, all_img_points, image_size)
        if ret:
            save_calibration_results(OUTPUT_DIR, camera_matrix, dist_coeffs, rvecs_, tvecs_)
    else:
        print("未采集到足够的标定图像，无法执行标定")




#相机内参矩阵：
#相机内参矩阵：
#[[5.36740251e+03 0.00000000e+00 9.47058475e+02]
# [0.00000000e+00 4.00703241e+03 5.41325778e+02]
# [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
##畸变系数：
#[[    3.55427205 -1164.6404438      0.             0.
#     -4.17346434]]
#旋转向量 (rvec):
#[[-0.02600489]
# [ 0.75751227]
# [-0.01345969]]
#平移向量 (tvec):
#[[ -11.46874027]
# [  10.2919095 ]
# [1533.58921233]]
#标定结果已保存到 calibration_results/camera_calib_6.npz
#相机内参矩阵：
#相机内参矩阵：
#[[1.10851188e+03 0.00000000e+00 1.07578694e+03]
# [0.00000000e+00 1.12561408e+03 5.36692906e+02]
# [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
#畸变系数：
#[[-0.51278417 -0.51742545  0.          0.          3.74786421]]
#旋转向量 (rvec):
#[[-0.2120766 ]
# [ 0.02525703]
# [ 0.06893048]]
#平移向量 (tvec):
#[[ -6.9458661 ]
# [ 79.18618987]
# [437.87978921]]

#标定结果已保存到 calibration_results/camera_calib_5.npz



#相机内参矩阵：
#相机内参矩阵：
#相机内参矩阵：
#[[9.06119501e+02 0.00000000e+00 9.61691399e+02]
# [0.00000000e+00 1.09273915e+03 5.90185664e+02]
# [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
#畸变系数：
#[[  -1.18729241   19.74982105    0.            0.         -133.71558314]]
#旋转向量 (rvec):
#[[ 0.05848607]
# [-0.644885  ]
# [ 0.01493417]]
#平移向量 (tvec):
#[[-64.89042107]
# [-71.04057801]
# [344.28841856]]
#标定结果已保存到 calibration_results/camera_calib_4.npz


#相机内参矩阵：
#相机内参矩阵：
#[[4.68659370e+02 0.00000000e+00 1.05135468e+03]
# [0.00000000e+00 4.71325578e+02 5.57915894e+02]
# [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
#畸变系数：
#[[-0.2546635   0.57661026  0.          0.         -0.7603401 ]]
#旋转向量 (rvec):
#[[-0.06943149]
# [ 0.05546611]
# [-0.0156446 ]]
#平移向量 (tvec):
#[[-25.87796125]
# [ 45.08897431]
# [204.45491806]]
#标定结果已保存到 calibration_results/camera_calib_3.npz

#相机内参矩阵：
#相机内参矩阵：
#[[1.08503994e+03 0.00000000e+00 7.31829079e+02]
# [0.00000000e+00 1.09659257e+03 6.41575972e+02]
# [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
#畸变系数：
#[[ 0.09030105 -0.85380452  0.          0.          1.20505794]]
#旋转向量 (rvec):
#[[-0.01588351]
# [-0.20549196]
# [-0.00613856]]
#平移向量 (tvec):
#[[-114.33706765]
# [   7.80740157]
# [ 245.96431121]]
#标定结果已保存到 calibration_results/camera_calib_2.npz

#相机内参矩阵：
#相机内参矩阵：
#[[3.46372585e+03 0.00000000e+00 1.07141798e+03]
# [0.00000000e+00 3.23165370e+03 3.33628077e+02]
# [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
#畸变系数：
#[[  0.57213461 -11.5549736    0.           0.          57.43673517]]
#旋转向量 (rvec):
#[[-0.09780233]
# [-0.51972362]
# [ 0.01187643]]
#平移向量 (tvec):
#[[-209.63157779]
# [  70.67302414]
# [1047.27802925]]

#标定结果已保存到 calibration_results/camera_calib_1.npz