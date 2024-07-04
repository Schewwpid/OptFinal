import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img1 = cv2.imread('data/temple/temple/temple0001.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/temple/temple/temple0002.png', cv2.IMREAD_GRAYSCALE)

# 读取相机参数
def read_camera_parameters(filename):
    params = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) < 22:
                print(f"Skipping malformed line: {line}")
                continue
            img_name = parts[0]
            K = np.array(parts[1:10], dtype=float).reshape(3, 3)
            R = np.array(parts[10:19], dtype=float).reshape(3, 3)
            t = np.array(parts[19:22], dtype=float).reshape(3, 1)
            params[img_name] = (K, R, t)
    return params

camera_params = read_camera_parameters('data/temple/temple/temple_par.txt')

# 获取投影矩阵
def get_projection_matrix(K, R, t):
    Rt = np.hstack((R, t))
    P = np.dot(K, Rt)
    return P

# 获取图像对应的相机参数
K1, R1, t1 = camera_params['temple0001.png']
K2, R2, t2 = camera_params['temple0002.png']
P1 = get_projection_matrix(K1, R1, t1)
P2 = get_projection_matrix(K2, R2, t2)

# 创建 SIFT 特征提取器
sift = cv2.SIFT_create()

# 提取关键点和描述子
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 使用 BFMatcher 进行特征匹配
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# 将匹配结果按距离排序
matches = sorted(matches, key=lambda x: x.distance)

# 绘制前10个匹配点
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 使用 matplotlib 显示匹配结果
plt.figure(figsize=(10, 5))
plt.imshow(img_matches)
plt.title('Feature Matches')
plt.axis('off')
plt.show()

# 将关键点转换为 NumPy 数组
pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
pts2 = np.array([kp2[m.trainIdx].pt for m in matches])

# 使用 cv2.triangulatePoints 进行三角化
pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

# 将齐次坐标转换为三维坐标
pts3D = pts4D[:3] / pts4D[3]

print('3D Points:', pts3D.T)
