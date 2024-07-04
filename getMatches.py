import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img1 = cv2.imread('data/temple/temple/temple0001.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/temple/temple/temple0029.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功读取
if img1 is None:
    print(f"Failed to read image: data/temple/temple/temple0001.png")
if img2 is None:
    print(f"Failed to read image: data/temple/temple/temple0002.png")

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

print('Matched points in image 1:')
print(pts1)

print('Matched points in image 2:')
print(pts2)

# 将匹配的点对保存到文件中
try:
    with open('pointsMatches1&29.txt', 'w') as f:
        f.write('Image 1 Points (x1, y1), Image 2 Points (x2, y2)\n')
        for pt1, pt2 in zip(pts1, pts2):
            f.write(f'{pt1[0]:.6f}, {pt1[1]:.6f}, {pt2[0]:.6f}, {pt2[1]:.6f}\n')
    print('Matched points saved to pointsMatches1&29.txt')
except IOError as e:
    print(f"Error writing to file: {e}")
