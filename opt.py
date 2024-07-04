import numpy as np
import cv2
import time

# 读取点对数据
def read_points(file_path):
    points1 = []
    points2 = []
    count = 0  # 初始化计数器
    with open(file_path, 'r') as file:
        next(file)  # 跳过文件的第一行
        for line in file:
            x1, y1, x2, y2 = map(float, line.strip().split(','))
            points1.append([x1, y1])
            points2.append([x2, y2])
            count += 1
    print(f"读取了 {count} 个点对")
    return np.array(points1), np.array(points2)

# 读取相机参数
def read_camera_params(file_path):
    camera_params = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 22:  # 确保每行有足够的数据
                img_name = parts[0]
                K = np.array([float(parts[i]) for i in range(1, 10)]).reshape(3, 3)
                R = np.array([float(parts[i]) for i in range(10, 19)]).reshape(3, 3)
                t = np.array([float(parts[i]) for i in range(19, 22)]).reshape(3)
                camera_params[img_name] = (K, R, t)
    return camera_params

# 读取角度信息
def read_angles(file_path):
    angles = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:  # 确保每行有足够的数据
                lat, lon, img_name = float(parts[0]), float(parts[1]), parts[2]
                angles[img_name] = (lat, lon)
    return angles

# 计算初始本质矩阵
def compute_initial_essential_matrix(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return E, R, t

# 构建Q9矩阵
def compute_Q9_matrix(pts1, pts2):
    N = len(pts1)
    Q9 = np.zeros((9, 9))
    for p1, p2 in zip(pts1, pts2):
        p1_h = np.append(p1, 1)
        p2_h = np.append(p2, 1)
        Q9 += np.outer(np.kron(p2_h, p1_h), np.kron(p2_h, p1_h))
    Q9 /= N
    return Q9

# 构建Q12矩阵
def compute_Q12_matrix(Q9):
    Q12 = np.zeros((12, 12))
    Q12[:9, :9] = Q9
    Q12[9:, 9:] = np.eye(3)
    return Q12

# 定义代数误差函数
def algebraic_error(params, Q12):
    return np.dot(params.T, np.dot(Q12, params))

# 计算梯度
def compute_gradient(params, Q12):
    return 2 * np.dot(Q12, params)

# 梯度下降优化
def gradient_descent_optimizer(objective_func, gradient_func, params_init, Q12, learning_rate=1e-20, max_iters=100000, tol=1e-10):
    params = params_init
    for i in range(max_iters):
        grad = gradient_func(params, Q12)
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            print(f"第 {i} 次迭代产生了 NaN 或 Inf 值，终止优化")
            break
        new_params = params - learning_rate * grad
        if np.any(np.isnan(new_params)) or np.any(np.isinf(new_params)):
            print(f"第 {i} 次迭代产生了 NaN 或 Inf 参数，终止优化")
            break
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params
    return params

# 计算重投影误差
def compute_reprojection_error(pts1, pts2, E, K):
    pts1_h = cv2.convertPointsToHomogeneous(pts1)[:, 0, :]
    pts2_h = cv2.convertPointsToHomogeneous(pts2)[:, 0, :]
    F = np.dot(np.linalg.inv(K).T, np.dot(E, np.linalg.inv(K)))
    lines1 = np.dot(F.T, pts2_h.T).T
    lines2 = np.dot(F, pts1_h.T).T
    error1 = np.mean(np.abs(np.sum(lines1 * pts1_h, axis=1)) / np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2))
    error2 = np.mean(np.abs(np.sum(lines2 * pts2_h, axis=1)) / np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2))
    return (error1 + error2) / 2

# 主函数
def main():
    # 读取点对数据
    pts1, pts2 = read_points('pointsMatches.txt')

    # 读取相机参数
    camera_params = read_camera_params('data/temple/temple/temple_par.txt')
    K = camera_params[list(camera_params.keys())[0]][0]  # 使用第一个相机的内参矩阵

    # 读取角度信息
    angles = read_angles('data/temple/temple/temple_ang.txt')

    # 计算初始本质矩阵
    E, R, t = compute_initial_essential_matrix(pts1, pts2, K)
    print("初始本质矩阵：\n", E)

    # 构建Q9和Q12矩阵
    Q9 = compute_Q9_matrix(pts1, pts2)
    Q12 = compute_Q12_matrix(Q9)
    print("Q9矩阵：\n", Q9)
    print("Q12矩阵：\n", Q12)

    # 初始参数
    params_init = np.hstack((E.ravel(), t.ravel()))

    # 记录开始时间
    start_time = time.time()

    # 使用梯度下降优化
    optimized_params = gradient_descent_optimizer(algebraic_error, compute_gradient, params_init, Q12)

    # 记录结束时间
    end_time = time.time()
    runtime = end_time - start_time

    # 提取优化后的本质矩阵
    E_optimized = optimized_params[:9].reshape(3, 3)
    t_optimized = optimized_params[9:]
    print("优化后的本质矩阵：\n", E_optimized)
    print("优化后的平移向量：\n", t_optimized)

    # 计算代数误差
    mean_algebraic_error = algebraic_error(optimized_params, Q12)
    print("平均代数误差：", mean_algebraic_error)
    print("运行时间：", runtime, "秒")

    # 计算重投影误差
    reprojection_error = compute_reprojection_error(pts1, pts2, E_optimized, K)
    print("重投影误差：", reprojection_error)

if __name__ == '__main__':
    main()
