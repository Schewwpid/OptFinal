import numpy as np
from tqdm import tqdm


def read_points(file_path):
    points = []
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]  # 跳过第一行标题
        for line in lines:
            parts = line.strip().split()
            points.append((float(parts[0].strip(',')), float(parts[1].strip(',')), float(parts[2].strip(',')),
                           float(parts[3].strip(','))))
    return points


def compute_initial_matrix(points):
    A = []
    for (x1, y1, x2, y2) in points:
        A.append([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    M = Vt[-1].reshape(3, 3)
    return M


def lagrangian_function(params, points):
    M = params[:9].reshape(3, 3)
    lambda1, lambda2 = params[9], params[10]

    error = 0
    for (x1, y1, x2, y2) in points:
        pt1 = np.array([x1, y1, 1])
        pt2 = np.array([x2, y2, 1])
        error += np.abs(np.dot(pt2.T, np.dot(M, pt1))) ** 2

    R = enforce_so3(M)
    ortho_constraint = np.linalg.norm(np.dot(R.T, R) - np.eye(3)) ** 2
    det_constraint = (np.linalg.det(R) - 1) ** 2

    return error + lambda1 * ortho_constraint + lambda2 * det_constraint


def enforce_so3(M):
    U, _, Vt = np.linalg.svd(M)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = np.dot(U, Vt)
    return R


def gradient_descent(points, initial_params, learning_rate=1e-20, max_iters=10000, tolerance=1e-6):
    params = initial_params
    prev_error = float('inf')

    for i in tqdm(range(max_iters), desc="Gradient Descent Progress"):
        grad = np.zeros_like(params)
        current_error = lagrangian_function(params, points)

        for j in range(len(params)):
            params_temp = np.copy(params)
            params_temp[j] += tolerance
            grad[j] = (lagrangian_function(params_temp, points) - current_error) / tolerance

        params -= learning_rate * grad
        M_reshaped = params[:9].reshape(3, 3)
        M_reshaped = enforce_so3(M_reshaped)
        params[:9] = M_reshaped.flatten()

        if abs(prev_error - current_error) < tolerance:
            break

        prev_error = current_error

    return params[:9].reshape(3, 3)


def main():
    points = read_points('pointsMatches.txt')
    initial_M = compute_initial_matrix(points)
    initial_params = np.append(initial_M.flatten(), [1, 1])

    optimized_M = gradient_descent(points, initial_params)

    print("初始基础矩阵：\n", initial_M)
    print("优化后的基础矩阵：\n", optimized_M)
    print("最终对极约束误差：", lagrangian_function(np.append(optimized_M.flatten(), [1, 1]), points))


if __name__ == "__main__":
    main()
