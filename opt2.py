import numpy as np


def read_points(file_path):
    points = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip the first line (header)
            parts = line.strip().replace(',', '').split()
            if len(parts) == 4:  # Ensure there are four parts
                points.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
    return points


def read_camera_params(file_path):
    with open(file_path, 'r') as file:
        parts = file.readline().strip().split()
        K = np.array([float(parts[i]) for i in range(1, 10)]).reshape(3, 3)
        R = np.array([float(parts[i]) for i in range(10, 19)]).reshape(3, 3)
        t = np.array([float(parts[i]) for i in range(19, 22)]).reshape(3)
    return K, R, t


def compute_initial_fundamental_matrix(pts1, pts2):
    A = []
    for (x1, y1, x2, y2) in zip(pts1[:, 0], pts1[:, 1], pts2[:, 0], pts2[:, 1]):
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), Vt))
    return F


def compute_epipolar_constraint_error(F, pts1, pts2):
    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    errors = []
    for pt1, pt2 in zip(pts1_h, pts2_h):
        error = np.abs(np.dot(pt2, np.dot(F, pt1)))
        errors.append(error)
    return np.mean(errors)


def gradient_descent(pts1, pts2, F, learning_rate=1e-20, max_iter=10000, tol=1e-6):
    for i in range(max_iter):
        grad = compute_gradient(pts1, pts2, F)
        F_new = F - learning_rate * grad
        error = compute_epipolar_constraint_error(F_new, pts1, pts2)
        if np.abs(compute_epipolar_constraint_error(F, pts1, pts2) - error) < tol:
            break
        F = F_new
        if i % 1000 == 0:
            print(f"Iteration {i}, Error: {error}")
    return F


def compute_gradient(pts1, pts2, F):
    grad = np.zeros(F.shape)
    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    for pt1, pt2 in zip(pts1_h, pts2_h):
        grad += np.outer(pt2, pt1) * np.dot(pt2, np.dot(F, pt1))
    return grad / pts1.shape[0]


def main():
    points = read_points('pointsMatches.txt')
    pts1 = np.array([(p[0], p[1]) for p in points])
    pts2 = np.array([(p[2], p[3]) for p in points])

    F_init = compute_initial_fundamental_matrix(pts1, pts2)
    print("初始基础矩阵：\n", F_init)

    F_optimized = gradient_descent(pts1, pts2, F_init)
    print("优化后的基础矩阵：\n", F_optimized)

    error = compute_epipolar_constraint_error(F_optimized, pts1, pts2)
    print("最终对极约束误差：", error)


if __name__ == "__main__":
    main()
