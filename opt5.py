import numpy as np
import cvxpy as cp

# 读取点对应文件
def read_points(file_path):
    points = []
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]  # 跳过第一行标题
        for line in lines:
            parts = line.strip().split()
            points.append((float(parts[0].strip(',')), float(parts[1].strip(',')), float(parts[2].strip(',')), float(parts[3].strip(','))))
    return points

# 示例文件路径
file_path = 'pointsMatches.txt'

# 读取点对应
points_matches = read_points(file_path)

# 缩放数据
def normalize_points(points):
    max_val = max(max(p) for p in points)
    return [(x1 / max_val, y1 / max_val, x2 / max_val, y2 / max_val) for (x1, y1, x2, y2) in points]

points_matches = normalize_points(points_matches)

# 计算 Kronecker 积
def kronecker_product(v1, v2):
    return np.kron(v1, v2)

# 计算 Q9 矩阵
def compute_Q9(points_matches):
    N = len(points_matches)
    Q9 = np.zeros((9, 9))

    for (x1, y1, x2, y2) in points_matches:
        f1_vec = np.array([x1, y1, 1])
        f2_vec = np.array([x2, y2, 1])
        kronecker_prod = kronecker_product(f2_vec, f1_vec)
        Q9 += np.outer(kronecker_prod, kronecker_prod)

    Q9 /= N
    return Q9

# 计算 Q9
Q9 = compute_Q9(points_matches)
print("Q9 matrix:\n", Q9)

# 构建 Q12 矩阵
Q12 = np.zeros((12, 12))
Q12[:9, :9] = Q9
Q12[9:, 9:] = np.eye(3)

print("Q12 matrix:\n", Q12)

# 定义 A_i 矩阵和 c_i 常数
A0 = np.zeros((12, 12))
A0[9, 9] = 1
A0[10, 10] = 1
A0[11, 11] = 1

A1 = np.zeros((12, 12))
A1[0, 0] = 1
A1[1, 1] = 1
A1[2, 2] = 1
A1[10, 10] = -1
A1[11, 11] = -1

A2 = np.zeros((12, 12))
A2[3, 3] = 1
A2[4, 4] = 1
A2[5, 5] = 1
A2[9, 9] = -1
A2[11, 11] = -1

A3 = np.zeros((12, 12))
A3[6, 6] = 1
A3[7, 7] = 1
A3[8, 8] = 1
A3[9, 9] = -1
A3[10, 10] = -1

A4 = np.zeros((12, 12))
A4[3, 6] = 0.5
A4[6, 3] = 0.5
A4[4, 7] = 0.5
A4[7, 4] = 0.5
A4[5, 8] = 0.5
A4[8, 5] = 0.5
A4[11, 10] = 0.5
A4[10, 11] = 0.5

A5 = np.zeros((12, 12))
A5[0, 6] = 0.5
A5[6, 0] = 0.5
A5[1, 7] = 0.5
A5[7, 1] = 0.5
A5[2, 8] = 0.5
A5[8, 2] = 0.5
A5[11, 9] = 0.5
A5[9, 11] = 0.5

A6 = np.zeros((12, 12))
A6[0, 3] = 0.5
A6[3, 0] = 0.5
A6[1, 4] = 0.5
A6[4, 1] = 0.5
A6[2, 5] = 0.5
A6[5, 2] = 0.5
A6[10, 9] = 0.5
A6[9, 10] = 0.5

A = [A0, A1, A2, A3, A4, A5, A6]
c = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 打印 A0 到 A6 矩阵
'''
for i, Ai in enumerate(A):
    print(f"A{i} matrix:\n", Ai)
'''
# 定义变量
X = cp.Variable((12, 12), symmetric=True)

# 目标函数
objective = cp.Minimize(cp.trace(Q12 @ X))

# 约束条件
constraints = [cp.trace(A[i] @ X) == c[i] for i in range(7)]
constraints += [X >> 0]  # X 是正半定的

# 检查约束条件类型
for constraint in constraints:
    print("Constraint:", constraint)

# 定义并求解问题
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.MOSEK, verbose=True)

# 输出结果
print("Optimal value:", problem.value)
print("Optimal matrix X:")
print(X.value)

# 提取x向量
eigenvalues, eigenvectors = np.linalg.eigh(X.value)
x = eigenvectors[:, -1]

print("Extracted vector x:")
print(x)

# 提取本质矩阵和平移向量
E = x[:9].reshape(3, 3)
t = x[9:]

print("Essential matrix E:")
print(E)
print("Translation vector t:")
print(t)

# 计算 \sum f'_i^T E f_i
sum_residuals = 0
for (x1, y1, x2, y2) in points_matches:
    f1_vec = np.array([x1, y1, 1])
    f2_vec = np.array([x2, y2, 1])
    residual = np.dot(f2_vec, np.dot(E, f1_vec))
    sum_residuals += residual

print("Sum of residuals ∑ f'_i^T E f_i:", sum_residuals)
