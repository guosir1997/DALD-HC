# 这是一个示例 Python 脚本。
import copy
import logging
# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

from collections import Counter
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("./datasets/processed_dataset_time_2_cos&sin_binary_low&medium.csv")  # 二分类

# 特征列（除了目标列 'Efficiency_Status'）
X = df.drop(columns=['Efficiency_Status'])

# 目标列（多类别分类：High, Medium, Low，已编码为 0, 1, 2）
y = df['Efficiency_Status']

# 假设 df 是你的原始数据框
X = df.drop('Efficiency_Status', axis=1)

# 🔑 先保存列名
feature_names = X.columns

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_train = y_train.values
y_test = y_test.values

# 标准化特征数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  使用 feature_names 构造 DataFrame
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

X_train = np.c_[np.ones(X_train.shape[0]), X_train]  # 给 X_train 添加偏置项
X_test = np.c_[np.ones(X_test.shape[0]), X_test]  # 给 X_test 添加偏置项


# def dirichlet_partition(y_array, num_clients, alpha=0.5, seed=42):
#     """
#     基于 Dirichlet 分布将标签 y_array 分配到 num_clients 个客户端中。
#     """
#     np.random.seed(seed)
#     class_labels = np.unique(y_array)
#     idx_by_class = {label: np.where(y_array == label)[0] for label in class_labels}
#     client_indices = [[] for _ in range(num_clients)]
#
#     for label in class_labels:
#         indices = idx_by_class[label]
#         np.random.shuffle(indices)
#         proportions = np.random.dirichlet([alpha] * num_clients)
#         proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
#         splits = np.split(indices, proportions)
#         for i, split in enumerate(splits):
#             client_indices[i].extend(split)
#
#     return client_indices


def dirichlet_partition(y_array, num_clients, alpha=0.5, seed=42, min_samples_per_client=1):
    """
    Dirichlet 非IID划分，并保证每个客户端最少 min_samples_per_client 条数据。
    """
    np.random.seed(seed)
    class_labels = np.unique(y_array)
    idx_by_class = {label: np.where(y_array == label)[0] for label in class_labels}

    success = False
    retry = 0
    while not success:
        retry += 1
        client_indices = [[] for _ in range(num_clients)]

        for label in class_labels:
            indices = idx_by_class[label]
            np.random.shuffle(indices)
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            splits = np.split(indices, proportions)
            for i, split in enumerate(splits):
                client_indices[i].extend(split)

        # 检查是否所有客户端都满足最小样本数要求
        client_sizes = [len(c) for c in client_indices]
        if min(client_sizes) >= min_samples_per_client:
            success = True
        elif retry > 100:
            raise RuntimeError("无法满足所有客户端最小样本要求，请尝试调大 alpha 或减少 client 数量。")

    return client_indices


def split_data(X_array, y_array, num_fog, num_edge, alpha=0.5, seed=42):
    """
    将数据划分成 m = num_fog * num_edge 台设备，再划分到 n 个工厂中。

    参数：
        X_array: np.ndarray，特征数据，第10列为 Machine_ID（可以忽略）
        y_array: np.ndarray，标签数据
        num_fog: 工厂数量（n）
        num_edge: 每个工厂的设备数（m_i）
        alpha: Dirichlet 分布参数，越小越 Non-IID
        seed: 随机种子，确保可重复性

    返回：
        client_data_[(i, j)] = {'a': X_sub, 'b': y_sub}
    """
    assert X_array.shape[0] == y_array.shape[0], "X 和 y 的样本数必须一致"

    total_clients = num_fog * num_edge
    client_indices = dirichlet_partition(y_array, total_clients, alpha, seed)

    client_data_ = dict()
    for cid, indices in enumerate(client_indices):
        fog_id = cid // num_edge
        edge_id = cid % num_edge
        X_sub = X_array[indices]
        y_sub = y_array[indices]
        client_data_[(fog_id, edge_id)] = {'a': X_sub, 'b': y_sub}

    return client_data_


def print_device_label_distribution(client_data):
    print(f"{'工厂':<5} {'设备':<5} | {'标签分布':<30} | {'样本总数':<10} | {'正类比例':<10}")
    print("-" * 80)
    for (i, j), data in sorted(client_data.items()):
        labels = data['b']
        counter = Counter(labels)
        total = len(labels)
        pos = counter[1] if 1 in counter else 0
        neg = counter[-1] if -1 in counter else 0
        pos_ratio = pos / total if total > 0 else 0
        print(f"{i:<5} {j:<5} | {-1}: {neg:<5}, 1: {pos:<5}       | {total:<10} | {pos_ratio:.2%}")


#
# def logistic_grad_solver_edge_i_j(x, y, i, j, mu, d, alpha, rho):
#     # 计算所有样本的分数 b_ij * (a_ij^T y_ij)
#     scores = client_data[i, j]['b'] * np.dot(client_data[i, j]['a'], y[i][j])
#
#     # 计算1/d * f_i 梯度：梯度 = sum( (-b_ij * X_i) / (1 + exp(b_ij * a_ij^T y_ij)) ) / d
#     grad_coeff = -client_data[i, j]['b'] * np.exp(-np.logaddexp(0, scores))  # shape: (n_samples,)
#     grad_1 = np.dot(client_data[i, j]['a'].T, grad_coeff) / d  # 归一化梯度
#
#     # 计算 松弛项梯度
#     # -μ_ij^k-ρ(x_i^(k,v-1)-y_ij)
#     grad_2 = - mu[i][j] - rho * (x[i] - y[i][j])
#     grad = grad_1 + grad_2
#
#     y_i_j_temp = y[i][j] - alpha * grad
#
#     dual_residual_i_j = y_i_j_temp - y[i][j]
#
#     y[i][j] = y_i_j_temp
#
#     # 计算局部增广拉格朗日函数
#     # 计算1 / d * f_i:
#     # 数值稳定地计算 log(1 + exp(-scores))
#     losses = np.logaddexp(0, -scores)  # 等价于 log(1 + exp(-scores))
#     average_loss = np.sum(losses) / d
#
#     # 添加增广增广松弛项
#     A_ij = np.dot(mu[i][j], (x[i] - y[i][j])) + rho/2 * np.sum((x[i] - y[i][j]) ** 2)
#
#     F_i = average_loss + A_ij
#
#     return F_i, average_loss, dual_residual_i_j

def logistic_grad_solver_edge_i_j(x, y, i, j, mu, d, alpha, rho):
    updated_y = copy.deepcopy(y[i][j])

    for e in range(E - 1):
        # Step 1: 计算逻辑回归分数：b * (a^T y)
        scores = client_data[i, j]['b'] * np.dot(client_data[i, j]['a'], updated_y)

        # Step 2: 计算逻辑损失梯度项
        grad_logit = -client_data[i, j]['b'] * np.exp(-np.logaddexp(0, scores))  # shape: (n_samples,)
        grad_f = np.dot(client_data[i, j]['a'].T, grad_logit) / d  # 标准化后梯度

        # Step 3: 增广项梯度
        grad_aug = - mu[i][j] - rho * (x[i] - updated_y)

        # Step 4: 合并梯度并更新 y
        grad_total = grad_f + grad_aug

        # 梯度下降
        updated_y = updated_y - alpha * grad_total

    dual_residual = updated_y - y[i][j]

    y[i][j] = updated_y

    # Step 5: 计算增广拉格朗日目标值
    losses = np.logaddexp(0, -scores)
    loss_avg = np.sum(losses) / d
    lag_term = np.dot(mu[i][j], (x[i] - y[i][j])) + rho / 2 * np.sum((x[i] - y[i][j]) ** 2)

    F_total = loss_avg + lag_term

    return F_total, loss_avg, dual_residual


def logistic_third_part_solver_edge_i_j(x, y, i, j, mu, d, alpha, rho):
    # 从数据集中取出 a 和 b
    a = client_data[i, j]['a']  # shape: (d, dim)
    b = client_data[i, j]['b']  # shape: (d,)，±1标签

    scores = client_data[i, j]['b'] * np.dot(client_data[i, j]['a'], y[i][j])

    # 当前的 y_ij 向量
    y_ij_init = copy.deepcopy(y[i][j])

    # 定义目标函数
    def objective(y_ij):
        # 第一个项：逻辑损失
        logits = b * np.dot(a, y_ij)
        log_loss = np.mean(np.logaddexp(0, -logits))

        # 第二项：线性乘子项
        linear_term = np.dot(mu[i][j], x[i] - y_ij)

        # 第三项：二次项
        quad_term = (rho / 2) * np.linalg.norm(x[i] - y_ij) ** 2

        return log_loss + linear_term + quad_term

    # 可选：定义梯度（加速收敛）
    def gradient(y_ij):
        logits = b * np.dot(a, y_ij)
        sigma = -b * np.exp(-np.logaddexp(0, logits))  # 逻辑回归梯度项
        grad_log_loss = np.dot(a.T, sigma) / d

        grad_linear = -mu[i][j]
        grad_quad = -rho * (x[i] - y_ij)

        return grad_log_loss + grad_linear + grad_quad

    # 调用最优化器
    result = minimize(
        objective,
        y_ij_init,
        jac=gradient,  # 提供梯度更高效
        method='L-BFGS-B'
    )

    # 更新 y 和 dual_residual
    updated_y = result.x
    dual_residual = updated_y - y[i][j]
    y[i][j] = copy.deepcopy(updated_y)

    # Step 5: 计算增广拉格朗日目标值
    losses = np.logaddexp(0, -scores)
    loss_avg = np.sum(losses) / d
    lag_term = np.dot(mu[i][j], (x[i] - y[i][j])) + rho / 2 * np.sum((x[i] - y[i][j]) ** 2)

    F_total = loss_avg + lag_term

    return F_total, loss_avg, dual_residual


def analytical_solution_2_fog_i(w, x, y, i, mu_0, mu, rho):
    x_i_temp = 1 / (m_i + 1) * (w + sum(y[i]) + 1 / rho * (mu_0[i] - sum(mu[i])))

    dual_residual_i = x_i_temp - x[i]

    x[i] = copy.deepcopy(x_i_temp)

    # 计算局部增广拉格朗日函数
    A_i = np.dot(mu_0[i], (w - x[i])) + rho / 2 * np.sum((w - x[i]) ** 2)

    for j in range(m_i):
        A_i += np.dot(mu[i][j], (x[i] - y[i][j])) + rho / 2 * np.sum((x[i] - y[i][j]) ** 2)

    return A_i, dual_residual_i


def analytical_solution_2_cloud_center():
    w_temp = 1 / n * (sum(x) - 1 / rho * (sum(mu_0)))

    dual_residual_i = w_temp - w

    w[:] = copy.deepcopy(w_temp)  # 改变外部 w 的内容	修改的是原数组对象的内容

    # 计算局部增广拉格朗日函数
    A_0 = 0

    for i in range(n):
        A_0 += np.dot(mu_0[i], (w - x[i])) + rho / 2 * np.sum((w - x[i]) ** 2)

    return A_0, dual_residual_i


def predict(X, w):
    scores = np.dot(X, w)
    return np.sign(scores)  # 1 或 -1


if __name__ == '__main__':
    np.random.seed(seed=42)

    # m = X_train_df['Machine_ID'].unique().shape[0]

    # 总设备数
    m = 10

    # 工厂数
    n = 10

    # 单个工厂设备数
    m_i = int(m / n)

    # 总样本数
    d = X_train.shape[0]

    # 总特征数
    p = X_train.shape[1]

    # 划分数据
    client_data = split_data(X_train, y_train, num_fog=n, num_edge=m_i, alpha=1.0)
    #
    print_device_label_distribution(client_data)

    # client_data = split_data(X_train, y_train, num_fog=n, num_edge=m_i, alpha=0.5)
    #
    # print_device_label_distribution(client_data)

    total_samples = 0
    for (fog_key, edge_key), data in client_data.items():
        X_sub = data['a']
        y_sub = data['b']
        n_samples = X_sub.shape[0]
        print(f"工厂 {fog_key} 设备 {edge_key} 样本数: {n_samples}")
        total_samples += n_samples

    print(f"\n所有设备总样本数: {total_samples}")

    # 初始化参数
    w = np.zeros(p)  # w
    x = [np.zeros(p) for i in range(n)]  # x1, x2, x3, x4
    y = [[np.zeros(p) for j in range(m_i)] for i in range(n)]  # y_ij

    mu_0 = [np.zeros(p) for i in range(n)]

    mu = [[np.zeros(p) for j in range(m_i)] for i in range(n)]

    dual_residual = {}
    pri_residual = {}

    F_values = []
    avg_loss = []
    avg_accuracies_history = []
    std_history = []

    eps_pri = 1e-5
    eps_dual = 1e-5

    v_max = 1

    rho = 1.0

    alpha = 0.001  # 梯度下降步长

    E = 10

    max_iter = 301

    N_iter = 0

    logger = logging.getLogger('log')

    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(
        f'./log/z_logger_HC_E_{E}_rho_{rho}_m_{m}_n_{n}_m_i_{m_i}_v_max_{v_max}_eps_pri_{eps_pri}_eps_dual_{eps_dual}_alpha_{alpha}.log',
        'w')

    ch = logging.StreamHandler()

    formatter_1 = logging.Formatter(f'%(message)s')
    formatter_2 = logging.Formatter(f'%(message)s')

    fh.setFormatter(formatter_1)
    fh.setLevel(logging.DEBUG)

    ch.setFormatter(formatter_2)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)

    for k in range(2001):  # 可以根据需要调整迭代次数
        v = 0
        while True and (not (v == v_max)):
            N_iter += 1
            v += 1
            F = []
            loss = []

            for i in range(n):
                for j in range(m_i):
                    F_i_j, avg_loss_i, dual_residual[(i, j)] = logistic_grad_solver_edge_i_j(x, y, i, j, mu, d, alpha,
                                                                                             rho)
                    # F_i_j, avg_loss_i, dual_residual[(i, j)] = logistic_third_part_solver_edge_i_j(x, y, i, j, mu, d, alpha, rho)
                    F.append(F_i_j)
                    loss.append(avg_loss_i)

            for i in range(n):
                F_i, dual_residual[i] = analytical_solution_2_fog_i(w, x, y, i, mu_0, mu, rho)
                F.append(F_i)

            F_0, dual_residual[-1] = analytical_solution_2_cloud_center()
            F.append(F_0)

            # print(F_0,dual_residual[-1].shape)
            # print(w)

            F_values.append(sum(F))
            avg_loss.append(sum(loss))

            acc = []
            for i in range(n):
                for j in range(m_i):
                    test_pred = predict(X_test, y[i][j])
                    test_accuracy = np.mean(test_pred == y_test)
                    acc.append(test_accuracy)
            # for i in range(n):
            #     test_pred = predict(X_test, x[i])
            #     test_accuracy = np.mean(test_pred == y_test)
            #     acc.append(test_accuracy)
            #
            # test_pred = predict(X_test, w)
            # test_accuracy = np.mean(test_pred == y_test)
            # acc.append(test_accuracy)

            acc_array = np.array(acc)
            mean_acc = np.mean(acc_array)
            std_acc = np.std(acc_array)
            avg_accuracies_history.append(mean_acc)
            std_history.append(std_acc)

            logger.info(
                f"第{k + 1: >2}-{v: >2}({N_iter: >4})次迭代: F_Value: {F_values[-1]},  avg_loss: {avg_loss[-1]}, mean_acc: {mean_acc}, std_acc: {std_acc}")

            if N_iter == max_iter:
                break

            temp_1 = 0
            for key in dual_residual.keys():
                # print(np.max(abs(dual_residual[key])))
                if (abs(dual_residual[key]) <= np.ones(dual_residual[key].shape) * eps_dual).all():
                    temp_1 += 1
            # logger.info(f'temp_1 = {temp_1}')
            if temp_1 == len(dual_residual):
                logger.info(f'break: temp_1 = {temp_1}')
                break

        for i in range(n):
            for j in range(m_i):
                pri_residual[i, j] = x[i] - y[i][j]
                mu[i][j] = mu[i][j] + rho * pri_residual[i, j]

        for i in range(n):
            pri_residual[i] = w - x[i]
            mu_0[i] = mu_0[i] + rho * pri_residual[i]

        if N_iter == max_iter:
            break

        temp_2 = 0
        for key in pri_residual.keys():
            if (abs(pri_residual[key]) <= np.ones(pri_residual[key].shape) * eps_pri).all():
                temp_2 += 1
        # logger.info(f'temp_2 = {temp_2}')

        if temp_1 + temp_2 == len(dual_residual) + len(pri_residual):
            logger.info(f'break: temp_1 + temp_2 = {temp_1 + temp_2}')
            break
    # 计算归一化的损失变化
    normalized_loss = abs((np.array(avg_loss) - avg_loss[-1])) / avg_loss[-1]

    # 创建 DataFrame
    df = pd.DataFrame({
        "F_values": F_values,
        "Avg_Loss": avg_loss,
        "Normalized_Loss": normalized_loss,
        "avg_acc_history": avg_accuracies_history,
        "std_history": std_history
    })

    # 保存为 csv 文件
    df.to_csv(
        f'./csv/data_HC_E_{E}_rho_{rho}_m_{m}_n_{n}_m_i_{m_i}_v_max_{v_max}_iter_{N_iter}_eps_pri_{eps_pri}_eps_dual_{eps_dual}_alpha_{alpha}.csv',
        index=False)

    logger.info(f'n: {n}')
    logger.info(f'm_i: {m_i}')
    logger.info(f'E: {E}')
    logger.info(f'alpha: {alpha}')
    logger.info(f'N_iter: {N_iter}')
    logger.info(f'v_max: {v_max}')
    logger.info(f'所有客户端的平均准确率: {mean_acc:.8f}')
    logger.info(f'所有客户端的准确率标准差: {std_acc:.8f}')
    logger.info(f'w: {w}')

    # for i in range(n):
    #     for j in range(m_i):
    #         test_pred = predict(X_test, y[i][j])
    #         test_accuracy = np.mean(test_pred == y_test)
    #         logger.info(f'设备{i}-{j}准确率: {test_accuracy}')
    #
    # for i in range(n):
    #     test_pred = predict(X_test, x[i])
    #     test_accuracy = np.mean(test_pred == y_test)
    #     logger.info(f'工厂{i}准确率: {test_accuracy}')
    #
    # test_pred = predict(X_test, w)
    # test_accuracy = np.mean(test_pred == y_test)
    # logger.info(f'cloud center 准确率: {test_accuracy}')

    a = 1
