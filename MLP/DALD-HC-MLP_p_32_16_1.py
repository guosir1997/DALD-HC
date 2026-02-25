# -*- coding: UTF-8 -*-
"""
@Project ：代码_new 
@File    ：DALD-HC-MLP_p_32_16_1.py
@Author  ：Wenyou Guo
@Date    ：2026/2/24 12:31 
"""

import copy
import logging
from collections import deque
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlp_utils import (forward, backward, edge_param_update,
                       compute_augmented_lagrangian, sigmoid)

df = pd.read_csv("./datasets/processed_dataset_time_2_cos&sin_binary_low&medium.csv")  # 二分类

# 特征列（除了目标列 'Efficiency_Status'）
X = df.drop(columns=['Efficiency_Status'])

# 目标列（多类别分类：High, Medium, Low，已编码为 0, 1, 2）
y = df['Efficiency_Status']

# 假设 df 是你的原始数据框
X = df.drop('Efficiency_Status', axis=1)

# 先保存列名
feature_names = X.columns

y = ((y + 1) / 2).astype(int)

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


# ══════════════════════════════════════════════
# 边缘节点求解器（MLP 版）
# ══════════════════════════════════════════════
def mlp_grad_solver_edge_i_j(client_data, x, e_fog,
                             y, e_edge,
                             i, j,
                             mu, lamb,
                             d, alpha, rho, L, E):
    """
    在边缘节点 (i,j) 上运行 E 步梯度下降，更新 y[i][j] 与 e_edge[i][j]。

    参数
    ----
    client_data  : 全局数据字典
    x            : 雾节点权重参数，x[i][l].shape = (M_{l+1}, M_l)
    e_fog        : 雾节点偏置参数，e_fog[i][l].shape = (M_{l+1},)
    y            : 边缘节点权重参数（将被原地更新）
    e_edge       : 边缘节点偏置参数（将被原地更新）
    mu           : 拉格朗日乘子（权重），mu[i][j][l].shape = (M_{l+1}, M_l)
    lamb         : 拉格朗日乘子（偏置），lamb[i][j][l].shape = (M_{l+1},)
    d            : 全局总样本数（归一化用）
    alpha        : 梯度下降步长
    rho          : 惩罚系数
    L            : 网络层数
    E            : 本地迭代次数

    返回
    ----
    F_total      : 本次迭代后的增广拉格朗日值
    loss_avg     : 归一化损失
    dual_residual: list[np.ndarray]（权重残差）+ list[np.ndarray]（偏置残差）
                   拼接为一个 list，用于收敛判断
    """
    X_batch = client_data[(i, j)]['a']  # (n_samples, M_0)
    b_batch = client_data[(i, j)]['b']  # (n_samples,)

    # 深拷贝当前参数，避免影响其他流程
    cur_y_w = copy.deepcopy(y[i][j])  # list[ndarray]，长度 L
    cur_y_b = copy.deepcopy(e_edge[i][j])  # list[ndarray]，长度 L

    # 固定雾节点参数（本轮视为常数）
    x_w_fixed = x[i]  # list[ndarray]
    x_b_fixed = e_fog[i]  # list[ndarray]

    mu_w = mu[i][j]  # list[ndarray]
    mu_b = lamb[i][j]  # list[ndarray]

    for _ in range(E):
        # 反向传播：计算 ∂f_ij/∂y_w，∂f_ij/∂y_b
        grad_w, grad_b = backward(X_batch, b_batch,
                                  cur_y_w, cur_y_b,
                                  L, d)

        # 参数梯度下降（含增广项）
        cur_y_w, cur_y_b = edge_param_update(
            cur_y_w, cur_y_b,
            grad_w, grad_b,
            x_w_fixed, x_b_fixed,
            mu_w, mu_b,
            alpha, rho, L
        )

    # 对偶残差（更新前后之差）
    # 权重残差列表 + 偏置残差列表
    dual_residual_w = [cur_y_w[l] - y[i][j][l] for l in range(L)]
    dual_residual_b = [cur_y_b[l] - e_edge[i][j][l] for l in range(L)]

    # 写回
    y[i][j] = cur_y_w
    e_edge[i][j] = cur_y_b

    # 增广拉格朗日值
    F_total, loss_avg = compute_augmented_lagrangian(
        X_batch, b_batch,
        cur_y_w, cur_y_b,
        x_w_fixed, x_b_fixed,
        mu_w, mu_b,
        rho, L, d
    )

    return F_total, loss_avg, dual_residual_w, dual_residual_b


# ══════════════════════════════════════════════
# 雾节点闭式解（MLP 版，按层独立）
# ══════════════════════════════════════════════
def analytical_solution_fog_i(w, e_cloud,
                              x, e_fog,
                              y, e_edge,
                              i, m_i,
                              mu_0, mu,
                              lamb_0, lamb,
                              rho, L):
    """
    对雾节点 i 的每层参数求闭式解（文档公式）：

      x_i^l = 1/(m_i+1) [ w^l + Σ_j y_ij^l + 1/ρ (μ_0i^l - Σ_j μ_ij^l) ]
      e_i^l = 1/(m_i+1) [ e^l  + Σ_j e_ij^l + 1/ρ (λ_0i^l - Σ_j λ_ij^l) ]

    返回 (A_i, dual_residual_w_list, dual_residual_b_list)
    """
    dual_res_w = []
    dual_res_b = []

    for l in range(L):
        # ── 权重 ──
        sum_y = sum(y[i][j][l] for j in range(m_i))
        sum_mu = sum(mu[i][j][l] for j in range(m_i))
        x_new = (1.0 / (m_i + 1)) * (w[l] + sum_y
                                     + (1.0 / rho) * (mu_0[i][l] - sum_mu))
        dual_res_w.append(x_new - x[i][l])
        x[i][l] = x_new

        # ── 偏置 ──
        sum_eb = sum(e_edge[i][j][l] for j in range(m_i))
        sum_lb = sum(lamb[i][j][l] for j in range(m_i))
        eb_new = (1.0 / (m_i + 1)) * (e_cloud[l] + sum_eb
                                      + (1.0 / rho) * (lamb_0[i][l] - sum_lb))
        dual_res_b.append(eb_new - e_fog[i][l])
        e_fog[i][l] = eb_new

    # 计算增广拉格朗日惩罚项（标量，仅用于监控）
    A_i = 0.0
    for l in range(L):
        diff_w0 = w[l] - x[i][l]
        A_i += (np.sum(mu_0[i][l] * diff_w0)
                + 0.5 * rho * np.sum(diff_w0 ** 2))
        diff_b0 = e_cloud[l] - e_fog[i][l]
        A_i += (np.dot(lamb_0[i][l], diff_b0)
                + 0.5 * rho * np.sum(diff_b0 ** 2))
        for j in range(m_i):
            diff_wj = x[i][l] - y[i][j][l]
            A_i += (np.sum(mu[i][j][l] * diff_wj)
                    + 0.5 * rho * np.sum(diff_wj ** 2))
            diff_bj = e_fog[i][l] - e_edge[i][j][l]
            A_i += (np.dot(lamb[i][j][l], diff_bj)
                    + 0.5 * rho * np.sum(diff_bj ** 2))

    return A_i, dual_res_w, dual_res_b


# ══════════════════════════════════════════════
# 云节点闭式解（MLP 版，按层独立）
# ══════════════════════════════════════════════
def analytical_solution_cloud(w, e_cloud,
                              x, e_fog,
                              mu_0, lamb_0,
                              n, rho, L):
    """
      w^l     = 1/n [ Σ_i x_i^l - 1/ρ Σ_i μ_0i^l ]
      e^l     = 1/n [ Σ_i e_i^l  - 1/ρ Σ_i λ_0i^l ]

    返回 (A_0, dual_residual_w_list, dual_residual_b_list)
    """
    dual_res_w = []
    dual_res_b = []

    for l in range(L):
        # ── 权重 ──
        sum_x = sum(x[i][l] for i in range(n))
        sum_mu0 = sum(mu_0[i][l] for i in range(n))
        w_new = (1.0 / n) * (sum_x - (1.0 / rho) * sum_mu0)
        dual_res_w.append(w_new - w[l])
        w[l] = w_new

        # ── 偏置 ──
        sum_ef = sum(e_fog[i][l] for i in range(n))
        sum_lb0 = sum(lamb_0[i][l] for i in range(n))
        e_new = (1.0 / n) * (sum_ef - (1.0 / rho) * sum_lb0)
        dual_res_b.append(e_new - e_cloud[l])
        e_cloud[l] = e_new

    # 增广拉格朗日（云节点）
    A_0 = 0.0
    for l in range(L):
        for i in range(n):
            diff_w = w[l] - x[i][l]
            A_0 += (np.sum(mu_0[i][l] * diff_w)
                    + 0.5 * rho * np.sum(diff_w ** 2))
            diff_b = e_cloud[l] - e_fog[i][l]
            A_0 += (np.dot(lamb_0[i][l], diff_b)
                    + 0.5 * rho * np.sum(diff_b ** 2))

    return A_0, dual_res_w, dual_res_b


# ══════════════════════════════════════════════
# 预测（使用 MLP 云节点参数 w）
# ══════════════════════════════════════════════
def predict_mlp(X, w_cloud, e_cloud, L, threshold=0.5):
    """
    X        : shape=(N, M_0)
    w_cloud  : list[np.ndarray]，云节点权重
    e_cloud  : list[np.ndarray]，云节点偏置
    返回预测标签 0/1，shape=(N,)
    """
    preds = []
    for idx in range(X.shape[0]):
        _, c_list, _ = forward(X[idx], w_cloud, e_cloud, L)
        prob = c_list[-1][0]  # 标量
        preds.append(1 if prob >= threshold else 0)
    return np.array(preds)


def predict_mlp_edge(X, y_w, y_b, L, threshold=0.5):
    """单个边缘节点预测"""
    preds = []
    for idx in range(X.shape[0]):
        _, c_list, _ = forward(X[idx], y_w, y_b, L)
        prob = c_list[-1][0]
        preds.append(1 if prob >= threshold else 0)
    return np.array(preds)


# ══════════════════════════════════════════════
# 收敛判断辅助
# ══════════════════════════════════════════════
def _all_converged(residual_list, eps):
    """
    residual_list : list of list[np.ndarray]（每个节点对应 L 层残差）
    eps           : 收敛阈值
    """
    for res_layers in residual_list:
        for r in res_layers:
            if np.any(np.abs(r) > eps):
                return False
    return True


if __name__ == '__main__':
    np.random.seed(seed=42)

    # m = X_train_df['Machine_ID'].unique().shape[0]

    # 总设备数
    m = 50

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

    total_samples = 0
    for (fog_key, edge_key), data in client_data.items():
        X_sub = data['a']
        y_sub = data['b']
        n_samples = X_sub.shape[0]
        print(f"工厂 {fog_key} 设备 {edge_key} 样本数: {n_samples}")
        total_samples += n_samples

    print(f"\n所有设备总样本数: {total_samples}")

    # ── 网络结构 ──
    L = 2
    structure = [p, 16, 1]

    # L = 3
    # structure = [p, 32, 16, 1]

    weight_shapes = [(structure[l], structure[l - 1]) for l in range(1, L + 1)]
    bias_shapes = [structure[l] for l in range(1, L + 1)]

    # 云节点参数：w^l ∈ R^{M_l × M_{l-1}}，e^l ∈ R^{M_l}
    w = [np.zeros(shape) for shape in weight_shapes]  # 长度L的列表
    e_cloud = [np.zeros(size) for size in bias_shapes]  # 长度L的列表

    # # 雾节点参数：x_i^l，e_i^l
    # x = [[np.zeros(shape) for shape in weight_shapes] for _ in range(n)]
    # e_fog = [[np.zeros(size) for size in bias_shapes] for _ in range(n)]
    #
    # # 边缘节点参数：y_ij^l，e_ij^l
    # y = [[[np.zeros(shape) for shape in weight_shapes] for _ in range(m_i)] for _ in range(n)]
    # e_edge = [[[np.zeros(size) for size in bias_shapes] for _ in range(m_i)] for _ in range(n)]

    # ── Xavier 初始化（比全零更利于训练）──
    # def init_weights(shapes):
    #     ws = []
    #     for (out, inp) in shapes:
    #         limit = np.sqrt(6.0 / (inp + out))
    #         ws.append(np.random.uniform(-limit, limit, (out, inp)))
    #     return ws

    def init_weights(shapes, std=0.01):
        ws = []
        for (out, inp) in shapes:
            ws.append(np.random.normal(0, std, (out, inp)))
        return ws

    def init_biases(sizes):
        return [np.zeros(s) for s in sizes]

    # 云节点
    w = init_weights(weight_shapes)
    e_cloud = init_biases(bias_shapes)

    # 雾节点
    x = [init_weights(weight_shapes) for _ in range(n)]
    e_fog = [init_biases(bias_shapes) for _ in range(n)]

    # 边缘节点
    y = [[init_weights(weight_shapes) for _ in range(m_i)] for _ in range(n)]
    e_edge = [[init_biases(bias_shapes) for _ in range(m_i)] for _ in range(n)]

    # 拉格朗日乘子（权重）：mu_0i^l（Fog-Cloud），mu_ij^l（Edge-Fog）
    mu_0 = [[np.zeros(shape) for shape in weight_shapes] for _ in range(n)]
    mu = [[[np.zeros(shape) for shape in weight_shapes] for _ in range(m_i)] for _ in range(n)]

    # 拉格朗日乘子（偏置）：lambda_0i^l（Fog-Cloud），lambda_ij^l（Edge-Fog）
    lamb_0 = [[np.zeros(size) for size in bias_shapes] for _ in range(n)]
    lamb = [[[np.zeros(size) for size in bias_shapes] for _ in range(m_i)] for _ in range(n)]

    # ── 超参数 ──
    eps_pri = 1e-5
    eps_dual = 1e-5
    v_max = 1
    rho = 1.0
    alpha = 0.001  # 梯度下降步长
    E = 10
    max_iter = 101

    N_iter = 0

    # ── 日志 ──
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

    # ── 训练历史 ──
    F_values = []
    avg_loss_history = []
    avg_accuracies_history = []
    std_history = []

    N_iter = 0

    # ── 对偶 / 原始残差存储（按层列表） ──
    dual_res_edge_w = {}  # (i,j) -> list[L个ndarray]
    dual_res_edge_b = {}
    dual_res_fog_w = {}  # i     -> list[L个ndarray]
    dual_res_fog_b = {}
    dual_res_cloud_w = None
    dual_res_cloud_b = None

    pri_res_edge_w = {}  # (i,j) -> list[L个ndarray]
    pri_res_edge_b = {}
    pri_res_fog_w = {}  # i     -> list[L个ndarray]
    pri_res_fog_b = {}

    acc_stable_window = deque(maxlen=10)  # 存最近10次 (mean_acc, std_acc)

    for k in range(2001):  # 可以根据需要调整迭代次数
        v = 0
        temp_1 = 0  # 对偶收敛计数（初始化，外层也需要）
        while True and (not (v == v_max)):
            N_iter += 1
            v += 1
            F = []
            loss = []

            # ── Step 1：边缘节点更新 ──
            for i in range(n):
                for j in range(m_i):
                    (F_ij, loss_ij,
                     dr_w, dr_b) = mlp_grad_solver_edge_i_j(
                        client_data, x, e_fog,
                        y, e_edge,
                        i, j,
                        mu, lamb,
                        d, alpha, rho, L, E
                    )
                    dual_res_edge_w[(i, j)] = dr_w
                    dual_res_edge_b[(i, j)] = dr_b
                    F.append(F_ij)
                    loss.append(loss_ij)

            # ── Step 2：雾节点更新 ──
            for i in range(n):
                A_i, dr_w, dr_b = analytical_solution_fog_i(
                    w, e_cloud, x, e_fog, y, e_edge,
                    i, m_i, mu_0, mu, lamb_0, lamb,
                    rho, L
                )
                dual_res_fog_w[i] = dr_w
                dual_res_fog_b[i] = dr_b
                F.append(A_i)

            # ── Step 3：云节点更新 ──
            A_0, dr_w, dr_b = analytical_solution_cloud(
                w, e_cloud, x, e_fog,
                mu_0, lamb_0, n, rho, L
            )
            dual_res_cloud_w = dr_w
            dual_res_cloud_b = dr_b
            F.append(A_0)

            F_values.append(sum(F))
            avg_loss_history.append(sum(loss))

            # ── 精度评估（用边缘节点参数） ──
            acc = []
            for i in range(n):
                for j in range(m_i):
                    preds = predict_mlp_edge(X_test, y[i][j], e_edge[i][j], L)
                    acc.append(np.mean(preds == y_test))

            acc_array = np.array(acc)
            mean_acc = np.mean(acc_array)
            std_acc = np.std(acc_array)
            avg_accuracies_history.append(mean_acc)
            std_history.append(std_acc)

            logger.info(
                f"第{k + 1: >2}-{v: >2}({N_iter: >4})次迭代: F_Value: {F_values[-1]},  avg_loss: {avg_loss_history[-1]}, mean_acc: {mean_acc}, std_acc: {std_acc}")

            # ── 新增：精度稳定停止标准 ──
            acc_stable_window.append((mean_acc, std_acc))
            if len(acc_stable_window) == 10:
                diffs = [
                    abs(acc_stable_window[t][0] - acc_stable_window[t - 1][0]) == 0 and
                    abs(acc_stable_window[t][1] - acc_stable_window[t - 1][1]) == 0
                    for t in range(1, 10)
                ]
                if all(diffs):
                    logger.info(f'精度连续10次无变化，算法停止 at k={k + 1}, v={v}')
                    N_iter = max_iter  # 触发外层 break
                    break

            if N_iter == max_iter:
                break

            # ── 对偶收敛判断 ──
            all_dual_w = (list(dual_res_edge_w.values())
                          + list(dual_res_fog_w.values())
                          + [dual_res_cloud_w])
            all_dual_b = (list(dual_res_edge_b.values())
                          + list(dual_res_fog_b.values())
                          + [dual_res_cloud_b])

            if (_all_converged(all_dual_w, eps_dual) and
                    _all_converged(all_dual_b, eps_dual)):
                logger.info(f'对偶残差收敛，break at k={k + 1}, v={v}')
                temp_1 = 1
                break

        # ── 拉格朗日乘子更新 ──
        for i in range(n):
            for j in range(m_i):
                # 原始残差：x_i^l - y_ij^l
                pri_res_edge_w[(i, j)] = [x[i][l] - y[i][j][l] for l in range(L)]
                pri_res_edge_b[(i, j)] = [e_fog[i][l] - e_edge[i][j][l] for l in range(L)]
                for l in range(L):
                    mu[i][j][l] += rho * pri_res_edge_w[(i, j)][l]
                    lamb[i][j][l] += rho * pri_res_edge_b[(i, j)][l]

        for i in range(n):
            # 原始残差：w^l - x_i^l
            pri_res_fog_w[i] = [w[l] - x[i][l] for l in range(L)]
            pri_res_fog_b[i] = [e_cloud[l] - e_fog[i][l] for l in range(L)]
            for l in range(L):
                mu_0[i][l] += rho * pri_res_fog_w[i][l]
                lamb_0[i][l] += rho * pri_res_fog_b[i][l]

        if N_iter >= max_iter:
            break

        # ── 原始收敛判断 ──
        all_pri_w = list(pri_res_edge_w.values()) + list(pri_res_fog_w.values())
        all_pri_b = list(pri_res_edge_b.values()) + list(pri_res_fog_b.values())

        if (temp_1 == 1 and
                _all_converged(all_pri_w, eps_pri) and
                _all_converged(all_pri_b, eps_pri)):
            logger.info(f'原始残差也收敛，全局终止 at k={k + 1}')
            break
    # 计算归一化的损失变化
    normalized_loss = abs((np.array(avg_loss_history) - avg_loss_history[-1])) / avg_loss_history[-1]

    # 创建 DataFrame
    df = pd.DataFrame({
        "F_values": F_values,
        "Avg_Loss": avg_loss_history,
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

    a = 1
