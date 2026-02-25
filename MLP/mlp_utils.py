# -*- coding: UTF-8 -*-
"""
@Project ：代码_new 
@File    ：mlp_utils.py
@Author  ：Wenyou Guo
@Date    ：2026/2/24 20:10 
"""
"""
mlp_module.py
-------------
在 DALD-HC 框架下，为边缘节点 (i, j) 提供 MLP 的
  - 前向传播
  - 反向传播（误差项计算）
  - 参数梯度与一步梯度下降更新

参数布局（均为 list，长度 = L，按层索引 l = 0,1,...,L-1）
  y_w  : list[np.ndarray]  权重矩阵，y_w[l].shape = (M_{l+1}, M_l)
  y_b  : list[np.ndarray]  偏置向量，y_b[l].shape = (M_{l+1},)

对应文档符号：
  y_ij^{l}  -> y_w[l-1]   （文档 l 从 1 开始，代码 l 从 0 开始）
  e_ij^{l}  -> y_b[l-1]
"""

import numpy as np
import copy


# ─────────────────────────────────────────────
# 激活函数
# ─────────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z):
    return np.maximum(0.0, z)


def relu_deriv(z):
    return (z > 0).astype(float)


def linear(z):
    return z


def linear_deriv(z):
    return np.ones_like(z)


# 按层配置激活函数（隐藏层 ReLU，输出层 Sigmoid 适用于二分类）
def get_activation(l, L):
    """
    l: 0-based 层索引（0 ~ L-1）
    L: 总层数
    返回 (激活函数, 激活函数导数)
    """
    if l < L - 1:          # 隐藏层
        return relu, relu_deriv
    else:                  # 输出层
        return sigmoid, sigmoid_deriv


# ─────────────────────────────────────────────
# 前向传播
# 返回：
#   z_list  : list[np.ndarray]  净输入，z_list[l].shape = (M_{l+1},)
#   c_list  : list[np.ndarray]  激活输出，c_list[l].shape = (M_{l+1},)
#             c_list[-1] 即网络预测输出 b_hat ∈ (0,1)
#   c_input : np.ndarray        输入特征，shape = (M_0,)
# ─────────────────────────────────────────────
def forward(a, y_w, y_b, L):
    """
    a     : 单样本输入特征向量，shape = (M_0,)
    y_w   : list[np.ndarray]，y_w[l].shape = (M_{l+1}, M_l)
    y_b   : list[np.ndarray]，y_b[l].shape = (M_{l+1},)
    L     : 网络层数
    """
    c = a.copy()            # c_ij^0
    c_list = []             # c_ij^1 ... c_ij^L
    z_list = []             # z_ij^1 ... z_ij^L
    c_input = c.copy()      # 保存输入，便于梯度计算

    for l in range(L):
        z = y_w[l] @ c + y_b[l]          # z_ij^{l+1}，shape=(M_{l+1},)
        act, _ = get_activation(l, L)
        c = act(z)                         # c_ij^{l+1}
        z_list.append(z)
        c_list.append(c)

    return z_list, c_list, c_input


# ─────────────────────────────────────────────
# 单样本损失（二元交叉熵）及对输出层激活的梯度
#   l(b_hat, b) = -b*log(b_hat) - (1-b)*log(1-b_hat)
#   ∂l/∂b_hat  = -b/b_hat + (1-b)/(1-b_hat)
# ─────────────────────────────────────────────
def loss_grad_output(b_hat, b):
    """
    b_hat : 标量，网络输出（sigmoid 后）
    b     : 标量，真实标签（0 或 1）
    返回 ∂l/∂b_hat，标量
    """
    eps = 1e-12
    b_hat = np.clip(b_hat, eps, 1 - eps)
    return -b / b_hat + (1 - b) / (1 - b_hat)


# ─────────────────────────────────────────────
# 反向传播（批量）
# 返回：
#   grad_w  : list[np.ndarray]  ∂f_ij/∂y_w[l]，shape 同 y_w[l]
#   grad_b  : list[np.ndarray]  ∂f_ij/∂y_b[l]，shape 同 y_b[l]
# ─────────────────────────────────────────────
def backward(X_batch, b_batch, y_w, y_b, L, d_total):
    """
    X_batch  : shape = (n_samples, M_0)
    b_batch  : shape = (n_samples,)，真实标签（0/1）
    y_w, y_b : 当前参数
    L        : 网络层数
    d_total  : 全局总样本数，用于归一化（对应文档中的 1/d）
    """
    n_samples = X_batch.shape[0]

    # 累积梯度（按文档公式，对所有样本求和后除以 d_total）
    grad_w = [np.zeros_like(w) for w in y_w]
    grad_b = [np.zeros_like(b) for b in y_b]

    for idx in range(n_samples):
        a = X_batch[idx]                          # (M_0,)
        b = b_batch[idx]                          # 标量

        # 前向
        z_list, c_list, c_input = forward(a, y_w, y_b, L)

        # 输出层误差项 δ^L（文档公式）
        # δ_ij^L = (∂f_ij/∂c^L) ⊙ (σ^L)'(z^L)
        b_hat = c_list[-1][0]                     # 标量（输出层 M_L=1）
        df_dc = np.array([loss_grad_output(b_hat, b)])   # shape=(1,)
        _, act_d = get_activation(L - 1, L)
        delta = df_dc * act_d(z_list[-1])         # shape=(M_L,)=(1,)

        # 逐层反向
        for l in range(L - 1, -1, -1):
            c_prev = c_list[l - 1] if l > 0 else c_input    # c^{l-1}，shape=(M_l,)

            # 文档：∂f_ij/∂y_w[l] += δ^{l+1} (c^l)^T
            grad_w[l] += np.outer(delta, c_prev)             # (M_{l+1}, M_l)
            grad_b[l] += delta                               # (M_{l+1},)

            if l > 0:
                # 传播误差项：δ^l = (y_w[l])^T δ^{l+1} ⊙ (σ^l)'(z^l)
                _, act_d = get_activation(l - 1, L)
                delta = (y_w[l].T @ delta) * act_d(z_list[l - 1])

    # 除以全局总样本数 d（文档中的 1/d 归一化）
    grad_w = [g / d_total for g in grad_w]
    grad_b = [g / d_total for g in grad_b]

    return grad_w, grad_b


# ─────────────────────────────────────────────
# 边缘节点单步参数更新（对应文档公式）：
#
#   y_w[l] ← y_w[l] - α [ ∂f/∂y_w[l]  - μ_w[l]  - ρ(x_w[l]  - y_w[l]) ]
#   y_b[l] ← y_b[l] - α [ ∂f/∂y_b[l]  - μ_b[l]  - ρ(x_b[l]  - y_b[l]) ]
#
# 其中 x_w, x_b 为对应雾节点参数（上一轮固定值）
# ─────────────────────────────────────────────
def edge_param_update(y_w, y_b,
                      grad_w, grad_b,
                      x_w, x_b,
                      mu_w, mu_b,
                      alpha, rho, L):
    """
    原地更新 y_w, y_b，返回 (new_y_w, new_y_b)。
    所有列表均长度为 L，各元素为 np.ndarray。
    """
    new_y_w = []
    new_y_b = []
    for l in range(L):
        # 权重梯度 = ∂f/∂y_w[l] - μ_w[l] - ρ*(x_w[l] - y_w[l])
        g_w = grad_w[l] - mu_w[l] - rho * (x_w[l] - y_w[l])
        new_y_w.append(y_w[l] - alpha * g_w)

        # 偏置梯度 = ∂f/∂y_b[l] - μ_b[l] - ρ*(x_b[l] - y_b[l])
        g_b = grad_b[l] - mu_b[l] - rho * (x_b[l] - y_b[l])
        new_y_b.append(y_b[l] - alpha * g_b)

    return new_y_w, new_y_b


# ─────────────────────────────────────────────
# 计算增广拉格朗日函数值（边缘节点局部）
# ─────────────────────────────────────────────
def compute_augmented_lagrangian(X_batch, b_batch, y_w, y_b,
                                  x_w, x_b, mu_w, mu_b,
                                  rho, L, d_total):
    """
    返回 (F_total, loss_avg)
    F_total  = f_ij(y) + A_rho^ij
    loss_avg = f_ij(y)  (归一化损失)
    """
    n_samples = X_batch.shape[0]
    eps = 1e-12
    total_loss = 0.0

    for idx in range(n_samples):
        _, c_list, _ = forward(X_batch[idx], y_w, y_b, L)
        b_hat = np.clip(c_list[-1][0], eps, 1 - eps)
        b = b_batch[idx]
        total_loss += -b * np.log(b_hat) - (1 - b) * np.log(1 - b_hat)

    loss_avg = total_loss / d_total

    # 惩罚项 A_rho^ij（权重 + 偏置）
    aug = 0.0
    for l in range(L):
        diff_w = x_w[l] - y_w[l]
        diff_b = x_b[l] - y_b[l]
        aug += (np.sum(mu_w[l] * diff_w)
                + 0.5 * rho * np.sum(diff_w ** 2)
                + np.dot(mu_b[l], diff_b)
                + 0.5 * rho * np.sum(diff_b ** 2))

    return loss_avg + aug, loss_avg