"""
DALD-HC with MLP (Multi-Layer Perceptron) backbone
====================================================
Architecture:
    Cloud  (1 global model:  w,       e_cloud)
      └── Fog nodes   (n fog models:  x[i],    e_fog[i])
            └── Edge devices (m_i edge models: y[i][j], e_edge[i][j])

Each node maintains separate weight matrices and bias vectors per layer.
Dual variables (mu, lamb) enforce consensus between adjacent tiers.
MLP utilities (forward, backward, edge_param_update, etc.) are imported
from mlp_utils.py.
"""

import copy
import logging
from collections import Counter, deque

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlp_utils import (forward, backward, edge_param_update,
                       compute_augmented_lagrangian, sigmoid)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Data Loading & Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

# Binary classification dataset; labels converted from {−1,+1} to {0,1}
df = pd.read_csv("./datasets/processed_dataset_time_2_cos&sin_binary_low&medium.csv")

X = df.drop(columns=["Efficiency_Status"])
y = df["Efficiency_Status"]

feature_names = X.columns  # preserve column names before array conversion

# Convert labels from ±1 encoding to 0/1 for binary cross-entropy
y = ((y + 1) / 2).astype(int)

# Stratified split to maintain class balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_train = y_train.values
y_test  = y_test.values

# Standardize: fit on train only, then apply the same transform to test
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Named DataFrames kept for analysis / debugging purposes
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df  = pd.DataFrame(X_test,  columns=feature_names)

# Note: no bias column prepended here — MLP handles biases explicitly


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Non-IID Data Partitioning (Dirichlet)
# ──────────────────────────────────────────────────────────────────────────────

def dirichlet_partition(y_array, num_clients, alpha=0.5, seed=42,
                        min_samples_per_client=1):
    """
    Partition sample indices across clients using a Dirichlet distribution.

    A small alpha produces highly heterogeneous (non-IID) splits;
    alpha → ∞ approaches a uniform (IID) split.

    Args:
        y_array (np.ndarray): Label array of the full dataset.
        num_clients (int): Number of clients to distribute data across.
        alpha (float): Dirichlet concentration parameter.
        seed (int): Random seed for reproducibility.
        min_samples_per_client (int): Minimum samples guaranteed per client.

    Returns:
        list[list[int]]: Per-client lists of sample indices.

    Raises:
        RuntimeError: If the constraint cannot be satisfied after 100 retries.
    """
    np.random.seed(seed)
    class_labels = np.unique(y_array)
    idx_by_class = {lbl: np.where(y_array == lbl)[0] for lbl in class_labels}

    for retry in range(1, 101):
        client_indices = [[] for _ in range(num_clients)]

        for label in class_labels:
            indices     = idx_by_class[label].copy()
            np.random.shuffle(indices)
            proportions = np.random.dirichlet([alpha] * num_clients)
            cuts        = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            for cid, split in enumerate(np.split(indices, cuts)):
                client_indices[cid].extend(split.tolist())

        if min(len(c) for c in client_indices) >= min_samples_per_client:
            return client_indices

    raise RuntimeError(
        "Cannot satisfy min_samples_per_client after 100 retries. "
        "Try increasing alpha or reducing num_clients."
    )


def split_data(X_array, y_array, num_fog, num_edge, alpha=0.5, seed=42):
    """
    Distribute data across a two-tier hierarchy of fog nodes and edge devices.

    Total clients = num_fog × num_edge.
    Client index cid maps to (fog_id = cid // num_edge, edge_id = cid % num_edge).

    Args:
        X_array (np.ndarray): Feature matrix.
        y_array (np.ndarray): Label array (0/1).
        num_fog (int): Number of fog nodes.
        num_edge (int): Edge devices per fog node.
        alpha (float): Dirichlet heterogeneity parameter.
        seed (int): Random seed.

    Returns:
        dict: {(fog_id, edge_id): {'a': X_subset, 'b': y_subset}}
    """
    assert X_array.shape[0] == y_array.shape[0], \
        "X and y must have the same number of samples."

    client_indices = dirichlet_partition(y_array, num_fog * num_edge, alpha, seed)

    client_data_ = {}
    for cid, indices in enumerate(client_indices):
        fog_id  = cid // num_edge
        edge_id = cid %  num_edge
        client_data_[(fog_id, edge_id)] = {
            "a": X_array[indices],
            "b": y_array[indices],
        }
    return client_data_


def print_device_label_distribution(client_data):
    """Print a summary table of label counts and positive-class ratio per device."""
    print(f"{'Fog':<5} {'Edge':<5} | {'Label Distribution':<28} | {'Total':<8} | {'Pos Ratio'}")
    print("-" * 72)
    for (i, j), data in sorted(client_data.items()):
        labels    = data["b"]
        counter   = Counter(labels)
        total     = len(labels)
        pos       = counter.get(1, 0)
        neg       = counter.get(0, 0)
        pos_ratio = pos / total if total > 0 else 0
        print(f"{i:<5} {j:<5} | 0: {neg:<6} 1: {pos:<6}          | {total:<8} | {pos_ratio:.2%}")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Convergence Helper
# ──────────────────────────────────────────────────────────────────────────────

def _all_converged(residual_list, eps):
    """
    Check whether all layer-wise residuals fall within the convergence threshold.

    Args:
        residual_list (list[list[np.ndarray]]): Per-node list of L-layer residuals.
        eps (float): Convergence threshold.

    Returns:
        bool: True if every element of every residual array satisfies |r| ≤ eps.
    """
    for res_layers in residual_list:
        for r in res_layers:
            if np.any(np.abs(r) > eps):
                return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Subproblem Solvers
# ──────────────────────────────────────────────────────────────────────────────

def mlp_grad_solver_edge_i_j(client_data, x, e_fog,
                              y, e_edge,
                              i, j,
                              mu, lamb,
                              d, alpha, rho, L, E):
    """
    Edge-level subproblem for DALD-HC (MLP version).

    Runs E gradient-descent steps on the augmented Lagrangian w.r.t. the
    edge model (y[i][j], e_edge[i][j]), with the fog model (x[i], e_fog[i])
    held fixed as the consensus anchor.

    The augmented Lagrangian at the edge level is:
        L_ij = f_ij(y_ij, e_ij)
               + Σ_l [ μ_ij^l ⊙ (x_i^l − y_ij^l)  + (ρ/2)‖x_i^l − y_ij^l‖² ]
               + Σ_l [ λ_ij^l · (e_i^l − e_ij^l)   + (ρ/2)‖e_i^l − e_ij^l‖² ]

    Gradient computation and parameter update are delegated to mlp_utils:
        - backward()          computes ∂f_ij / ∂y_w and ∂f_ij / ∂y_b
        - edge_param_update() applies the gradient step with augmented terms

    Args:
        client_data (dict): Global data dictionary {(i,j): {'a': X, 'b': y}}.
        x     (list[list[np.ndarray]]): Fog weight parameters; x[i][l] ∈ R^{M_{l+1}×M_l}.
        e_fog (list[list[np.ndarray]]): Fog bias parameters;   e_fog[i][l] ∈ R^{M_{l+1}}.
        y     (list[list[list[np.ndarray]]]): Edge weight parameters (updated in-place).
        e_edge(list[list[list[np.ndarray]]]): Edge bias parameters  (updated in-place).
        i, j  (int): Fog and edge indices.
        mu    (list[list[list[np.ndarray]]]): Dual variables for weights (fog-edge).
        lamb  (list[list[list[np.ndarray]]]): Dual variables for biases  (fog-edge).
        d     (int): Total training samples (loss normalisation denominator).
        alpha (float): Gradient-descent step size.
        rho   (float): Penalty parameter.
        L     (int): Number of network layers.
        E     (int): Number of local gradient steps.

    Returns:
        F_total       (float): Augmented Lagrangian value after the update.
        loss_avg      (float): Normalised cross-entropy loss over local data.
        dual_residual_w (list[np.ndarray]): Per-layer weight residuals (y_new − y_old).
        dual_residual_b (list[np.ndarray]): Per-layer bias residuals   (e_new − e_old).
    """
    X_batch = client_data[(i, j)]["a"]   # (n_samples, M_0)
    b_batch = client_data[(i, j)]["b"]   # (n_samples,)

    # Deep-copy current parameters to avoid side effects during the update
    cur_y_w = copy.deepcopy(y[i][j])         # list of L weight arrays
    cur_y_b = copy.deepcopy(e_edge[i][j])    # list of L bias arrays

    # Fog parameters are held fixed as the consensus anchor for this round
    x_w_fixed = x[i]       # list of L weight arrays
    x_b_fixed = e_fog[i]   # list of L bias arrays

    mu_w = mu[i][j]     # fog-edge dual variables for weights
    mu_b = lamb[i][j]   # fog-edge dual variables for biases

    for _ in range(E):
        # Compute gradients of the local loss w.r.t. weights and biases
        grad_w, grad_b = backward(X_batch, b_batch, cur_y_w, cur_y_b, L, d)

        # Gradient-descent step including augmented Lagrangian penalty terms
        cur_y_w, cur_y_b = edge_param_update(
            cur_y_w, cur_y_b,
            grad_w, grad_b,
            x_w_fixed, x_b_fixed,
            mu_w, mu_b,
            alpha, rho, L
        )

    # Compute per-layer dual residuals: parameter change over E steps
    dual_residual_w = [cur_y_w[l] - y[i][j][l]     for l in range(L)]
    dual_residual_b = [cur_y_b[l] - e_edge[i][j][l] for l in range(L)]

    # Write updated parameters back in-place
    y[i][j]      = cur_y_w
    e_edge[i][j] = cur_y_b

    # Evaluate augmented Lagrangian at the final iterate
    F_total, loss_avg = compute_augmented_lagrangian(
        X_batch, b_batch,
        cur_y_w, cur_y_b,
        x_w_fixed, x_b_fixed,
        mu_w, mu_b,
        rho, L, d
    )

    return F_total, loss_avg, dual_residual_w, dual_residual_b


def analytical_solution_fog_i(w, e_cloud,
                               x, e_fog,
                               y, e_edge,
                               i, m_i,
                               mu_0, mu,
                               lamb_0, lamb,
                               rho, L):
    """
    Fog-level subproblem for DALD-HC (MLP version): closed-form update.

    Setting ∂L / ∂x_i^l = 0 and ∂L / ∂e_i^l = 0 yields per-layer updates:

        x_i^l ← (1/(m_i+1)) · [ w^l + Σ_j y_ij^l + (1/ρ)(μ_0i^l − Σ_j μ_ij^l) ]
        e_i^l ← (1/(m_i+1)) · [ e^l + Σ_j e_ij^l + (1/ρ)(λ_0i^l − Σ_j λ_ij^l) ]

    Args:
        w       (list[np.ndarray]): Cloud weight parameters w^l.
        e_cloud (list[np.ndarray]): Cloud bias parameters e^l.
        x       (list[list[np.ndarray]]): Fog weight parameters (updated in-place).
        e_fog   (list[list[np.ndarray]]): Fog bias parameters   (updated in-place).
        y       (list[list[list[np.ndarray]]]): Edge weight parameters.
        e_edge  (list[list[list[np.ndarray]]]): Edge bias parameters.
        i       (int): Fog node index.
        m_i     (int): Number of edge devices per fog node.
        mu_0    (list[list[np.ndarray]]): Cloud-fog dual variables for weights.
        mu      (list[list[list[np.ndarray]]]): Fog-edge dual variables for weights.
        lamb_0  (list[list[np.ndarray]]): Cloud-fog dual variables for biases.
        lamb    (list[list[list[np.ndarray]]]): Fog-edge dual variables for biases.
        rho     (float): Penalty parameter.
        L       (int): Number of network layers.

    Returns:
        A_i          (float): Local augmented Lagrangian contribution for fog i.
        dual_res_w   (list[np.ndarray]): Per-layer weight dual residuals (x_new − x_old).
        dual_res_b   (list[np.ndarray]): Per-layer bias dual residuals   (e_new − e_old).
    """
    dual_res_w = []
    dual_res_b = []

    for l in range(L):
        # ── Weight update (closed-form) ──
        sum_y   = sum(y[i][j][l]   for j in range(m_i))
        sum_mu  = sum(mu[i][j][l]  for j in range(m_i))
        x_new   = (1.0 / (m_i + 1)) * (
            w[l] + sum_y + (1.0 / rho) * (mu_0[i][l] - sum_mu)
        )
        dual_res_w.append(x_new - x[i][l])
        x[i][l] = x_new

        # ── Bias update (closed-form) ──
        sum_eb  = sum(e_edge[i][j][l] for j in range(m_i))
        sum_lb  = sum(lamb[i][j][l]   for j in range(m_i))
        eb_new  = (1.0 / (m_i + 1)) * (
            e_cloud[l] + sum_eb + (1.0 / rho) * (lamb_0[i][l] - sum_lb)
        )
        dual_res_b.append(eb_new - e_fog[i][l])
        e_fog[i][l] = eb_new

    # Augmented Lagrangian contribution for fog node i (used for monitoring)
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


def analytical_solution_cloud(w, e_cloud,
                               x, e_fog,
                               mu_0, lamb_0,
                               n, rho, L):
    """
    Cloud-level subproblem for DALD-HC (MLP version): closed-form update.

    Setting ∂L / ∂w^l = 0 and ∂L / ∂e^l = 0 yields per-layer updates:

        w^l ← (1/n) · [ Σ_i x_i^l − (1/ρ) Σ_i μ_0i^l ]
        e^l ← (1/n) · [ Σ_i e_i^l  − (1/ρ) Σ_i λ_0i^l ]

    Args:
        w       (list[np.ndarray]): Cloud weight parameters (updated in-place).
        e_cloud (list[np.ndarray]): Cloud bias parameters   (updated in-place).
        x       (list[list[np.ndarray]]): Fog weight parameters.
        e_fog   (list[list[np.ndarray]]): Fog bias parameters.
        mu_0    (list[list[np.ndarray]]): Cloud-fog dual variables for weights.
        lamb_0  (list[list[np.ndarray]]): Cloud-fog dual variables for biases.
        n       (int): Number of fog nodes.
        rho     (float): Penalty parameter.
        L       (int): Number of network layers.

    Returns:
        A_0        (float): Cloud-level augmented Lagrangian value.
        dual_res_w (list[np.ndarray]): Per-layer weight dual residuals (w_new − w_old).
        dual_res_b (list[np.ndarray]): Per-layer bias dual residuals   (e_new − e_old).
    """
    dual_res_w = []
    dual_res_b = []

    for l in range(L):
        # ── Weight update (closed-form) ──
        sum_x    = sum(x[i][l]    for i in range(n))
        sum_mu0  = sum(mu_0[i][l] for i in range(n))
        w_new    = (1.0 / n) * (sum_x - (1.0 / rho) * sum_mu0)
        dual_res_w.append(w_new - w[l])
        w[l] = w_new

        # ── Bias update (closed-form) ──
        sum_ef   = sum(e_fog[i][l]   for i in range(n))
        sum_lb0  = sum(lamb_0[i][l]  for i in range(n))
        e_new    = (1.0 / n) * (sum_ef - (1.0 / rho) * sum_lb0)
        dual_res_b.append(e_new - e_cloud[l])
        e_cloud[l] = e_new

    # Cloud-level augmented Lagrangian value (used for monitoring)
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


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Inference
# ──────────────────────────────────────────────────────────────────────────────

def predict_mlp(X, w_cloud, e_cloud, L, threshold=0.5):
    """
    Inference using the global cloud MLP model.

    Args:
        X        (np.ndarray): Input features, shape (N, M_0).
        w_cloud  (list[np.ndarray]): Cloud weight parameters.
        e_cloud  (list[np.ndarray]): Cloud bias parameters.
        L        (int): Number of network layers.
        threshold (float): Decision threshold for the sigmoid output.

    Returns:
        np.ndarray: Predicted labels in {0, 1}, shape (N,).
    """
    preds = []
    for idx in range(X.shape[0]):
        _, c_list, _ = forward(X[idx], w_cloud, e_cloud, L)
        prob = c_list[-1][0]   # scalar sigmoid output
        preds.append(1 if prob >= threshold else 0)
    return np.array(preds)


def predict_mlp_edge(X, y_w, y_b, L, threshold=0.5):
    """
    Inference using a single edge device MLP model.

    Args:
        X         (np.ndarray): Input features, shape (N, M_0).
        y_w       (list[np.ndarray]): Edge weight parameters.
        y_b       (list[np.ndarray]): Edge bias parameters.
        L         (int): Number of network layers.
        threshold (float): Decision threshold for the sigmoid output.

    Returns:
        np.ndarray: Predicted labels in {0, 1}, shape (N,).
    """
    preds = []
    for idx in range(X.shape[0]):
        _, c_list, _ = forward(X[idx], y_w, y_b, L)
        prob = c_list[-1][0]
        preds.append(1 if prob >= threshold else 0)
    return np.array(preds)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Main: DALD-HC (MLP) Training
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    # ── Topology ──────────────────────────────────────────────────────────────
    n   = 10          # number of fog nodes
    m   = 50          # total edge devices (must be divisible by n)
    m_i = m // n      # edge devices per fog node

    # ── Dataset dimensions ────────────────────────────────────────────────────
    d = X_train.shape[0]   # total training samples
    p = X_train.shape[1]   # input feature dimension M_0

    # ── Data partitioning ─────────────────────────────────────────────────────
    # alpha=1.0 → nearly IID; decrease alpha for stronger heterogeneity
    client_data = split_data(X_train, y_train, num_fog=n, num_edge=m_i, alpha=1.0)
    print_device_label_distribution(client_data)

    total_samples = 0
    for (fog_key, edge_key), data in client_data.items():
        n_samples = data["a"].shape[0]
        print(f"Fog {fog_key}  Edge {edge_key}: {n_samples} samples")
        total_samples += n_samples
    print(f"\nTotal samples across all devices: {total_samples}")

    # ── Network architecture ──────────────────────────────────────────────────
    L         = 2
    structure = [p, 16, 1]          # input → hidden → output
    # L         = 3
    # structure = [p, 32, 16, 1]    # alternative deeper architecture

    weight_shapes = [(structure[l], structure[l - 1]) for l in range(1, L + 1)]
    bias_shapes   = [structure[l] for l in range(1, L + 1)]

    # ── Parameter initialisation ──────────────────────────────────────────────
    def init_weights(shapes, std=0.01):
        """Initialise weight matrices with small Gaussian noise."""
        return [np.random.normal(0, std, (out, inp)) for (out, inp) in shapes]

    def init_biases(sizes):
        """Initialise bias vectors to zero."""
        return [np.zeros(s) for s in sizes]

    # Cloud node parameters
    w       = init_weights(weight_shapes)   # w[l] ∈ R^{M_{l+1} × M_l}
    e_cloud = init_biases(bias_shapes)      # e_cloud[l] ∈ R^{M_{l+1}}

    # Fog node parameters
    x     = [init_weights(weight_shapes) for _ in range(n)]
    e_fog = [init_biases(bias_shapes)    for _ in range(n)]

    # Edge node parameters
    y      = [[init_weights(weight_shapes) for _ in range(m_i)] for _ in range(n)]
    e_edge = [[init_biases(bias_shapes)    for _ in range(m_i)] for _ in range(n)]

    # ── Dual variable initialisation ──────────────────────────────────────────
    # mu_0[i][l], lamb_0[i][l]: cloud-fog consensus multipliers for weights / biases
    # mu[i][j][l], lamb[i][j][l]: fog-edge consensus multipliers for weights / biases
    mu_0   = [[np.zeros(shape) for shape in weight_shapes] for _ in range(n)]
    mu     = [[[np.zeros(shape) for shape in weight_shapes] for _ in range(m_i)] for _ in range(n)]
    lamb_0 = [[np.zeros(size)  for size  in bias_shapes]   for _ in range(n)]
    lamb   = [[[np.zeros(size)  for size  in bias_shapes]   for _ in range(m_i)] for _ in range(n)]

    # ── Convergence bookkeeping ───────────────────────────────────────────────
    # Residuals stored as {key: list[L arrays]} to handle per-layer MLP structure
    dual_res_edge_w  = {}   # (i,j) → list[L arrays]
    dual_res_edge_b  = {}
    dual_res_fog_w   = {}   # i     → list[L arrays]
    dual_res_fog_b   = {}
    dual_res_cloud_w = None
    dual_res_cloud_b = None

    pri_res_edge_w   = {}   # (i,j) → list[L arrays]
    pri_res_edge_b   = {}
    pri_res_fog_w    = {}   # i     → list[L arrays]
    pri_res_fog_b    = {}

    F_values               = []
    avg_loss_history       = []
    avg_accuracies_history = []
    std_history            = []

    # Sliding window to detect accuracy plateau (early stopping)
    acc_stable_window = deque(maxlen=10)

    # ── Hyperparameters ───────────────────────────────────────────────────────
    eps_pri  = 1e-5    # primal residual convergence tolerance
    eps_dual = 1e-5    # dual residual convergence tolerance
    v_max    = 1       # inner iterations per outer communication round
    rho      = 1.0     # penalty parameter
    alpha    = 0.001   # gradient-descent step size
    E        = 10      # local gradient steps per edge update
    max_iter = 101     # hard cap on total inner iterations

    N_iter = 0         # global iteration counter

    # ── Logging setup ─────────────────────────────────────────────────────────
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    log_path = (
        f"./log/DALD-HC-MLP_E{E}_rho{rho}_n{n}_m{m}_mi{m_i}"
        f"_vmax{v_max}_epspri{eps_pri}_epsdual{eps_dual}_alpha{alpha}.log"
    )
    fh = logging.FileHandler(log_path, "w")
    ch = logging.StreamHandler()

    fmt = logging.Formatter("%(message)s")
    fh.setFormatter(fmt);  fh.setLevel(logging.DEBUG)
    ch.setFormatter(fmt);  ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # ── DALD-HC (MLP) main loop ───────────────────────────────────────────────
    for k in range(2001):
        v      = 0
        temp_1 = 0   # dual convergence flag (1 = converged)

        # Inner loop: run up to v_max primal updates before a dual step
        while v < v_max:
            N_iter += 1
            v      += 1
            F, loss = [], []

            # Step 1 – Edge update: E gradient steps on augmented Lagrangian
            for i in range(n):
                for j in range(m_i):
                    F_ij, loss_ij, dr_w, dr_b = mlp_grad_solver_edge_i_j(
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

            # Step 2 – Fog update: closed-form per-layer solution
            for i in range(n):
                A_i, dr_w, dr_b = analytical_solution_fog_i(
                    w, e_cloud, x, e_fog, y, e_edge,
                    i, m_i, mu_0, mu, lamb_0, lamb, rho, L
                )
                dual_res_fog_w[i] = dr_w
                dual_res_fog_b[i] = dr_b
                F.append(A_i)

            # Step 3 – Cloud update: closed-form per-layer solution
            A_0, dr_w, dr_b = analytical_solution_cloud(
                w, e_cloud, x, e_fog,
                mu_0, lamb_0, n, rho, L
            )
            dual_res_cloud_w = dr_w
            dual_res_cloud_b = dr_b
            F.append(A_0)

            F_values.append(sum(F))
            avg_loss_history.append(sum(loss))

            # Evaluate per-device test accuracy using edge model parameters
            acc = [
                np.mean(predict_mlp_edge(X_test, y[i][j], e_edge[i][j], L) == y_test)
                for i in range(n)
                for j in range(m_i)
            ]
            acc_array = np.array(acc)
            mean_acc  = acc_array.mean()
            std_acc   = acc_array.std()
            avg_accuracies_history.append(mean_acc)
            std_history.append(std_acc)

            logger.info(
                f"Iter {k+1:>3}-{v:>2} ({N_iter:>4}): "
                f"F={F_values[-1]:.6f}  loss={avg_loss_history[-1]:.6f}  "
                f"acc={mean_acc:.4f}±{std_acc:.4f}"
            )

            # Early stopping: halt if accuracy has not changed for 10 consecutive iters
            acc_stable_window.append((mean_acc, std_acc))
            if len(acc_stable_window) == 10:
                plateau = all(
                    acc_stable_window[t][0] == acc_stable_window[t - 1][0] and
                    acc_stable_window[t][1] == acc_stable_window[t - 1][1]
                    for t in range(1, 10)
                )
                if plateau:
                    logger.info(f"Accuracy unchanged for 10 iters — early stop at k={k+1}, v={v}.")
                    N_iter = max_iter   # trigger outer break
                    break

            if N_iter == max_iter:
                break

            # Check dual convergence across all residual lists
            all_dual_w = (list(dual_res_edge_w.values())
                          + list(dual_res_fog_w.values())
                          + [dual_res_cloud_w])
            all_dual_b = (list(dual_res_edge_b.values())
                          + list(dual_res_fog_b.values())
                          + [dual_res_cloud_b])

            if _all_converged(all_dual_w, eps_dual) and _all_converged(all_dual_b, eps_dual):
                logger.info(f"Dual converged at k={k+1}, v={v}.")
                temp_1 = 1
                break

        # ── Dual variable update (multiplier step) ────────────────────────────
        # Fog-edge multipliers
        for i in range(n):
            for j in range(m_i):
                pri_res_edge_w[(i, j)] = [x[i][l] - y[i][j][l]     for l in range(L)]
                pri_res_edge_b[(i, j)] = [e_fog[i][l] - e_edge[i][j][l] for l in range(L)]
                for l in range(L):
                    mu[i][j][l]   += rho * pri_res_edge_w[(i, j)][l]
                    lamb[i][j][l] += rho * pri_res_edge_b[(i, j)][l]

        # Cloud-fog multipliers
        for i in range(n):
            pri_res_fog_w[i] = [w[l]       - x[i][l]    for l in range(L)]
            pri_res_fog_b[i] = [e_cloud[l] - e_fog[i][l] for l in range(L)]
            for l in range(L):
                mu_0[i][l]   += rho * pri_res_fog_w[i][l]
                lamb_0[i][l] += rho * pri_res_fog_b[i][l]

        if N_iter >= max_iter:
            break

        # Check primal convergence across all residual lists
        all_pri_w = list(pri_res_edge_w.values()) + list(pri_res_fog_w.values())
        all_pri_b = list(pri_res_edge_b.values()) + list(pri_res_fog_b.values())

        if (temp_1 == 1
                and _all_converged(all_pri_w, eps_pri)
                and _all_converged(all_pri_b, eps_pri)):
            logger.info(f"Full convergence (dual + primal) at k={k+1}.")
            break

    # ──────────────────────────────────────────────────────────────────────────
    # 7.  Save Results
    # ──────────────────────────────────────────────────────────────────────────

    # Normalised loss: relative distance from the final loss value
    normalized_loss = np.abs(
        np.array(avg_loss_history) - avg_loss_history[-1]
    ) / avg_loss_history[-1]

    results_df = pd.DataFrame({
        "F_values":        F_values,
        "Avg_Loss":        avg_loss_history,
        "Normalized_Loss": normalized_loss,
        "Mean_Acc":        avg_accuracies_history,
        "Std_Acc":         std_history,
    })

    csv_path = (
        f"./csv/DALD-HC-MLP_E{E}_rho{rho}_n{n}_m{m}_mi{m_i}"
        f"_vmax{v_max}_iter{N_iter}_epspri{eps_pri}_epsdual{eps_dual}_alpha{alpha}.csv"
    )
    results_df.to_csv(csv_path, index=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Fog nodes (n):              {n}")
    logger.info(f"Edge devices per fog (m_i): {m_i}")
    logger.info(f"Network structure:          {structure}")
    logger.info(f"Local steps (E):            {E}")
    logger.info(f"Step size (alpha):          {alpha}")
    logger.info(f"Total iterations:           {N_iter}")
    logger.info(f"v_max:                      {v_max}")
    logger.info(f"Mean test accuracy:         {mean_acc:.8f}")
    logger.info(f"Std  test accuracy:         {std_acc:.8f}")
    logger.info(f"Cloud weights w:            {w}")