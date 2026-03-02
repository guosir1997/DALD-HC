import copy
import logging
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Data Loading & Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

# Binary classification dataset (labels encoded as ±1)
df = pd.read_csv("./datasets/processed_dataset_time_2_cos&sin_binary_low&medium.csv")

X = df.drop(columns=["Efficiency_Status"])
y = df["Efficiency_Status"]

feature_names = X.columns  # preserve column names before array conversion

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

# Prepend bias column (column of ones) so weights include intercept
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test  = np.c_[np.ones(X_test.shape[0]),  X_test]


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
        X_array (np.ndarray): Feature matrix (with bias column prepended).
        y_array (np.ndarray): Label array (±1).
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
        pos       = counter.get(1,  0)
        neg       = counter.get(-1, 0)
        pos_ratio = pos / total if total > 0 else 0
        print(f"{i:<5} {j:<5} | -1: {neg:<6} 1: {pos:<6}         | {total:<8} | {pos_ratio:.2%}")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Subproblem Solvers
# ──────────────────────────────────────────────────────────────────────────────

def logistic_grad_solver_edge_i_j(x, y, i, j, mu, d, alpha, rho):
    """
    Edge-level local update for FedNova-HC.

    Each edge device starts from the current fog model x[i] and runs E − 1
    steps of gradient descent on its local logistic loss (no dual / penalty
    terms).  After the local steps the normalised gradient accumulator s_ij
    is computed as:
        s_ij = (x_i − y_ij_final) / E

    This normalised gradient is later used by the fog node to correct for
    heterogeneous local step counts (the core idea of FedNova).

    Local objective:
        f_ij(y_ij) = (1/d) Σ_k log(1 + exp(−b_k · a_k^T y_ij))

    Args:
        x   (list[np.ndarray]): Fog-level parameters x[i]; used as the
                                starting point for local updates.
        y   (list[list[np.ndarray]]): Edge-level parameters y[i][j]
                                      (updated in-place).
        i, j (int): Fog and edge indices.
        mu  (list[list[np.ndarray]]): Dual variables (unused in FedNova-HC,
                                      kept for interface consistency).
        d   (int): Total training samples (loss normalisation denominator).
        alpha (float): Gradient-descent step size.
        rho (float): Penalty parameter (unused in FedNova-HC,
                     kept for interface consistency).

    Returns:
        F_total (float): Local logistic loss (no augmented terms).
        loss_avg (float): Average logistic loss over local data.
        dual_residual (np.ndarray): y_new − y_old.

    Side effect:
        Updates s_ij[i][j] with the normalised gradient accumulator.
    """
    # FedNova-HC: initialise local model from the fog model x[i], not y[i][j]
    updated_y = copy.deepcopy(x[i])
    a, b      = client_data[i, j]["a"], client_data[i, j]["b"]

    for _ in range(E - 1):
        # Logistic scores: b · (a^T y)
        scores = b * (a @ updated_y)

        # Gradient of logistic loss: ∂/∂y Σ log(1 + exp(−scores))
        grad_logit = -b * np.exp(-np.logaddexp(0, scores))   # (n_samples,)
        grad_f     = (a.T @ grad_logit) / d                  # normalised

        # FedNova-HC: no dual correction term (grad_aug = 0)
        updated_y -= alpha * grad_f

    dual_residual = updated_y - y[i][j]

    # Normalised gradient accumulator: average parameter drift per local step
    s_ij[i][j] = (x[i] - updated_y) / E

    y[i][j] = updated_y

    # Evaluate local logistic loss at the final iterate (no augmented terms)
    scores   = b * (a @ y[i][j])
    losses   = np.logaddexp(0, -scores)
    loss_avg = losses.sum() / d
    F_total  = loss_avg   # lag_term = 0 in FedNova-HC

    return F_total, loss_avg, dual_residual


def analytical_solution_fog_i(w, x, y, i, mu_0, mu, rho):
    """
    Fog-level aggregation for FedNova-HC.

    Applies a Nova-style corrected update using the weighted sum of
    normalised gradient accumulators s_ij and the effective local step
    count τ_eff[i]:
        x_i ← x_i − τ_eff[i] · Σ_j ( p_ij · α · s_ij )

    This corrects for the objective inconsistency caused by heterogeneous
    numbers of local steps across edge devices.

    Args:
        w    (np.ndarray): Current global cloud model (unused here,
                           kept for interface consistency).
        x    (list[np.ndarray]): Fog-level parameters (updated in-place).
        y    (list[list[np.ndarray]]): Edge-level parameters.
        i    (int): Fog node index.
        mu_0 (list[np.ndarray]): Cloud-fog dual variables (unused in
                                 FedNova-HC, kept for interface consistency).
        mu   (list[list[np.ndarray]]): Fog-edge dual variables (unused in
                                       FedNova-HC, kept for interface consistency).
        rho  (float): Penalty parameter (unused in FedNova-HC,
                      kept for interface consistency).

    Returns:
        A_i (float): Placeholder aggregation value (0 in FedNova-HC).
        dual_residual_i (np.ndarray): x_new − x_old.
    """
    # Weighted sum of normalised gradient accumulators across edge devices
    grad_sum = sum(p_ij[i][j] * alpha * s_ij[i][j] for j in range(m_i))

    # Nova-corrected fog model update using effective local step count
    x_new           = x[i] - tau_eff[i] * grad_sum
    dual_residual_i = x_new - x[i]
    x[i]            = copy.deepcopy(x_new)

    # FedNova-HC: no augmented Lagrangian penalty terms
    A_i = 0.0

    return A_i, dual_residual_i


def analytical_solution_cloud():
    """
    Cloud-level aggregation for FedNova-HC.

    Computes a data-size-weighted average of all fog models:
        w ← Σ_i p_i · x_i

    Uses the global variables w, x, p_i, n (defined in __main__).

    Returns:
        A_0 (float): Placeholder aggregation value (0 in FedNova-HC).
        dual_residual (np.ndarray): w_new − w_old.
    """
    w_new         = sum(p_i[i] * x[i] for i in range(n))
    dual_residual = w_new - w
    w[:]          = copy.deepcopy(w_new)   # in-place update propagates to callers

    # FedNova-HC: no augmented Lagrangian penalty terms
    A_0 = 0.0

    return A_0, dual_residual


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Inference
# ──────────────────────────────────────────────────────────────────────────────

def predict(X, w):
    """
    Binary prediction using the sign of the linear decision function.

    Args:
        X (np.ndarray): Feature matrix with bias column, shape (n_samples, p).
        w (np.ndarray): Weight vector, shape (p,).

    Returns:
        np.ndarray: Predicted labels in {−1, +1}.
    """
    return np.sign(X @ w)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Main: FedNova-HC Training
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    # ── Topology ──────────────────────────────────────────────────────────────
    n   = 10            # number of fog nodes
    m   = 100          # total edge devices (must be divisible by n)
    m_i = m // n        # edge devices per fog node

    # ── Dataset dimensions ────────────────────────────────────────────────────
    d = X_train.shape[0]   # total training samples
    p = X_train.shape[1]   # feature dimension (including bias)

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

    # ── Data-size weights ─────────────────────────────────────────────────────
    # d_ij: number of samples on each edge device (fog_id, edge_id)
    d_ij = np.zeros((n, m_i), dtype=int)
    for (fog_id, edge_id), data in client_data.items():
        d_ij[fog_id, edge_id] = data["a"].shape[0]

    d_i  = d_ij.sum(axis=1)          # total samples per fog node
    p_i  = d_i / d                   # fog-level aggregation weights  (shape: n,)
    p_ij = (d_ij.T / d_i).T          # edge-level aggregation weights (shape: n × m_i)

    # ── Primal variable initialisation ────────────────────────────────────────
    w    = np.zeros(p)                                              # global model
    x    = [np.zeros(p) for _ in range(n)]                         # fog models
    y    = [[np.zeros(p) for _ in range(m_i)] for _ in range(n)]   # edge models
    s_ij = [[np.zeros(p) for _ in range(m_i)] for _ in range(n)]   # normalised gradient accumulators

    # ── Dual variable initialisation (unused in FedNova-HC, kept for consistency)
    mu_0 = [np.zeros(p) for _ in range(n)]                           # cloud-fog multipliers
    mu   = [[np.zeros(p) for _ in range(m_i)] for _ in range(n)]     # fog-edge multipliers

    # ── Convergence bookkeeping ───────────────────────────────────────────────
    dual_residual          = {}
    pri_residual           = {}
    F_values               = []
    avg_loss               = []
    avg_accuracies_history = []
    std_history            = []

    # ── Hyperparameters ───────────────────────────────────────────────────────
    eps_pri  = 1e-5    # primal residual convergence tolerance
    eps_dual = 1e-5    # dual residual convergence tolerance
    v_max    = 1       # inner iterations per outer communication round
    rho      = 1.0     # penalty parameter (unused in FedNova-HC, kept for consistency)
    alpha    = 0.001   # edge gradient-descent step size
    E        = 10     # local gradient steps per edge update
    max_iter = 301     # hard cap on total inner iterations

    # Effective local step count per fog node: τ_eff_i = Σ_j ( p_ij · E )
    # With uniform E across devices this reduces to E, but kept in general form.
    tau_eff = np.sum(p_ij * E, axis=1)   # shape: (n,)

    N_iter = 0         # global iteration counter

    # ── Logging setup ─────────────────────────────────────────────────────────
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    log_path = (
        f"./log/FedNova-HC_E{E}_rho{rho}_n{n}_m{m}_mi{m_i}"
        f"_vmax{v_max}_epspri{eps_pri}_epsdual{eps_dual}_alpha{alpha}.log"
    )
    fh = logging.FileHandler(log_path, "w")
    ch = logging.StreamHandler()

    fmt = logging.Formatter("%(message)s")
    fh.setFormatter(fmt);  fh.setLevel(logging.DEBUG)
    ch.setFormatter(fmt);  ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # ── FedNova-HC main loop ──────────────────────────────────────────────────
    for k in range(2001):
        v      = 0
        temp_1 = 0   # number of dual residuals that have converged

        # Inner loop: run up to v_max local update rounds before aggregation
        while v < v_max:
            N_iter += 1
            v      += 1
            F, loss = [], []

            # Step 1 – Edge update: local gradient descent from fog model x[i]
            #          s_ij is updated as a side effect inside the solver
            for i in range(n):
                for j in range(m_i):
                    F_ij, loss_ij, dual_residual[(i, j)] = \
                        logistic_grad_solver_edge_i_j(x, y, i, j, mu, d, alpha, rho)
                    F.append(F_ij)
                    loss.append(loss_ij)

            # Step 2 – Fog aggregation: Nova-corrected update using s_ij and τ_eff
            for i in range(n):
                F_i, dual_residual[i] = \
                    analytical_solution_fog_i(w, x, y, i, mu_0, mu, rho)
                F.append(F_i)

            # Step 3 – Cloud aggregation: data-size-weighted average of fog models
            F_0, dual_residual[-1] = analytical_solution_cloud()
            F.append(F_0)

            F_values.append(sum(F))
            avg_loss.append(sum(loss))

            # Evaluate per-device test accuracy and compute mean ± std
            acc = [
                np.mean(predict(X_test, y[i][j]) == y_test)
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
                f"F={F_values[-1]:.6f}  loss={avg_loss[-1]:.6f}  "
                f"acc={mean_acc:.4f}±{std_acc:.4f}"
            )

            # Hard stop if maximum iterations reached
            if N_iter == max_iter:
                break

            # Check dual convergence: count variables below eps_dual
            temp_1 = sum(
                1 for key in dual_residual
                if (np.abs(dual_residual[key]) <= eps_dual).all()
            )
            if temp_1 == len(dual_residual):
                logger.info(f"Dual converged: {temp_1}/{len(dual_residual)} variables below threshold.")
                break

        # ── Primal residual computation (no multiplier update in FedNova-HC) ──
        for i in range(n):
            for j in range(m_i):
                pri_residual[i, j] = x[i] - y[i][j]
                # mu[i][j] += rho * pri_residual[i, j]   # disabled in FedNova-HC

        for i in range(n):
            pri_residual[i] = w - x[i]
            # mu_0[i] += rho * pri_residual[i]           # disabled in FedNova-HC

        if N_iter == max_iter:
            break

        # Check primal convergence: count residuals below eps_pri
        temp_2 = sum(
            1 for key in pri_residual
            if (np.abs(pri_residual[key]) <= eps_pri).all()
        )
        if temp_1 + temp_2 == len(dual_residual) + len(pri_residual):
            logger.info(f"Full convergence: dual={temp_1}, primal={temp_2}.")
            break

    # ──────────────────────────────────────────────────────────────────────────
    # 6.  Save Results
    # ──────────────────────────────────────────────────────────────────────────

    # Normalised loss: relative distance from the final loss value
    normalized_loss = np.abs(np.array(avg_loss) - avg_loss[-1]) / avg_loss[-1]

    results_df = pd.DataFrame({
        "F_values":        F_values,
        "Avg_Loss":        avg_loss,
        "Normalized_Loss": normalized_loss,
        "Mean_Acc":        avg_accuracies_history,
        "Std_Acc":         std_history,
    })

    csv_path = (
        f"./csv/FedNova-HC_E{E}_rho{rho}_n{n}_m{m}_mi{m_i}"
        f"_vmax{v_max}_iter{N_iter}_epspri{eps_pri}_epsdual{eps_dual}_alpha{alpha}.csv"
    )
    results_df.to_csv(csv_path, index=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Fog nodes (n):              {n}")
    logger.info(f"Edge devices per fog (m_i): {m_i}")
    logger.info(f"Local steps (E):            {E}")
    logger.info(f"Step size (alpha):          {alpha}")
    logger.info(f"Total iterations:           {N_iter}")
    logger.info(f"v_max:                      {v_max}")
    logger.info(f"Mean test accuracy:         {mean_acc:.8f}")
    logger.info(f"Std  test accuracy:         {std_acc:.8f}")
    logger.info(f"Global model w:             {w}")