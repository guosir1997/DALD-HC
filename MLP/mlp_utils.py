"""
mlp_utils.py
============
MLP utility functions for the DALD-HC framework.

Provides per-edge-device (i, j):
    - Forward pass
    - Backward pass (gradient computation via backpropagation)
    - One-step parameter update with augmented Lagrangian terms
    - Augmented Lagrangian value computation

Parameter layout (all lists of length L, indexed l = 0, 1, ..., L−1):
    y_w : list[np.ndarray]   weight matrices,  y_w[l].shape = (M_{l+1}, M_l)
    y_b : list[np.ndarray]   bias vectors,     y_b[l].shape = (M_{l+1},)

Notation mapping to paper symbols (paper uses 1-based layer index):
    y_ij^{l}  →  y_w[l−1]
    e_ij^{l}  →  y_b[l−1]
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Activation Functions
# ──────────────────────────────────────────────────────────────────────────────

def sigmoid(z):
    """Sigmoid activation: σ(z) = 1 / (1 + exp(−z)), clipped for stability."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_deriv(z):
    """Derivative of sigmoid: σ'(z) = σ(z) · (1 − σ(z))."""
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z):
    """ReLU activation: max(0, z)."""
    return np.maximum(0.0, z)


def relu_deriv(z):
    """Derivative of ReLU: 1 if z > 0, else 0."""
    return (z > 0).astype(float)


def linear(z):
    """Linear (identity) activation: f(z) = z."""
    return z


def linear_deriv(z):
    """Derivative of linear activation: f'(z) = 1."""
    return np.ones_like(z)


def get_activation(l, L):
    """
    Return the activation function and its derivative for layer l.

    Convention:
        Hidden layers (l < L−1): ReLU
        Output layer  (l = L−1): Sigmoid  (binary classification)

    Args:
        l (int): 0-based layer index (0 → first hidden, L−1 → output).
        L (int): Total number of layers.

    Returns:
        tuple: (activation_fn, activation_derivative_fn)
    """
    if l < L - 1:
        return relu, relu_deriv       # hidden layers
    else:
        return sigmoid, sigmoid_deriv  # output layer


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Forward Pass
# ──────────────────────────────────────────────────────────────────────────────

def forward(a, y_w, y_b, L):
    """
    Compute the forward pass for a single input sample.

    For each layer l = 0, ..., L−1:
        z^{l+1} = y_w[l] @ c^l + y_b[l]
        c^{l+1} = activation(z^{l+1})

    Args:
        a   (np.ndarray): Input feature vector, shape (M_0,).
        y_w (list[np.ndarray]): Weight matrices; y_w[l].shape = (M_{l+1}, M_l).
        y_b (list[np.ndarray]): Bias vectors;   y_b[l].shape = (M_{l+1},).
        L   (int): Number of network layers.

    Returns:
        z_list  (list[np.ndarray]): Pre-activation values z^1, ..., z^L.
        c_list  (list[np.ndarray]): Post-activation values c^1, ..., c^L;
                                    c_list[-1] is the network output ŷ ∈ (0, 1).
        c_input (np.ndarray): Copy of the input a (= c^0), used in backprop.
    """
    c       = a.copy()      # c^0 = input features
    c_list  = []            # c^1, ..., c^L
    z_list  = []            # z^1, ..., z^L
    c_input = c.copy()      # save c^0 for gradient computation

    for l in range(L):
        z = y_w[l] @ c + y_b[l]      # pre-activation, shape (M_{l+1},)
        act, _ = get_activation(l, L)
        c = act(z)                     # post-activation, shape (M_{l+1},)
        z_list.append(z)
        c_list.append(c)

    return z_list, c_list, c_input


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Loss and Output Gradient
# ──────────────────────────────────────────────────────────────────────────────

def loss_grad_output(b_hat, b):
    """
    Gradient of binary cross-entropy loss w.r.t. the network output ŷ.

    Loss:    ℓ(ŷ, b) = −b·log(ŷ) − (1−b)·log(1−ŷ)
    Gradient: ∂ℓ/∂ŷ  = −b/ŷ + (1−b)/(1−ŷ)

    Args:
        b_hat (float): Network output after sigmoid, clipped to (0, 1).
        b     (float): True label (0 or 1).

    Returns:
        float: ∂ℓ/∂ŷ
    """
    eps   = 1e-12
    b_hat = np.clip(b_hat, eps, 1 - eps)
    return -b / b_hat + (1 - b) / (1 - b_hat)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Backward Pass (Batch Gradient Computation)
# ──────────────────────────────────────────────────────────────────────────────

def backward(X_batch, b_batch, y_w, y_b, L, d_total):
    """
    Compute the gradients of the local loss f_ij w.r.t. all weights and biases
    via backpropagation, accumulated over the entire local mini-batch.

    Gradients are normalised by d_total (global sample count) to match the
    paper's (1/d) convention.

    Algorithm (per sample):
        1. Forward pass to obtain z_list, c_list.
        2. Output-layer error term:
               δ^L = (∂ℓ/∂c^L) ⊙ σ'^L(z^L)
        3. Accumulate weight / bias gradients:
               ∂f/∂y_w[l] += δ^{l+1} (c^l)^T
               ∂f/∂y_b[l] += δ^{l+1}
        4. Propagate error term backwards:
               δ^l = (y_w[l])^T δ^{l+1} ⊙ σ'^l(z^l)

    Args:
        X_batch  (np.ndarray): Local feature batch,  shape (n_samples, M_0).
        b_batch  (np.ndarray): Local label batch,    shape (n_samples,).  Labels ∈ {0, 1}.
        y_w      (list[np.ndarray]): Current weight matrices.
        y_b      (list[np.ndarray]): Current bias vectors.
        L        (int): Number of network layers.
        d_total  (int): Global total sample count for (1/d) normalisation.

    Returns:
        grad_w (list[np.ndarray]): ∂f_ij/∂y_w[l], same shapes as y_w.
        grad_b (list[np.ndarray]): ∂f_ij/∂y_b[l], same shapes as y_b.
    """
    n_samples = X_batch.shape[0]

    # Initialise accumulated gradients to zero
    grad_w = [np.zeros_like(w) for w in y_w]
    grad_b = [np.zeros_like(b) for b in y_b]

    for idx in range(n_samples):
        a = X_batch[idx]   # (M_0,)
        b = b_batch[idx]   # scalar label

        # Forward pass
        z_list, c_list, c_input = forward(a, y_w, y_b, L)

        # Output-layer error term δ^L
        b_hat  = c_list[-1][0]                             # scalar sigmoid output
        df_dc  = np.array([loss_grad_output(b_hat, b)])    # shape (1,)
        _, act_d = get_activation(L - 1, L)
        delta  = df_dc * act_d(z_list[-1])                 # shape (M_L,) = (1,)

        # Backpropagate through all layers
        for l in range(L - 1, -1, -1):
            c_prev = c_list[l - 1] if l > 0 else c_input  # c^{l-1}, shape (M_l,)

            # Accumulate weight and bias gradients
            grad_w[l] += np.outer(delta, c_prev)   # (M_{l+1}, M_l)
            grad_b[l] += delta                     # (M_{l+1},)

            # Propagate error term to the previous layer
            if l > 0:
                _, act_d = get_activation(l - 1, L)
                delta = (y_w[l].T @ delta) * act_d(z_list[l - 1])

    # Normalise by global sample count (1/d in paper notation)
    grad_w = [g / d_total for g in grad_w]
    grad_b = [g / d_total for g in grad_b]

    return grad_w, grad_b


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Edge Parameter Update (One Gradient Step with Augmented Terms)
# ──────────────────────────────────────────────────────────────────────────────

def edge_param_update(y_w, y_b,
                      grad_w, grad_b,
                      x_w, x_b,
                      mu_w, mu_b,
                      alpha, rho, L):
    """
    Perform one gradient-descent step on the edge augmented Lagrangian.

    Update rule per layer l:
        y_w[l] ← y_w[l] − α · [ ∂f/∂y_w[l] − μ_w[l] − ρ·(x_w[l] − y_w[l]) ]
        y_b[l] ← y_b[l] − α · [ ∂f/∂y_b[l] − μ_b[l] − ρ·(x_b[l] − y_b[l]) ]

    The terms −μ[l] − ρ·(x[l] − y[l]) are the gradients of the augmented
    Lagrangian penalty w.r.t. y[l], enforcing consensus with the fog model x[l].

    Args:
        y_w   (list[np.ndarray]): Current edge weight matrices.
        y_b   (list[np.ndarray]): Current edge bias vectors.
        grad_w(list[np.ndarray]): Loss gradients ∂f/∂y_w[l].
        grad_b(list[np.ndarray]): Loss gradients ∂f/∂y_b[l].
        x_w   (list[np.ndarray]): Fog weight matrices (held fixed as consensus anchor).
        x_b   (list[np.ndarray]): Fog bias vectors   (held fixed as consensus anchor).
        mu_w  (list[np.ndarray]): Dual variables for weights (fog-edge).
        mu_b  (list[np.ndarray]): Dual variables for biases  (fog-edge).
        alpha (float): Gradient-descent step size.
        rho   (float): Penalty parameter.
        L     (int): Number of network layers.

    Returns:
        new_y_w (list[np.ndarray]): Updated edge weight matrices.
        new_y_b (list[np.ndarray]): Updated edge bias vectors.
    """
    new_y_w = []
    new_y_b = []

    for l in range(L):
        # Weight update: gradient of augmented Lagrangian w.r.t. y_w[l]
        g_w = grad_w[l] - mu_w[l] - rho * (x_w[l] - y_w[l])
        new_y_w.append(y_w[l] - alpha * g_w)

        # Bias update: gradient of augmented Lagrangian w.r.t. y_b[l]
        g_b = grad_b[l] - mu_b[l] - rho * (x_b[l] - y_b[l])
        new_y_b.append(y_b[l] - alpha * g_b)

    return new_y_w, new_y_b


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Augmented Lagrangian Value (Edge Node)
# ──────────────────────────────────────────────────────────────────────────────

def compute_augmented_lagrangian(X_batch, b_batch,
                                  y_w, y_b,
                                  x_w, x_b,
                                  mu_w, mu_b,
                                  rho, L, d_total):
    """
    Compute the local augmented Lagrangian value for edge device (i, j).

    F_total = f_ij(y_w, y_b)
              + Σ_l [ μ_w[l] ⊙ (x_w[l] − y_w[l]) + (ρ/2)‖x_w[l] − y_w[l]‖² ]
              + Σ_l [ μ_b[l] · (x_b[l] − y_b[l]) + (ρ/2)‖x_b[l] − y_b[l]‖² ]

    where f_ij is the average binary cross-entropy loss over the local batch,
    normalised by d_total.

    Args:
        X_batch (np.ndarray): Local feature batch,  shape (n_samples, M_0).
        b_batch (np.ndarray): Local label batch,    shape (n_samples,). Labels ∈ {0, 1}.
        y_w     (list[np.ndarray]): Current edge weight matrices.
        y_b     (list[np.ndarray]): Current edge bias vectors.
        x_w     (list[np.ndarray]): Fog weight matrices (consensus anchor).
        x_b     (list[np.ndarray]): Fog bias vectors   (consensus anchor).
        mu_w    (list[np.ndarray]): Dual variables for weights.
        mu_b    (list[np.ndarray]): Dual variables for biases.
        rho     (float): Penalty parameter.
        L       (int): Number of network layers.
        d_total (int): Global total sample count for (1/d) normalisation.

    Returns:
        F_total  (float): Augmented Lagrangian value f_ij + penalty terms.
        loss_avg (float): Normalised cross-entropy loss f_ij(y_w, y_b) only.
    """
    eps        = 1e-12
    total_loss = 0.0

    for idx in range(X_batch.shape[0]):
        _, c_list, _ = forward(X_batch[idx], y_w, y_b, L)
        b_hat       = np.clip(c_list[-1][0], eps, 1 - eps)
        b           = b_batch[idx]
        total_loss += -b * np.log(b_hat) - (1 - b) * np.log(1 - b_hat)

    loss_avg = total_loss / d_total

    # Augmented Lagrangian penalty terms (weights + biases, all layers)
    aug = 0.0
    for l in range(L):
        diff_w = x_w[l] - y_w[l]
        diff_b = x_b[l] - y_b[l]
        aug += (np.sum(mu_w[l] * diff_w) + 0.5 * rho * np.sum(diff_w ** 2)
                + np.dot(mu_b[l], diff_b) + 0.5 * rho * np.sum(diff_b ** 2))

    return loss_avg + aug, loss_avg