"""Tensor-Basis Neural Network (Ling, Kurzawski & Templeton, JFM 2016).

Predicts the Reynolds-stress anisotropy tensor b_ij as a linear combination of
an integrity basis of tensors T^(n) built from the (normalised) mean strain and
rotation rates, with scalar coefficients g^(n) that are functions of the tensor
invariants and are learned by a small MLP:

    b = sum_n  g^(n)(lambda_1..lambda_5) * T^(n).

Because the basis tensors and invariants are Galilean- and rotation-invariant by
construction, the predicted anisotropy transforms correctly under rotation of
the coordinate frame — an inductive bias a generic regressor lacks. This is the
recognised state-of-the-art structure for data-driven Reynolds-stress closures.

Here we use the leading 4 basis tensors and 2 invariants (sufficient for this
statistically 2-D flow; the higher tensors contribute negligibly and only add
noise). numpy for the tensor algebra, PyTorch for the coefficient network.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAVE_TORCH = True
except Exception:                                # pragma: no cover
    _HAVE_TORCH = False


def normalised_S_R(gradU: np.ndarray, omega: np.ndarray, Cmu: float = 0.09):
    """Non-dimensional strain (S) and rotation (R) tensors.

    Normalised by the turbulence time scale tau = 1/(Cmu*omega) (= k/eps), the
    standard TBNN scaling, so S, R are dimensionless and O(1).
    """
    g = np.asarray(gradU)
    if g.ndim == 2:
        g = g.reshape(-1, 3, 3)
    S = 0.5 * (g + np.transpose(g, (0, 2, 1)))
    W = 0.5 * (g - np.transpose(g, (0, 2, 1)))
    tau = 1.0 / (Cmu * np.maximum(np.asarray(omega), 1e-12))
    return S * tau[:, None, None], W * tau[:, None, None]


def _trace(A):
    return np.einsum("nii->n", A)


def invariants(S, R):
    """The 5 scalar invariants of the (S, R) system (Pope 1975):
    lambda1=tr(S^2), lambda2=tr(R^2), lambda3=tr(S^3),
    lambda4=tr(R^2 S), lambda5=tr(R^2 S^2).
    """
    S2 = np.einsum("nij,njk->nik", S, S)
    S3 = np.einsum("nij,njk->nik", S2, S)
    R2 = np.einsum("nij,njk->nik", R, R)
    R2S = np.einsum("nij,njk->nik", R2, S)
    R2S2 = np.einsum("nij,njk->nik", R2, S2)
    return np.stack([_trace(S2), _trace(R2), _trace(S3),
                     _trace(R2S), _trace(R2S2)], axis=1)


def normalise_basis(basis, scale=None):
    """Scale each basis tensor by its mean Frobenius norm (Ling et al.).

    Keeps the learned coefficients g ~ O(b) so training is well-conditioned.
    Pass the training ``scale`` back in for the test set.
    """
    if scale is None:
        scale = np.sqrt((basis**2).sum(axis=(2, 3))).mean(axis=0) + 1e-12
    return basis / scale[None, :, None, None], scale


def tensor_basis(S, R):
    """Leading 4 integrity-basis tensors (each traceless symmetric), (N,4,3,3).

    T1 = S
    T2 = SR - RS
    T3 = S^2 - (1/3) tr(S^2) I
    T4 = R^2 - (1/3) tr(R^2) I
    """
    n = S.shape[0]
    I = np.eye(3)[None]
    SR = np.einsum("nij,njk->nik", S, R)
    RS = np.einsum("nij,njk->nik", R, S)
    S2 = np.einsum("nij,njk->nik", S, S)
    R2 = np.einsum("nij,njk->nik", R, R)
    T = np.empty((n, 4, 3, 3))
    T[:, 0] = S
    T[:, 1] = SR - RS
    T[:, 2] = S2 - (1.0 / 3.0) * _trace(S2)[:, None, None] * I
    T[:, 3] = R2 - (1.0 / 3.0) * _trace(R2)[:, None, None] * I
    return T


if _HAVE_TORCH:

    class TBNN(nn.Module):
        """MLP: invariants -> basis coefficients g; output b = sum g_n T_n."""

        def __init__(self, n_inv=2, n_basis=4, hidden=(32, 32)):
            super().__init__()
            layers, d = [], n_inv
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU()]
                d = h
            layers += [nn.Linear(d, n_basis)]
            self.net = nn.Sequential(*layers)

        def forward(self, inv, basis):
            g = self.net(inv)                          # (N, n_basis)
            # b = sum_n g_n T_n  -> (N, 3, 3)
            return torch.einsum("nk,nkij->nij", g, basis)

    def train_tbnn(inv_tr, basis_tr, b_tr, *, epochs=60, lr=2e-3, batch=4096,
                   hidden=(64, 64), seed=0, verbose=False):
        """Fit a TBNN with mini-batch Adam. Returns the model + input scaler."""
        torch.manual_seed(seed)
        mu, sd = inv_tr.mean(0), inv_tr.std(0) + 1e-8
        Xi = torch.tensor((inv_tr - mu) / sd, dtype=torch.float32)
        Tb = torch.tensor(basis_tr, dtype=torch.float32)
        yb = torch.tensor(b_tr, dtype=torch.float32)
        n = Xi.shape[0]
        model = TBNN(n_inv=inv_tr.shape[1], n_basis=basis_tr.shape[1], hidden=hidden)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        lossf = nn.MSELoss()
        gen = torch.Generator().manual_seed(seed)
        for ep in range(epochs):
            perm = torch.randperm(n, generator=gen)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                opt.zero_grad()
                loss = lossf(model(Xi[idx], Tb[idx]), yb[idx])
                loss.backward()
                opt.step()
            sched.step()
            if verbose and ep % 10 == 0:
                with torch.no_grad():
                    full = lossf(model(Xi, Tb), yb).item()
                print(f"    epoch {ep}: loss {full:.3e}")
        return model, (mu, sd)

    def predict_tbnn(model, scaler, inv, basis):
        mu, sd = scaler
        with torch.no_grad():
            b = model(torch.tensor((inv - mu) / sd, dtype=torch.float32),
                      torch.tensor(basis, dtype=torch.float32))
        return b.numpy()
