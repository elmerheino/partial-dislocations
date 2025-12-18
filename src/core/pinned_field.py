"""
pinned_field.py

Generate a 2D Gaussian random field tau(x, y) on a box of size
(Nx, 2*Nx) with grid spacing dx = dy = 1 and correlations:

    Psi(x, y) = 1/16 [Phi_L(x) + 3 Phi_T(x)]
                         [3 Phi_L(y) + Phi_T(y)]

where (following Vaid 2022, Geslin 2021):

    Phi_L(u) = -30/u^3 [ sqrt(pi)*(1 - 12/u^2)*erf(u/2)
                         + (12 + u^2)/u * exp(-u^2/4) ]

    Phi_T(u) =  15/u^3 [ sqrt(pi)*(1 -  6/u^2)*erf(u/2)
                         + 6/u * exp(-u^2/4) ]

with u = |r| / a_phi and default a_phi = 1.
"""

import numpy as np
from scipy.special import erf


# ============================================================================
# 1D correlation kernels Phi_L(u), Phi_T(u)
# ============================================================================

def phi_L(u):
    """Longitudinal correlation Phi_L(u)."""
    u = np.asarray(u, float)
    out = np.zeros_like(u)
    eps = 1e-8
    mask = np.abs(u) > eps
    um = u[mask]
    um2 = um**2

    pref = -30.0 / (um**3)
    term1 = np.sqrt(np.pi) * (1.0 - 12.0/um2) * erf(um/2.0)
    term2 = (12.0 + um2)/um * np.exp(-um2/4.0)
    out[mask] = pref * (term1 + term2)

    out[~mask] = 1.0  # normalization at u=0
    return out


def phi_T(u):
    """Transverse correlation Phi_T(u)."""
    u = np.asarray(u, float)
    out = np.zeros_like(u)
    eps = 1e-8
    mask = np.abs(u) > eps
    um = u[mask]
    um2 = um**2

    pref = 15.0 / (um**3)
    term1 = np.sqrt(np.pi) * (1.0 - 6.0/um2) * erf(um/2.0)
    term2 = (6.0/um) * np.exp(-um2/4.0)
    out[mask] = pref * (term1 + term2)

    out[~mask] = 1.0
    return out


# ============================================================================
# 1D factors Psi_x and Psi_y
# ============================================================================

def psi_x_1d(x, a_phi=1.0):
    """Psi_x(dx) = 1/4 [ Phi_L(u) + 3 Phi_T(u) ]."""
    dx_abs = np.abs(np.asarray(x, float))
    u = dx_abs / a_phi
    return 0.25 * (phi_L(u) + 3.0 * phi_T(u))


def psi_y_1d(y, a_phi=1.0):
    """Psi_y(dy) = 1/4 [ 3 Phi_L(u) + Phi_T(u) ]."""
    dy_abs = np.abs(np.asarray(y, float))
    u = dy_abs / a_phi
    return 0.25 * (3.0 * phi_L(u) + phi_T(u))


# ============================================================================
# Build correlation grid for a given Nx
# ============================================================================

def build_correlation_grid(Nx, Ny=None, a_phi=1.0):
    """
    Build 2D Psi(x,y) on a grid of shape (Ny, Nx),
    where Nx is user-specified and Ny = 2*Nx by default.

    Box size is (Nx, 2*Nx) with dx=dy=1.
    """
    if Ny is None:
        Ny = 2 * Nx

    # Coordinates centered around 0:
    # x from -Nx/2 ... Nx/2
    # y from -Ny/2 ... Ny/2
    x = np.arange(Nx) - Nx//2
    y = np.arange(Ny) - Ny//2

    psi_x = psi_x_1d(x, a_phi)
    psi_y = psi_y_1d(y, a_phi)

    Psi_xy = np.outer(psi_y, psi_x)

    # Normalization at origin
    Psi_xy[Ny//2, Nx//2] = 1.0

    # Move origin to (0,0) for FFT use
    Psi_xy = np.fft.ifftshift(Psi_xy)
    return Psi_xy


# ============================================================================
# Generate Gaussian field with desired correlation
# ============================================================================

def generate_random_field(Nx,
                          Ny=None,
                          a_phi=1.0,
                          field_rms=1.0,
                          random_state=None):
    """
    Generate a correlated Gaussian random field tau(x,y)
    on a domain (Nx, 2*Nx) with dx=dy=1.

    Parameters
    ----------
    Nx : int
        Grid size in x direction.
    Ny : int or None
        Grid size in y. Default Ny = 2*Nx.
    a_phi : float
        Regularization length in Phi_L, Phi_T. Default 1.
    field_rms : float
        Desired RMS of tau.
    random_state : int or np.random.Generator or None

    Returns
    -------
    X, Y : np.ndarray
        Coordinates of shape (Ny, Nx).
    tau : np.ndarray
        Correlated random field with std ≈ field_rms.
    """
    if Ny is None:
        Ny = 2 * Nx

    Psi_xy = build_correlation_grid(Nx=Nx, Ny=Ny, a_phi=a_phi)

    # Power spectrum
    S_k = np.fft.fftn(Psi_xy).real
    S_k = np.maximum(S_k, 0.0)

    # RNG
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    w = rng.normal(0.0, 1.0, size=(Ny, Nx))
    W_k = np.fft.fftn(w)

    Tau_k = W_k * np.sqrt(S_k + 1e-15)
    tau = np.fft.ifftn(Tau_k).real

    # Normalize
    std = tau.std()
    if std > 0:
        tau *= field_rms / std

    # Real-space coordinate grids (centered)
    x = np.arange(Nx) - Nx//2
    y = np.arange(Ny) - Ny//2
    X, Y = np.meshgrid(x, y, indexing="xy")

    return X, Y, tau


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    Nx = 256
    X, Y, tau = generate_random_field(Nx, field_rms=50.0, random_state=0)

    print("Field shape:", tau.shape)
    print("RMS =", tau.std())

