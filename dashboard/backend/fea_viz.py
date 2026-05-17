"""FEA visualization — run beam FEA and return base64 PNG stress/deflection colormaps."""
import sys, io, base64
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # aero-opt root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from skfem import *
from skfem.models.elasticity import linear_elasticity


def simulate_fea_viz(params: dict) -> dict:
    L   = float(params.get("L",    0.1))
    H   = float(params.get("H",    0.008))
    E   = float(params.get("E",    70e9))
    nu  = float(params.get("nu",   0.33))
    rho = float(params.get("rho",  2700.0))
    P   = float(params.get("load", 500.0))
    ny  = int(params.get("ny", 10))
    nx  = int(params.get("nx", max(int(L / H * ny * 0.5), 20)))

    mesh  = MeshTri.init_tensor(
        np.linspace(0, L, nx + 1),
        np.linspace(0, H, ny + 1),
    ).with_defaults()

    elem  = ElementVector(ElementTriP2())
    basis = Basis(mesh, elem)

    lam = E * nu / (1.0 - nu**2)
    mu  = E / (2.0 * (1.0 + nu))
    K   = asm(linear_elasticity(lam, mu), basis)

    fb = FacetBasis(mesh, elem, facets=mesh.boundaries["right"])

    @LinearForm
    def traction(v, w):
        return -(P / H) * v[1]

    f     = traction.assemble(fb)
    fixed = basis.get_dofs(mesh.boundaries["left"]).all()
    u     = solve(*condense(K, f, D=fixed))

    n_nodes       = mesh.p.shape[1]
    (ux, _), (uy, _) = basis.split(u)
    ux_n = ux[:n_nodes]
    uy_n = uy[:n_nodes]

    # Per-element von Mises (P1 constant-strain)
    x, y = mesh.p[0, mesh.t], mesh.p[1, mesh.t]
    uu, vv = ux_n[mesh.t], uy_n[mesh.t]
    A = 0.5 * np.abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    dN_dx = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]]) / (2*A)
    dN_dy = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]]) / (2*A)
    eps_xx = np.einsum("ij,ij->j", dN_dx, uu)
    eps_yy = np.einsum("ij,ij->j", dN_dy, vv)
    gam_xy = np.einsum("ij,ij->j", dN_dy, uu) + np.einsum("ij,ij->j", dN_dx, vv)
    sig_xx = (lam + 2*mu)*eps_xx + lam*eps_yy
    sig_yy = lam*eps_xx + (lam + 2*mu)*eps_yy
    tau_xy = mu*gam_xy
    vm_elem = np.sqrt(sig_xx**2 - sig_xx*sig_yy + sig_yy**2 + 3*tau_xy**2)

    # Average per-element stress to nodes
    vm_nodes = np.zeros(n_nodes)
    cnt      = np.zeros(n_nodes)
    for i in range(3):
        np.add.at(vm_nodes, mesh.t[i], vm_elem)
        np.add.at(cnt,      mesh.t[i], 1)
    vm_nodes /= cnt

    # Build matplotlib triangulation (coords in mm)
    xn = mesh.p[0] * 1e3
    yn = mesh.p[1] * 1e3
    triang = mtri.Triangulation(xn, yn, mesh.t.T)

    BG = "#0a0f1e"
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), facecolor=BG)

    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#334155")

    def styled_colorbar(fig, ax, tc, label):
        cb = fig.colorbar(tc, ax=ax, pad=0.02, fraction=0.035)
        cb.set_label(label, color="#94a3b8", fontsize=9)
        cb.ax.yaxis.set_tick_params(color="#94a3b8", labelsize=8)
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#94a3b8")

    # ── Stress panel ──
    tc1 = axes[0].tripcolor(triang, vm_nodes / 1e6, cmap="plasma", shading="gouraud")
    styled_colorbar(fig, axes[0], tc1, "von Mises (MPa)")
    axes[0].set_title("Von Mises Stress", color="#f1f5f9", fontsize=11, fontweight="bold", pad=6)
    axes[0].set_xlabel("x (mm)", color="#94a3b8", fontsize=9)
    axes[0].set_ylabel("y (mm)", color="#94a3b8", fontsize=9)
    axes[0].set_aspect("equal")

    # ── Deflection panel ──
    tc2 = axes[1].tripcolor(triang, np.abs(uy_n) * 1e3, cmap="viridis", shading="gouraud")
    styled_colorbar(fig, axes[1], tc2, "|Deflection| (mm)")
    axes[1].set_title("Vertical Displacement", color="#f1f5f9", fontsize=11, fontweight="bold", pad=6)
    axes[1].set_xlabel("x (mm)", color="#94a3b8", fontsize=9)
    axes[1].set_ylabel("y (mm)", color="#94a3b8", fontsize=9)
    axes[1].set_aspect("equal")

    plt.tight_layout(pad=1.8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)

    return {
        "plot_b64":          base64.b64encode(buf.read()).decode(),
        "max_deflection_mm": float(np.max(np.abs(uy_n)) * 1e3),
        "max_von_mises_mpa": float(np.max(vm_elem) / 1e6),
        "mass_per_depth":    float(rho * L * H),
    }
