"""
visualization.py — Módulo 3: Visualización y análisis de resultados
==================================================================
Funciones para graficar pistas, trayectorias optimizadas, perfiles de velocidad
y tablas comparativas de resultados. Estilo visual consistente y de alta calidad
para presentaciones y análisis detallados.
Funciones principales:
- plot_track_result: Gráfica detallada de la pista con raceline optimizada.
- plot_velocity_comparison: Perfil de velocidad comparativo entre centro y raceline.
- plot_all_tracks: Vista general de todas las pistas cargadas.
- plot_results_table: Tabla resumen de resultados como figura.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from track_model import (reconstruct_trajectory, lap_time, velocity_profile,
                         track_length)


# -----------------------------------------------------------------------------
# PALETA DE COLORES
# -----------------------------------------------------------------------------
_BG       = "#ffffff"
_TRACK    = "#3a3a5c"
_BORDER_E = "#0a0a0a"
_BORDER_I = "#0a0a0a"
_CENTER   = "#B7B400"
_RACING   = "#ff260065"
_ACCENT   = "#00d4ff"
_TEXT     = "#e0e0e0"
_GOLD     = "#ffd700"

plt.rcParams.update({
    "figure.facecolor": _BG,
    "axes.facecolor":   _BG,
    "text.color":       _TEXT,
    "axes.labelcolor":  _TEXT,
    "xtick.color":      _TEXT,
    "ytick.color":      _TEXT,
    "font.family":      "sans-serif",
    "font.size":        11,
})

def _close_loop(x, y):
    return np.append(x, x[0]), np.append(y, y[0])

# -----------------------------------------------------------------------------
# GRÁFICA DE PISTA CON RACELINE OPTIMIZADA
# -----------------------------------------------------------------------------
def plot_track_result(geo, ctrl, u_opt, track_name, t_center, t_opt,
                      save_path=None):
    """
    Publication-quality track plot with:
      - Track surface (filled between borders)
      - Exterior / interior borders
      - Centerline (dashed)
      - Optimized raceline (bold red)
      - Time comparison box
    """
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)

    # -- Track surface --------------------------------------------------------
    ex, ey = _close_loop(geo["ext_x"], geo["ext_y"])
    ix, iy = _close_loop(geo["int_x"], geo["int_y"])

    ax.fill(ex, ey, color=_TRACK, alpha=0.55, zorder=1)
    ax.fill(ix, iy, color=_BG,    zorder=2)

    # -- Borders --------------------------------------------------------------
    ax.plot(ex, ey, color=_BORDER_E, lw=1.2, alpha=0.8, zorder=3,
            label="Track borders")
    ax.plot(ix, iy, color=_BORDER_I, lw=1.2, alpha=0.8, zorder=3)

    # -- Centerline -----------------------------------------------------------
    ccx, ccy = _close_loop(geo["cx"], geo["cy"])
    ax.plot(ccx, ccy, color=_CENTER, lw=1.4, ls="--", alpha=0.65, zorder=4,
            label=f"Centerline  ({t_center:.3f} s)")

    # -- Optimized raceline ---------------------------------------------------
    tx, ty = reconstruct_trajectory(ctrl, u_opt)
    tx_c, ty_c = _close_loop(tx, ty)
    ax.plot(tx_c, ty_c, color=_RACING, lw=2.4, alpha=0.95, zorder=5,
            path_effects=[pe.Stroke(linewidth=4.0, foreground="#000000", alpha=0.35),
                          pe.Normal()],
            label=f"Optimized   ({t_opt:.3f} s)")

    # -- Start / finish marker ------------------------------------------------
    ax.scatter(geo["cx"][0], geo["cy"][0], s=120, c=_GOLD, marker="D",
               edgecolors="black", linewidths=0.8, zorder=7,
               label="Start / Finish")

    # -- Time comparison box --------------------------------------------------
    mejora = (t_center - t_opt) / t_center * 100
    km = track_length(geo)
    box_text = (
        f"Circuit length: {km:.2f} km\n"
        f"-------------------------\n"
        f"Centerline time:   {t_center:.3f} s\n"
        f"Optimized time:    {t_opt:.3f} s\n"
        f"-------------------------\n"
        f"Improvement:  {mejora:.2f} %\n"
        f"Time saved:   {t_center - t_opt:.3f} s"
    )
    props = dict(boxstyle="round,pad=0.6", facecolor="#0d0d1a",
                 edgecolor=_ACCENT, alpha=0.92, linewidth=1.5)
    ax.text(0.02, 0.98, box_text, transform=ax.transAxes, fontsize=10.5,
            verticalalignment="top", fontfamily="monospace",
            bbox=props, zorder=10, color=_TEXT)

    # -- Legend ---------------------------------------------------------------
    leg = ax.legend(loc="lower right", fontsize=10, framealpha=0.85,
                    facecolor="#0d0d1a", edgecolor="#444466",
                    labelcolor=_TEXT)
    leg.get_frame().set_linewidth(1.2)

    # -- Axes -----------------------------------------------------------------
    ax.set_aspect("equal")
    ax.set_title(f"{track_name} — GA-Optimized Racing Line",
                 fontsize=16, fontweight="bold", color="white", pad=14)
    ax.set_xlabel("x  (m)", fontsize=12)
    ax.set_ylabel("y  (m)", fontsize=12)
    ax.grid(True, alpha=0.12, color="#ffffff")
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved → {save_path}")
    plt.close(fig)

# -----------------------------------------------------------------------------
# PERFIL DE VELOCIDAD COMPARATIVO
# -----------------------------------------------------------------------------

def plot_velocity_comparison(geo, ctrl, u_opt, track_name, save_path=None):
    """Velocity profile: centerline vs optimized."""
    # Centerline
    cx_c, cy_c = _close_loop(geo["cx"], geo["cy"])
    d_c, v_c = velocity_profile(cx_c, cy_c)

    # Optimized
    tx, ty = reconstruct_trajectory(ctrl, u_opt)
    tx_c, ty_c = _close_loop(tx, ty)
    d_o, v_o = velocity_profile(tx_c, ty_c)

    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=150)
    ax.fill_between(d_o / 1000, 0, v_o, color=_RACING, alpha=0.18, zorder=2)
    ax.plot(d_c / 1000, v_c, color=_CENTER, lw=1.0, alpha=0.6, zorder=3,
            label="Centerline")
    ax.plot(d_o / 1000, v_o, color=_RACING, lw=1.6, alpha=0.9, zorder=4,
            label="Optimized raceline")

    ax.set_xlabel("Distance  (km)", fontsize=12)
    ax.set_ylabel("Speed  (km/h)", fontsize=12)
    ax.set_title(f"{track_name} — Velocity Profile", fontsize=14,
                 fontweight="bold", color="white", pad=10)
    ax.set_xlim(0, d_o[-1] / 1000)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.15, color="#ffffff")
    leg = ax.legend(loc="upper right", fontsize=10, framealpha=0.85,
                    facecolor="#0d0d1a", edgecolor="#444466", labelcolor=_TEXT)
    leg.get_frame().set_linewidth(1.0)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved → {save_path}")
    plt.close(fig)

# -----------------------------------------------------------------------------
# VISTA GENERAL — 5 PISTAS EN GRID
# -----------------------------------------------------------------------------

def plot_all_tracks(all_geo, title="F1 Circuits", save_path=None):
    """Overview of all loaded circuits."""
    n = len(all_geo)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows), dpi=140)
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for i, (name, geo) in enumerate(all_geo.items()):
        ax = axes[i]
        ex, ey = _close_loop(geo["ext_x"], geo["ext_y"])
        ix, iy = _close_loop(geo["int_x"], geo["int_y"])
        ax.fill(ex, ey, color=_TRACK, alpha=0.5)
        ax.fill(ix, iy, color=_BG)
        ax.plot(ex, ey, color=_BORDER_E, lw=0.9, alpha=0.7)
        ax.plot(ix, iy, color=_BORDER_I, lw=0.9, alpha=0.7)
        ccx, ccy = _close_loop(geo["cx"], geo["cy"])
        ax.plot(ccx, ccy, color=_CENTER, lw=0.8, ls="--", alpha=0.5)
        km = track_length(geo)
        ax.set_title(f"{name}  ({km:.2f} km)", fontsize=13,
                     fontweight="bold", color="white")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.08, color="#ffffff")
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=17, fontweight="bold", color="white", y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved → {save_path}")
    plt.close(fig)

# -----------------------------------------------------------------------------
# TABLA COMPARATIVA DE RESULTADOS
# -----------------------------------------------------------------------------

def plot_results_table(all_results, save_path=None):
    """Summary table as a figure."""
    rows_data = []
    for name, r in all_results.items():
        mejora = (r["t_center"] - r["best"]) / r["t_center"] * 100
        rows_data.append([
            name,
            f'{r["t_center"]:.3f}',
            f'{r["best"]:.3f}',
            f'{r["mean"]:.3f}',
            f'{r["std"]:.4f}',
            f'{mejora:.2f}%',
        ])

    cols = ["Circuito", "Centro (s)", "Best (s)", "Mean (s)", "Std (s)", "Improv."]

    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.5 * len(rows_data)), dpi=150)
    ax.axis("off")

    table = ax.table(cellText=rows_data, colLabels=cols, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#444466")
        if r == 0:
            cell.set_facecolor("#1e3a5f")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#0d0d1a" if r % 2 == 0 else "#16162b")
            cell.set_text_props(color=_TEXT)

    ax.set_title("Optimization Results", fontsize=14,
                 fontweight="bold", color="white", pad=16)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved → {save_path}")
    plt.close(fig)
