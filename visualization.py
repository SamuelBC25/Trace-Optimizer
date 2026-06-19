"""
visualization.py — Módulo 3: Visualización y análisis de resultados
==================================================================
Versión optimizada para proyección: fondo blanco, colores de alto contraste,
líneas gruesas y tipografía grande para máxima legibilidad en presentaciones.

Funciones principales:
- plot_track_individual:    Pista individual SIN raceline (solo trazado/bordes).
- plot_track_result:        Gráfica detallada de la pista con raceline optimizada.
- plot_velocity_comparison: Perfil de velocidad comparativo entre centro y raceline.
- plot_all_tracks:          Vista general (grid) de todas las pistas cargadas.
- plot_results_table:       Tabla resumen de resultados como figura.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from track_model import (reconstruct_trajectory, lap_time, velocity_profile,
                         track_length)


# =============================================================================
# PALETA DE ALTO CONTRASTE — OPTIMIZADA PARA PROYECTOR
# =============================================================================
# Fondo blanco con texto oscuro: máxima legibilidad bajo cualquier iluminación.
# Colores saturados y bien separados en el espectro para que no se confundan.

_BG        = "#FFFFFF"       # Fondo blanco limpio
_TEXT      = "#1A1A1A"       # Texto negro/gris muy oscuro
_GRID      = "#CCCCCC"       # Rejilla gris suave

# Pista
_TRACK     = "#D5D5D5"       # Superficie de asfalto (gris claro)
_BORDER_E  = "#2C2C2C"       # Borde exterior (gris carbón)
_BORDER_I  = "#2C2C2C"       # Borde interior (gris carbón)

# Líneas principales — colores máximamente distinguibles
_CENTER    = "#1565C0"       # Línea central: azul intenso
_RACING    = "#D32F2F"       # Raceline optimizada: rojo intenso
_ACCENT    = "#00838F"       # Acento para cajas informativas: teal oscuro
_GOLD      = "#FF8F00"       # Marcador de salida/meta: ámbar

# Velocidad
_VEL_CENTER = "#1565C0"      # Perfil velocidad centro: mismo azul
_VEL_OPT    = "#D32F2F"      # Perfil velocidad optima: mismo rojo
_VEL_FILL_C = "#1565C0"      # Relleno centro
_VEL_FILL_O = "#D32F2F"      # Relleno optimizada

# Tabla
_TABLE_HEADER  = "#1565C0"   # Encabezado tabla: azul
_TABLE_ROW_A   = "#F5F5F5"   # Fila par: gris muy claro
_TABLE_ROW_B   = "#E8E8E8"   # Fila impar: gris claro

# =============================================================================
# CONFIGURACIÓN GLOBAL DE MATPLOTLIB — PROYECTOR-FRIENDLY
# =============================================================================
plt.rcParams.update({
    "figure.facecolor":   _BG,
    "axes.facecolor":     _BG,
    "text.color":         _TEXT,
    "axes.labelcolor":    _TEXT,
    "axes.edgecolor":     "#666666",
    "xtick.color":        _TEXT,
    "ytick.color":        _TEXT,
    "font.family":        "sans-serif",
    "font.size":          13,          # Base más grande para proyección
    "axes.titlesize":     18,
    "axes.labelsize":     14,
    "legend.fontsize":    12,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "lines.linewidth":    2.0,
    "axes.linewidth":     1.2,
    "grid.alpha":         0.3,
    "grid.color":         _GRID,
})

# -----------------------------------------------------------------------------
# Escala / tamaño de figura
# -----------------------------------------------------------------------------
# Tamaño físico (pulgadas) y resolución usados para las gráficas de pista
# individuales (plot_track_individual y plot_track_result). Subir estos
# valores hace que la pista se dibuje más grande y que la raceline se
# distinga con más claridad de los bordes de la pista.
_TRACK_FIGSIZE   = (18, 14)   # antes: (14, 10)
_TRACK_DPI_SHOW  = 160        # dpi en pantalla
_TRACK_DPI_SAVE  = 260        # dpi al guardar (antes: 200)
_TRACK_PAD_FRAC  = 0.04       # margen alrededor de la pista, como fracción de su tamaño


def _close_loop(x, y):
    """Cierra el circuito conectando el último punto con el primero."""
    return np.append(x, x[0]), np.append(y, y[0])


def _set_tight_limits(ax, *xy_pairs, pad_frac=_TRACK_PAD_FRAC):
    """
    Ajusta los límites de los ejes ceñidos a la geometría de la pista
    (en vez de dejar que matplotlib decida), con un margen pequeño.
    Esto "acerca" visualmente la pista, agrandando su escala aparente
    y dejando más espacio relativo para distinguir las líneas entre sí.
    """
    xs = np.concatenate([p[0] for p in xy_pairs])
    ys = np.concatenate([p[1] for p in xy_pairs])
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    dx = (x_max - x_min) or 1.0
    dy = (y_max - y_min) or 1.0
    pad_x = dx * pad_frac
    pad_y = dy * pad_frac
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)


# =============================================================================
# PISTA INDIVIDUAL — SIN RACELINE
# =============================================================================

def plot_track_individual(geo, track_name, save_path=None):
    """
    Gráfica individual de una sola pista, SIN raceline ni línea central
    optimizada: solo la superficie, los bordes interior/exterior y el
    marcador de salida/meta. Pensada como vista "limpia" de cada circuito,
    generada en un archivo propio (en vez de un grid con todas juntas).

    A diferencia del grid de plot_all_tracks, esta función usa una escala
    grande (figsize/dpi altos) y recorta los ejes ceñidos a la pista para
    que el trazado se aprecie con el mayor detalle posible.
    """
    fig, ax = plt.subplots(figsize=_TRACK_FIGSIZE, dpi=_TRACK_DPI_SHOW)

    # -- Superficie de pista --------------------------------------------------
    ex, ey = _close_loop(geo["ext_x"], geo["ext_y"])
    ix, iy = _close_loop(geo["int_x"], geo["int_y"])

    ax.fill(ex, ey, color=_TRACK, alpha=0.7, zorder=1)
    ax.fill(ix, iy, color=_BG,    zorder=2)

    # -- Bordes (gruesos, oscuros) --------------------------------------------
    ax.plot(ex, ey, color=_BORDER_E, lw=2.4, alpha=0.9, zorder=3,
            label="Borde exterior")
    ax.plot(ix, iy, color=_BORDER_I, lw=2.4, alpha=0.9, zorder=3,
            label="Borde interior")

    # -- Línea central (azul, punteada) ---------------------------------------
    ccx, ccy = _close_loop(geo["cx"], geo["cy"])
    ax.plot(ccx, ccy, color=_CENTER, lw=2.5, ls="--", alpha=0.85, zorder=4,
            label="Línea central")

    # -- Marcador de salida/meta ----------------------------------------------
    ax.scatter(geo["cx"][0], geo["cy"][0], s=220, c=_GOLD, marker="D",
               edgecolors="black", linewidths=1.4, zorder=7,
               label="Salida / Meta")

    # -- Caja informativa -------------------------------------------------
    km = track_length(geo)
    box_text = f"Longitud: {km:.2f} km\nPuntos: {len(geo['cx'])}"
    props = dict(boxstyle="round,pad=0.6", facecolor="#FAFAFA",
                 edgecolor=_ACCENT, alpha=0.95, linewidth=2.0)
    ax.text(0.02, 0.98, box_text, transform=ax.transAxes, fontsize=13,
            verticalalignment="top", fontfamily="monospace",
            bbox=props, zorder=10, color=_TEXT)

    leg = ax.legend(loc="lower right", fontsize=13, framealpha=0.95,
                    facecolor="#FAFAFA", edgecolor="#999999",
                    labelcolor=_TEXT)
    leg.get_frame().set_linewidth(1.5)

    # -- Ejes, escala ceñida a la pista ----------------------------------------
    ax.set_aspect("equal")
    _set_tight_limits(ax, (ex, ey), (ix, iy))
    ax.set_title(f"{track_name}  ({km:.2f} km)",
                 fontsize=20, fontweight="bold", color=_TEXT, pad=18)
    ax.set_xlabel("x  (m)", fontsize=15)
    ax.set_ylabel("y  (m)", fontsize=15)
    ax.grid(True, alpha=0.25, color=_GRID, linewidth=0.8)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=_TRACK_DPI_SAVE, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved → {save_path}")
    plt.close(fig)


# =============================================================================
# GRÁFICA DE PISTA CON RACELINE OPTIMIZADA
# =============================================================================

def plot_track_result(geo, ctrl, u_opt, track_name, t_center, t_opt,
                      save_path=None):
    """
    Gráfica de pista de alta calidad para proyección:
      - Superficie de pista (relleno entre bordes)
      - Bordes exterior / interior (negro grueso)
      - Línea central (azul, punteada)
      - Raceline optimizada (rojo, sólida gruesa)
      - Caja comparativa de tiempos

    Usa una escala grande (figsize/dpi altos) y recorta los ejes ceñidos
    a la geometría de la pista, de modo que la raceline se distinga con
    claridad frente a los bordes y la línea central.
    """
    fig, ax = plt.subplots(figsize=_TRACK_FIGSIZE, dpi=_TRACK_DPI_SHOW)

    # -- Superficie de pista --------------------------------------------------
    ex, ey = _close_loop(geo["ext_x"], geo["ext_y"])
    ix, iy = _close_loop(geo["int_x"], geo["int_y"])

    ax.fill(ex, ey, color=_TRACK, alpha=0.7, zorder=1)
    ax.fill(ix, iy, color=_BG,    zorder=2)

    # -- Bordes (gruesos, oscuros) --------------------------------------------
    ax.plot(ex, ey, color=_BORDER_E, lw=2.2, alpha=0.9, zorder=3,
            label="Bordes de pista")
    ax.plot(ix, iy, color=_BORDER_I, lw=2.2, alpha=0.9, zorder=3)

    # -- Línea central (azul, punteada) ---------------------------------------
    ccx, ccy = _close_loop(geo["cx"], geo["cy"])
    ax.plot(ccx, ccy, color=_CENTER, lw=2.7, ls="--", alpha=0.85, zorder=4,
            label=f"Línea central  ({t_center:.3f} s)")

    # -- Raceline optimizada (rojo, sólida gruesa) ----------------------------
    tx, ty = reconstruct_trajectory(ctrl, u_opt)
    tx_c, ty_c = _close_loop(tx, ty)
    ax.plot(tx_c, ty_c, color=_RACING, lw=3.6, alpha=0.95, zorder=5,
            path_effects=[pe.Stroke(linewidth=6.0, foreground="white", alpha=0.5),
                          pe.Normal()],
            label=f"Optimizada   ({t_opt:.3f} s)")

    # -- Marcador de salida/meta ----------------------------------------------
    ax.scatter(geo["cx"][0], geo["cy"][0], s=200, c=_GOLD, marker="D",
               edgecolors="black", linewidths=1.3, zorder=7,
               label="Salida / Meta")

    # -- Caja comparativa de tiempos ------------------------------------------
    mejora = (t_center - t_opt) / t_center * 100
    km = track_length(geo)
    box_text = (
        f"Longitud: {km:.2f} km\n"
        f"─────────────────────\n"
        f"Tiempo central:    {t_center:.3f} s\n"
        f"Tiempo optimizado: {t_opt:.3f} s\n"
        f"─────────────────────\n"
        f"Mejora:  {mejora:.2f} %\n"
        f"Ahorro:  {t_center - t_opt:.3f} s"
    )
    props = dict(boxstyle="round,pad=0.6", facecolor="#FAFAFA",
                 edgecolor=_ACCENT, alpha=0.95, linewidth=2.0)
    ax.text(0.02, 0.98, box_text, transform=ax.transAxes, fontsize=13,
            verticalalignment="top", fontfamily="monospace",
            bbox=props, zorder=10, color=_TEXT)

    # -- Leyenda --------------------------------------------------------------
    leg = ax.legend(loc="lower right", fontsize=13, framealpha=0.95,
                    facecolor="#FAFAFA", edgecolor="#999999",
                    labelcolor=_TEXT)
    leg.get_frame().set_linewidth(1.5)

    # -- Ejes, escala ceñida a la pista ----------------------------------------
    ax.set_aspect("equal")
    _set_tight_limits(ax, (ex, ey), (ix, iy), (tx_c, ty_c))
    ax.set_title(f"{track_name} — Línea de Carrera Optimizada (AG)",
                 fontsize=20, fontweight="bold", color=_TEXT, pad=18)
    ax.set_xlabel("x  (m)", fontsize=15)
    ax.set_ylabel("y  (m)", fontsize=15)
    ax.grid(True, alpha=0.25, color=_GRID, linewidth=0.8)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=_TRACK_DPI_SAVE, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved → {save_path}")
    plt.close(fig)


# =============================================================================
# PERFIL DE VELOCIDAD COMPARATIVO
# =============================================================================

def plot_velocity_comparison(geo, ctrl, u_opt, track_name, save_path=None):
    """
    Perfil de velocidad: línea central vs optimizada.
    Colores claramente diferenciados con relleno semitransparente.
    """
    # Línea central
    cx_c, cy_c = _close_loop(geo["cx"], geo["cy"])
    d_c, v_c = velocity_profile(cx_c, cy_c)

    # Optimizada
    tx, ty = reconstruct_trajectory(ctrl, u_opt)
    tx_c, ty_c = _close_loop(tx, ty)
    d_o, v_o = velocity_profile(tx_c, ty_c)

    fig, ax = plt.subplots(figsize=(16, 5.5), dpi=150)

    # Interpolar ambos perfiles a un eje de distancia normalizado común
    # para poder comparar y hacer fill_between sin errores de tamaño
    max_dist = max(d_c[-1], d_o[-1])
    n_common = 2000
    d_common = np.linspace(0, max_dist, n_common)
    v_c_interp = np.interp(d_common, d_c, v_c)
    v_o_interp = np.interp(d_common, d_o, v_o)
    d_km = d_common / 1000

    # Relleno de diferencia: resalta dónde cada una es más rápida
    ax.fill_between(d_km, v_c_interp, v_o_interp,
                    where=v_o_interp > v_c_interp,
                    color=_VEL_FILL_O, alpha=0.18, zorder=1,
                    interpolate=True)
    ax.fill_between(d_km, v_c_interp, v_o_interp,
                    where=v_c_interp > v_o_interp,
                    color=_VEL_FILL_C, alpha=0.15, zorder=1,
                    interpolate=True)

    # Línea optimizada primero (debajo), luego central encima para que no se pierda
    ax.plot(d_o / 1000, v_o, color=_VEL_OPT, lw=2.8, alpha=0.90,
            ls="-", zorder=3, label="Raceline optimizada")
    ax.plot(d_c / 1000, v_c, color=_VEL_CENTER, lw=2.5, alpha=0.95,
            ls="-", zorder=4,
            path_effects=[pe.Stroke(linewidth=4.5, foreground="white", alpha=0.6),
                          pe.Normal()],
            label="Línea central")

    # Estadísticas de velocidad en caja informativa
    v_mean_c = np.mean(v_c)
    v_mean_o = np.mean(v_o)
    v_max_c  = np.max(v_c)
    v_max_o  = np.max(v_o)
    stats_text = (
        f"Vel. media central:    {v_mean_c:.1f} km/h\n"
        f"Vel. media optimizada: {v_mean_o:.1f} km/h\n"
        f"Vel. máx central:      {v_max_c:.1f} km/h\n"
        f"Vel. máx optimizada:   {v_max_o:.1f} km/h"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="#FAFAFA",
                 edgecolor="#999999", alpha=0.95, linewidth=1.5)
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=10.5,
            verticalalignment="top", fontfamily="monospace",
            bbox=props, zorder=10, color=_TEXT)

    ax.set_xlabel("Distancia  (km)", fontsize=14)
    ax.set_ylabel("Velocidad  (km/h)", fontsize=14)
    ax.set_title(f"{track_name} — Perfil de Velocidad", fontsize=16,
                 fontweight="bold", color=_TEXT, pad=12)
    ax.set_xlim(0, max(d_c[-1], d_o[-1]) / 1000)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.25, color=_GRID, linewidth=0.8)

    leg = ax.legend(loc="upper right", fontsize=12, framealpha=0.95,
                    facecolor="#FAFAFA", edgecolor="#999999", labelcolor=_TEXT)
    leg.get_frame().set_linewidth(1.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved → {save_path}")
    plt.close(fig)


# =============================================================================
# VISTA GENERAL — PISTAS EN GRID
# =============================================================================

def plot_all_tracks(all_geo, title="Circuitos F1 Seleccionados", save_path=None):
    """Vista general de todos los circuitos cargados, con alto contraste."""
    n = len(all_geo)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows), dpi=150)
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for i, (name, geo) in enumerate(all_geo.items()):
        ax = axes[i]
        ex, ey = _close_loop(geo["ext_x"], geo["ext_y"])
        ix, iy = _close_loop(geo["int_x"], geo["int_y"])

        ax.fill(ex, ey, color=_TRACK, alpha=0.6)
        ax.fill(ix, iy, color=_BG)
        ax.plot(ex, ey, color=_BORDER_E, lw=1.8, alpha=0.85)
        ax.plot(ix, iy, color=_BORDER_I, lw=1.8, alpha=0.85)

        ccx, ccy = _close_loop(geo["cx"], geo["cy"])
        ax.plot(ccx, ccy, color=_CENTER, lw=1.6, ls="--", alpha=0.7)

        km = track_length(geo)
        ax.set_aspect("equal")
        _set_tight_limits(ax, (ex, ey), (ix, iy))
        ax.set_title(f"{name}  ({km:.2f} km)", fontsize=16,
                     fontweight="bold", color=_TEXT)
        ax.grid(True, alpha=0.15, color=_GRID)
        ax.tick_params(labelsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=22, fontweight="bold", color=_TEXT, y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved → {save_path}")
    plt.close(fig)


# =============================================================================
# TABLA COMPARATIVA DE RESULTADOS
# =============================================================================

def plot_results_table(all_results, save_path=None):
    """Tabla resumen como figura, con colores legibles para proyección."""
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

    cols = ["Circuito", "Centro (s)", "Mejor (s)", "Media (s)", "Std (s)", "Mejora"]

    fig, ax = plt.subplots(figsize=(11, 1.5 + 0.6 * len(rows_data)), dpi=150)
    ax.axis("off")

    table = ax.table(cellText=rows_data, colLabels=cols, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.8)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#999999")
        cell.set_linewidth(1.2)
        if r == 0:
            # Encabezado
            cell.set_facecolor(_TABLE_HEADER)
            cell.set_text_props(color="white", fontweight="bold", fontsize=13)
        else:
            # Filas de datos
            cell.set_facecolor(_TABLE_ROW_A if r % 2 == 0 else _TABLE_ROW_B)
            cell.set_text_props(color=_TEXT, fontsize=12)

    ax.set_title("Resultados de Optimización", fontsize=17,
                 fontweight="bold", color=_TEXT, pad=18)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved → {save_path}")
    plt.close(fig)