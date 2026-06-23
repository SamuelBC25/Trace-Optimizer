"""
compare.py — Comparativa: penalización en TODA la trayectoria vs SOLO en
                          los puntos de control
=============================================================================
USO:
    python compare.py

Para cada pista en TRACKS, corre el AG dos veces (misma config, mismo
N_CTRL, mismas semillas relativas por corrida) cambiando únicamente la
función objetivo:

    FULL       → objective_function            (track_model.py)
                 penaliza la violación en TODOS los puntos de la
                 trayectoria reconstruida (N * oversample puntos, tras
                 interpolación Akima).

    CTRL_ONLY  → objective_function_ctrl_only   (track_model.py)
                 penaliza la violación SOLO en los N_CTRL puntos de
                 control (igual conjunto de puntos que control_bounds).

No se modifica nada de main.py / genetic_algorithm.py / visualization.py:
este script es un experimento adicional que reutiliza esas piezas.

Genera, por cada pista, en outputs/{Pista}/:

    outputs/{Pista}/full/
        {Pista}_raceline.csv
        {Pista}_raceline.png
        {Pista}_velocity.png
    outputs/{Pista}/ctrl_only/
        {Pista}_raceline.csv
        {Pista}_raceline.png
        {Pista}_velocity.png
    outputs/{Pista}/comparativa_{Pista}.png   → tabla resumen FULL vs CTRL_ONLY
    outputs/{Pista}/comparativa_{Pista}.txt   → resumen en texto plano

Y al final, una tabla global:
    outputs/99_comparativa_global.png
"""

import os
import time
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from track_model import (load_track, compute_track_geometry,
                         select_control_points, objective_function,
                         objective_function_ctrl_only, track_length,
                         control_bounds, trajectory_violation,
                         reconstruct_trajectory, PENALTY_WEIGHT)
from genetic_algorithm import multi_run, GAConfig
from visualization import (plot_track_result, plot_velocity_comparison,
                           _TEXT, _TABLE_HEADER, _TABLE_ROW_A, _TABLE_ROW_B)

# -----------------------------------------------------------------------------
# CONFIGURACIÓN — toda desde config.py (punto único de verdad)
# -----------------------------------------------------------------------------
from config import TRACKS, N_CTRL, N_RUNS, OUTPUT

ga_config = GAConfig()   # defaults de GAConfig también vienen de config.py


def track_output_dir(name):
    return os.path.join(OUTPUT, name)


# -----------------------------------------------------------------------------
# GUARDAR RACELINE CSV  (idéntico a main.py)
# -----------------------------------------------------------------------------

def save_raceline_csv(ctrl, u_opt, filepath):
    tx, ty = reconstruct_trajectory(ctrl, u_opt)
    with open(filepath, "w") as f:
        f.write("# x_m,y_m\n")
        for x, y in zip(tx, ty):
            f.write(f"{x:.6f},{y:.6f}\n")
    print(f"      CSV → {filepath}  ({len(tx)} puntos)")


# -----------------------------------------------------------------------------
# CORRER UNA VARIANTE (full | ctrl_only) PARA UNA PISTA
# -----------------------------------------------------------------------------

def run_variant(name, geo, ctrl, lb, ub, obj_fn, variant_dir, label, penalty):
    """
    Ejecuta multi_run con la función objetivo dada y guarda CSV + gráficas
    en variant_dir. Retorna el dict de stats (igual que multi_run) más
    t_center y métricas de violación medidas SIEMPRE de forma completa
    (sobre toda la interpolación) para que la comparación final sea justa,
    sin importar qué función objetivo se usó para optimizar.

    penalty: peso de penalización pasado a obj_fn.
      - FULL      → PENALTY_WEIGHT (penaliza violaciones en toda la
                    interpolación, como hasta ahora).
      - CTRL_ONLY → 0.0 (formulación fiel a la Ec. restricción de caja -dI<=u<=dE SOLO en los N
                    control points, sin término de penalización extra).
    """
    os.makedirs(variant_dir, exist_ok=True)  # por si run_variant se llama directo (no desde run_pair_sequential)
    obj = lambda u, _c=ctrl: obj_fn(u, _c, penalty=penalty)

    u_center = np.zeros(len(lb))
    t_center = obj(u_center)

    print(f"    [{label}] Tiempo línea central: {t_center:.3f} s")
    t0   = time.time()           # reloj de pared (incluye carga del sistema)
    cpu0 = time.process_time()   # tiempo de CPU del proceso (estable vs. carga externa)
    stats = multi_run(obj, lb, ub, ga_config, n_runs=N_RUNS, verbose=True)
    elapsed = time.time() - t0
    cpu     = time.process_time() - cpu0

    # Violación medida SIEMPRE en toda la trayectoria interpolada (criterio
    # objetivo común para comparar full vs ctrl_only de forma justa).
    viol_t, viol_m = trajectory_violation(ctrl, stats["best_u"])

    # Tiempo de vuelta REAL (sin penalización) de la mejor solución, para
    # comparar "manzanas con manzanas" entre ambas variantes.
    t_real = objective_function(stats["best_u"], ctrl, penalty=0.0)

    stats["t_center"] = t_center
    stats["t_real"]   = t_real
    stats["elapsed"]  = elapsed
    stats["cpu"]      = cpu
    stats["viol_total"] = viol_t
    stats["viol_max"]   = viol_m

    print(f"      Mejor fitness (con penal.) : {stats['best']:.3f} s")
    print(f"      Tiempo de vuelta REAL      : {t_real:.3f} s")
    print(f"      Violación real             : máx {viol_m:.3f} m | total {viol_t:.1f} m")
    print(f"      Tiempo CPU                 : {cpu:.1f} s")
    print(f"      Tiempo reloj (pared)       : {elapsed:.1f} s")

    csv_path = os.path.join(variant_dir, f"{name}_raceline.csv")
    save_raceline_csv(ctrl, stats["best_u"], csv_path)

    plot_track_result(
        geo, ctrl, stats["best_u"], f"{name} [{label}]",
        t_center, t_real,
        save_path=os.path.join(variant_dir, f"{name}_raceline.png"),
    )
    plot_velocity_comparison(
        geo, ctrl, stats["best_u"], f"{name} [{label}]",
        save_path=os.path.join(variant_dir, f"{name}_velocity.png"),
    )

    # La variante FULL también es el resultado "principal" de la pista:
    # copiar sus archivos al nivel outputs/{Pista}/ para que main.py
    # no tenga que optimizar la pista una segunda vez.
    if label == "FULL":
        parent_dir = track_output_dir(name)
        os.makedirs(parent_dir, exist_ok=True)
        for ext in ("_raceline.csv", "_raceline.png", "_velocity.png"):
            src = os.path.join(variant_dir, f"{name}{ext}")
            dst = os.path.join(parent_dir, f"{name}{ext}")
            shutil.copy2(src, dst)

    return stats


_VARIANTS = {
    "FULL":      (objective_function, PENALTY_WEIGHT, "full"),
    "CTRL_ONLY": (objective_function_ctrl_only, 0.0, "ctrl_only"),
}


def run_pair_sequential(name, csv_path):
    """
    Corre FULL y luego CTRL_ONLY de forma secuencial (uno termina, arranca
    el otro). Esto garantiza tiempos de reloj limpios e independientes para
    cada variante: el elapsed de FULL no se solapa con el de CTRL_ONLY y
    viceversa, permitiendo una comparación de tiempos justa.
    Retorna (stats_full, stats_ctrl).
    """
    track = load_track(csv_path)
    geo   = compute_track_geometry(track)
    ctrl  = select_control_points(geo, n_ctrl=N_CTRL, adaptive=True)
    lb, ub = control_bounds(ctrl)

    # Pre-crear toda la jerarquía de directorios antes de lanzar las variantes.
    # Hacerlo aquí (y no dentro de run_variant) garantiza que en Windows la
    # cadena outputs/ → outputs/{Pista}/ → outputs/{Pista}/full/ (y ctrl_only/)
    # exista completamente cuando run_variant intente escribir archivos.
    track_dir = track_output_dir(name)
    for subdir in ("full", "ctrl_only"):
        os.makedirs(os.path.join(track_dir, subdir), exist_ok=True)

    print(f"\n  -- [{name}] Variante FULL (penaliza toda la interpolación) --")
    obj_fn, penalty, subdir = _VARIANTS["FULL"]
    stats_full = run_variant(
        name, geo, ctrl, lb, ub, obj_fn,
        os.path.join(track_dir, subdir), "FULL", penalty,
    )

    print(f"\n  -- [{name}] Variante CTRL_ONLY (sin penalización) --")
    obj_fn, penalty, subdir = _VARIANTS["CTRL_ONLY"]
    stats_ctrl = run_variant(
        name, geo, ctrl, lb, ub, obj_fn,
        os.path.join(track_dir, subdir), "CTRL_ONLY", penalty,
    )

    return stats_full, stats_ctrl



def plot_comparison_table(name, stats_full, stats_ctrl, save_path):
    rows = [
        ["Tiempo línea central (s)",
         f'{stats_full["t_center"]:.3f}', f'{stats_ctrl["t_center"]:.3f}'],
        ["Tiempo real mejor sol. (s)",
         f'{stats_full["t_real"]:.3f}', f'{stats_ctrl["t_real"]:.3f}'],
        ["Fitness c/ penalización (s)",
         f'{stats_full["best"]:.3f}', f'{stats_ctrl["best"]:.3f}'],
        ["Media ± σ fitness (s)",
         f'{stats_full["mean"]:.3f} ± {stats_full["std"]:.4f}',
         f'{stats_ctrl["mean"]:.3f} ± {stats_ctrl["std"]:.4f}'],
        ["Violación total real (m)",
         f'{stats_full["viol_total"]:.2f}', f'{stats_ctrl["viol_total"]:.2f}'],
        ["Violación máxima real (m)",
         f'{stats_full["viol_max"]:.3f}', f'{stats_ctrl["viol_max"]:.3f}'],
        ["Tiempo de CPU (s)",
         f'{stats_full["cpu"]:.1f}', f'{stats_ctrl["cpu"]:.1f}'],
        ["Tiempo de reloj (s)",
         f'{stats_full["elapsed"]:.1f}', f'{stats_ctrl["elapsed"]:.1f}'],
    ]
    cols = ["Métrica", "FULL\n(penaliza toda la interpolación)",
            "CTRL_ONLY\n(penaliza solo 60 ctrl points)"]

    fig, ax = plt.subplots(figsize=(11, 1.5 + 0.6 * len(rows)), dpi=150)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.0)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#999999")
        cell.set_linewidth(1.2)
        if r == 0:
            cell.set_facecolor(_TABLE_HEADER)
            cell.set_text_props(color="white", fontweight="bold", fontsize=11.5)
        else:
            cell.set_facecolor(_TABLE_ROW_A if r % 2 == 0 else _TABLE_ROW_B)
            cell.set_text_props(color=_TEXT, fontsize=11.5)
            if c == 0:
                cell.set_text_props(color=_TEXT, fontsize=11.5, fontweight="bold")

    ax.set_title(f"{name} — Penalización: TODA la pista vs SOLO control points",
                 fontsize=15, fontweight="bold", color=_TEXT, pad=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Saved → {save_path}")


def write_comparison_text(name, stats_full, stats_ctrl, save_path):
    lines = []
    lines.append(f"COMPARATIVA — {name}")
    lines.append("Penalización en TODA la interpolación (FULL) vs SOLO en los "
                  "control points (CTRL_ONLY)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"{'Métrica':<32}{'FULL':>18}{'CTRL_ONLY':>22}")
    lines.append("-" * 78)

    def row(label, f, c, fmt="{:.3f}"):
        lines.append(f"{label:<32}{fmt.format(f):>18}{fmt.format(c):>22}")

    row("Tiempo línea central (s)", stats_full["t_center"], stats_ctrl["t_center"])
    row("Tiempo real mejor sol. (s)", stats_full["t_real"], stats_ctrl["t_real"])
    row("Fitness c/ penalización (s)", stats_full["best"], stats_ctrl["best"])
    row("Media fitness (s)", stats_full["mean"], stats_ctrl["mean"])
    row("σ fitness (s)", stats_full["std"], stats_ctrl["std"])
    row("Violación total real (m)", stats_full["viol_total"], stats_ctrl["viol_total"], "{:.2f}")
    row("Violación máxima real (m)", stats_full["viol_max"], stats_ctrl["viol_max"], "{:.3f}")
    row("Tiempo de CPU (s)", stats_full["cpu"], stats_ctrl["cpu"], "{:.1f}")
    row("Tiempo de reloj (s)", stats_full["elapsed"], stats_ctrl["elapsed"], "{:.1f}")

    lines.append("")
    mejora_full = (stats_full["t_center"] - stats_full["t_real"]) / stats_full["t_center"] * 100
    mejora_ctrl = (stats_ctrl["t_center"] - stats_ctrl["t_real"]) / stats_ctrl["t_center"] * 100
    lines.append(f"Mejora real vs línea central — FULL:      {mejora_full:6.2f} %")
    lines.append(f"Mejora real vs línea central — CTRL_ONLY: {mejora_ctrl:6.2f} %")
    lines.append("")
    lines.append("Nota: 'Tiempo real' y 'Violación' se miden SIEMPRE sobre la trayectoria")
    lines.append("interpolada completa (N*oversample puntos), independientemente de qué")
    lines.append("función objetivo se usó para optimizar. Esto permite comparar de forma")
    lines.append("justa qué tanto se sale de pista cada estrategia en la práctica.")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"    Saved → {save_path}")


def plot_global_summary(all_comparisons, save_path):
    rows = []
    for name, (sf, sc) in all_comparisons.items():
        rows.append([
            name,
            f'{sf["t_real"]:.3f}', f'{sc["t_real"]:.3f}',
            f'{sf["viol_total"]:.2f}', f'{sc["viol_total"]:.2f}',
            f'{sf["viol_max"]:.3f}', f'{sc["viol_max"]:.3f}',
        ])
    cols = ["Circuito", "T. real\nFULL (s)", "T. real\nCTRL_ONLY (s)",
            "Viol. total\nFULL (m)", "Viol. total\nCTRL_ONLY (m)",
            "Viol. máx\nFULL (m)", "Viol. máx\nCTRL_ONLY (m)"]

    fig, ax = plt.subplots(figsize=(14, 1.8 + 0.7 * len(rows)), dpi=150)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1, 2.0)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#999999")
        cell.set_linewidth(1.2)
        if r == 0:
            cell.set_facecolor(_TABLE_HEADER)
            cell.set_text_props(color="white", fontweight="bold", fontsize=10.5)
        else:
            cell.set_facecolor(_TABLE_ROW_A if r % 2 == 0 else _TABLE_ROW_B)
            cell.set_text_props(color=_TEXT, fontsize=11)

    ax.set_title("Comparativa global — Penalización FULL vs CTRL_ONLY",
                 fontsize=17, fontweight="bold", color=_TEXT, pad=18)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Saved → {save_path}")


# -----------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# -----------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT, exist_ok=True)

    print("\n" + "=" * 70)
    print("  COMPARATIVA: penalización FULL vs CTRL_ONLY")
    print("=" * 70)

    all_geo = {}
    for name, csv_path in TRACKS.items():
        if not os.path.exists(csv_path):
            print(f"    {csv_path} no encontrado, se omite.")
            continue
        track = load_track(csv_path)
        geo = compute_track_geometry(track)
        all_geo[name] = geo
        print(f"    {name:<14} {track_length(geo):.2f} km  |  {len(geo['cx'])} puntos")

    if not all_geo:
        print("\n    No se encontraron archivos CSV. Revisa la carpeta tracks/")
        return

    print(f"\n  Configuración AG (igual para ambas variantes):")
    print(f"    pop_size={ga_config.pop_size}, max_evals={ga_config.max_evals}, "
          f"pc={ga_config.pc}, eta_c={ga_config.eta_c}, eta_m={ga_config.eta_m}, "
          f"tournament_k={ga_config.tournament_k}, n_runs={N_RUNS}")

    all_comparisons = {}

    for name, geo in all_geo.items():
        print(f"\n{'=' * 70}")
        print(f"  {name}  ({track_length(geo):.2f} km)")
        print(f"{'=' * 70}")

        track_dir = track_output_dir(name)

        stats_full, stats_ctrl = run_pair_sequential(name, TRACKS[name])

        all_comparisons[name] = (stats_full, stats_ctrl)

        plot_comparison_table(
            name, stats_full, stats_ctrl,
            save_path=os.path.join(track_dir, f"comparativa_{name}.png"),
        )
        write_comparison_text(
            name, stats_full, stats_ctrl,
            save_path=os.path.join(track_dir, f"comparativa_{name}.txt"),
        )

    if len(all_comparisons) >= 1:
        plot_global_summary(
            all_comparisons,
            save_path=os.path.join(OUTPUT, "99_comparativa_global.png"),
        )

    print(f"\n{'=' * 70}")
    print(f"  Comparativa completa. Archivos en '{OUTPUT}/{{Pista}}/full/' y")
    print(f"  '{OUTPUT}/{{Pista}}/ctrl_only/', resumen en "
          f"'{OUTPUT}/{{Pista}}/comparativa_{{Pista}}.png'")
    print(f"{'=' * 70}\n")

    return all_comparisons


if __name__ == "__main__":
    main()