"""
main.py — Módulo 4: Pipeline completo de optimización de racelines en pistas de F1
====================================================================================
USO:
    python main.py

1. Carga y visualiza las 5 pistas (vista general + una imagen individual por pista)
2. Para cada pista:
       a. Prepara geometría y puntos de control
       b. Ejecuta el AG (n_runs corridas independientes)
       c. Muestra trayectoria optimizada (CSV y gráfico) y perfil de velocidad
3. Muestra tabla comparativa de resultados

Genera en la carpeta outputs/, organizado en subcarpetas:

    outputs/
    --- tracks/
    |   --- 00_circuits_overview.png   → vista general (grid) de todas las pistas
    |   --- {Pista}_track.png          → vista individual de la pista, SIN raceline
    --- {Pista}/
    |   --- {Pista}_raceline.csv       → coordenadas x,y de la mejor trayectoria
    |   --- {Pista}_raceline.png       → gráfica de la pista con raceline y tiempos
    |   --- {Pista}_velocity.png       → perfil de velocidad comparativo
    --- 99_summary_table.png           → tabla comparativa entre todas las pistas
"""

import os
import time
import numpy as np

from track_model import (load_track, compute_track_geometry,
                         select_control_points, objective_function,
                         track_length, control_bounds, trajectory_violation,
                         reconstruct_trajectory, lap_time)
from genetic_algorithm import multi_run, GAConfig
from visualization import (plot_all_tracks, plot_track_individual,
                           plot_track_result, plot_velocity_comparison,
                           plot_results_table)


# -----------------------------------------------------------------------------
# CONFIGURACIÓN Y CONSTANTES DE PISTAS
# -----------------------------------------------------------------------------

TRACKS = {
    "Shanghai":    "tracks/Shanghai.csv",
    "Suzuka":      "tracks/Suzuka.csv",
    "Silverstone": "tracks/Silverstone.csv",
    "Zandvoort":   "tracks/Zandvoort.csv",
    "Spa":         "tracks/Spa.csv",
}

N_CTRL   = 60      # Puntos de control (más = mejor pero más lento)
N_RUNS   = 10      # Corridas independientes del AG
OUTPUT   = "outputs"
TRACKS_DIR = os.path.join(OUTPUT, "tracks")   # vistas sin raceline + overview


def track_output_dir(name):
    """Carpeta de salida para los resultados optimizados de una pista: outputs/{Pista}/"""
    return os.path.join(OUTPUT, name)

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DEL ALGORITMO GENÉTICO
# -----------------------------------------------------------------------------
# Todos los parámetros se controlan desde GAConfig en genetic_algorithm.py.
# Para ajustar el AG, modifica los defaults allá — no aquí.
# -----------------------------------------------------------------------------

ga_config = GAConfig()


# -----------------------------------------------------------------------------
# GUARDAR RACELINE CSV
# -----------------------------------------------------------------------------

def save_raceline_csv(geo, ctrl, u_opt, filepath):
    """Guarda la trayectoria optimizada en formato TUMFTM: # x_m,y_m"""
    tx, ty = reconstruct_trajectory(ctrl, u_opt)
    with open(filepath, "w") as f:
        f.write("# x_m,y_m\n")
        for x, y in zip(tx, ty):
            f.write(f"{x:.6f},{y:.6f}\n")
    print(f"    CSV → {filepath}  ({len(tx)} puntos)")


# -----------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# -----------------------------------------------------------------------------

def main():
    """
    Pipeline completo: carga, optimización y visualización.
    Genera archivos CSV y gráficos para cada pista, más una tabla comparativa.
    """
    os.makedirs(OUTPUT, exist_ok=True)
    os.makedirs(TRACKS_DIR, exist_ok=True)

    # -- Cargar pistas --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PASO 1: Cargando pistas")
    print("=" * 60)

    all_geo = {}
    for name, csv_path in TRACKS.items():
        if not os.path.exists(csv_path):
            print(f"    {csv_path} no encontrado, se omite.")
            continue
        track = load_track(csv_path)
        geo   = compute_track_geometry(track)
        all_geo[name] = geo
        print(f"    {name:<14} {track_length(geo):.2f} km  |  {len(geo['cx'])} puntos")

    if not all_geo:
        print("\n    No se encontraron archivos CSV. Revisa la carpeta tracks/")
        return

    # -- Vista general de circuitos (grid) ------------------------------------
    print("\n    Generando vista general de circuitos...")
    plot_all_tracks(
        all_geo,
        title="Circuitos F1 Seleccionados — Temporada 2026",
        save_path=os.path.join(TRACKS_DIR, "00_circuits_overview.png"),
    )

    # -- Vista individual de cada circuito (sin raceline) ----------------------
    print("    Generando vista individual de cada circuito (sin raceline)...")
    for name, geo in all_geo.items():
        plot_track_individual(
            geo, name,
            save_path=os.path.join(TRACKS_DIR, f"{name}_track.png"),
        )

    # -- Optimización por pista -----------------------------------------------
    print("\n" + "=" * 60)
    print("  PASO 2: Optimización con Algoritmo Genético")
    print("=" * 60)

    # Mostrar configuración activa del AG
    print(f"\n  Configuración AG:")
    print(f"    poblacion={ga_config.pop_size}, Generaciones={ga_config.n_generations}, "
          f"pc={ga_config.pc}, eta_c={ga_config.eta_c}, eta_m={ga_config.eta_m}, "
          f"Tam. Torneo={ga_config.tournament_k}")

    all_results = {}

    for name, geo in all_geo.items():
        print(f"\n{'-' * 55}")
        print(f"    {name}  ({track_length(geo):.2f} km)")
        print(f"{'-' * 55}")

        ctrl = select_control_points(geo, n_ctrl=N_CTRL, adaptive=True)
        lb, ub = control_bounds(ctrl)
        obj = lambda u, _c=ctrl: objective_function(u, _c)

        # Carpeta de salida específica de esta pista: outputs/{Pista}/
        track_dir = track_output_dir(name)
        os.makedirs(track_dir, exist_ok=True)

        # Tiempo línea central
        u_center = np.zeros(N_CTRL)
        t_center = obj(u_center)
        print(f"  Tiempo línea central: {t_center:.3f} s")

        # Ejecutar AG
        t0 = time.time()
        stats = multi_run(obj, lb, ub, ga_config, n_runs=N_RUNS, verbose=True)
        elapsed = time.time() - t0

        stats["t_center"] = t_center
        stats["ctrl"]     = ctrl
        stats["elapsed"]  = elapsed
        all_results[name] = stats

        mejora = (t_center - stats["best"]) / t_center * 100
        viol_t, viol_m = trajectory_violation(ctrl, stats["best_u"])
        print(f"\n  Mejor tiempo   : {stats['best']:.3f} s")
        print(f"  Media ± σ      : {stats['mean']:.3f} ± {stats['std']:.4f} s")
        print(f"  Mejora         : {mejora:.2f}%")
        print(f"  Violación      : máx {viol_m:.2f} m | total {viol_t:.1f} m")
        print(f"  Tiempo cómputo : {elapsed:.1f} s")

        # Guardar CSV de la raceline
        csv_path = os.path.join(track_dir, f"{name}_raceline.csv")
        save_raceline_csv(geo, ctrl, stats["best_u"], csv_path)

        # Gráfica de pista + raceline + tiempos
        plot_track_result(
            geo, ctrl, stats["best_u"], name,
            t_center, stats["best"],
            save_path=os.path.join(track_dir, f"{name}_raceline.png"),
        )

        # Perfil de velocidad
        plot_velocity_comparison(
            geo, ctrl, stats["best_u"], name,
            save_path=os.path.join(track_dir, f"{name}_velocity.png"),
        )

    # -- Tabla comparativa ----------------------------------------------------
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("  PASO 3: Resultados comparativos")
        print(f"{'=' * 60}")
        header = f"  {'Circuito':<14} {'Centro':>8} {'Mejor':>8} {'Media':>8} {'σ':>7} {'Mejora':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, r in all_results.items():
            mej = (r["t_center"] - r["best"]) / r["t_center"] * 100
            print(f"  {name:<14} {r['t_center']:>8.3f} {r['best']:>8.3f}"
                  f" {r['mean']:>8.3f} {r['std']:>7.4f} {mej:>7.2f}%")

        plot_results_table(
            all_results,
            save_path=os.path.join(OUTPUT, "99_summary_table.png"),
        )

    print(f"\n{'=' * 60}")
    print(f"   Pipeline completo. Todos los archivos en '{OUTPUT}/'")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()