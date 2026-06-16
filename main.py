"""
main.py — Módulo 4: Pipeline completo de optimización de racelines en pistas de F1
Conecta todos los módulos y ejecuta el pipeline completo:
====================================================================================
USO:
    python main.py

1. Carga y visualiza las 5 pistas
2. Para cada pista:
       a. Prepara geometría y puntos de control
       b. Ejecuta el AG (n_runs corridas independientes)
       c. Muestra trayectoria optimizada(CSV y grafico) y perfil de velocidad
3. Muestra tabla comparativa de resultados
    
Genera en la carpeta outputs/:
    - {Pista}_raceline.csv   → coordenadas x,y de la mejor trayectoria
    - {Pista}_raceline.png   → gráfica de la pista con líneas y tiempos
    - {Pista}_velocity.png   → perfil de velocidad comparativo
    - 00_circuits_overview.png
    - 99_summary_table.png
"""

import os
import time
import urllib.request
import numpy as np

from track_model import (load_track, compute_track_geometry,
                         select_control_points, objective_function,
                         track_length, control_bounds, trajectory_violation,
                         reconstruct_trajectory, lap_time)
from genetic_algorithm import multi_run, GAConfig
from visualization import (plot_all_tracks, plot_track_result,
                           plot_velocity_comparison, plot_results_table)


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
N_RUNS   = 10       # Corridas independientes del AG
OUTPUT   = "outputs"

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

    ga_config = GAConfig(
        """
        Configuración del Algoritmo Genético:
        - pop_size: tamaño de la población (30)
        - max_evals: número máximo de evaluaciones (3000)
        - pc: probabilidad de cruce (0.90)
        - eta_c: distribución de cruce (5.0)
        - eta_m: distribución de mutación (20.0)
        - tournament_k: tamaño del torneo (2)
        - seed: semilla para reproducibilidad (none = aleatoria) 
        - verbose: mostrar progreso detallado (False)
        """,
        pop_size     = 30,
        max_evals    = 3_000,
        pc           = 0.90,
        eta_c        = 5.0,
        eta_m        = 20.0,
        tournament_k = 2,
        seed         = None,
        verbose      = False,
    )

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

    # -- Vista general de circuitos -------------------------------------------
    print("\n    Generando vista general de circuitos...")
    plot_all_tracks(
        all_geo,
        title="Circuitos F1 Seleccionados — Temporada 2026",
        save_path=os.path.join(OUTPUT, "00_circuits_overview.png"),
    )

    # -- Optimización por pista -----------------------------------------------
    print("\n" + "=" * 60)
    print("  PASO 2: Optimización con Algoritmo Genético")
    print("=" * 60)

    all_results = {}

    for name, geo in all_geo.items():
        print(f"\n{'-' * 55}")
        print(f"    {name}  ({track_length(geo):.2f} km)")
        print(f"{'-' * 55}")

        ctrl = select_control_points(geo, n_ctrl=N_CTRL, adaptive=True)
        lb, ub = control_bounds(ctrl)
        obj = lambda u, _c=ctrl: objective_function(u, _c)

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
        csv_path = os.path.join(OUTPUT, f"{name}_raceline.csv")
        save_raceline_csv(geo, ctrl, stats["best_u"], csv_path)

        # Gráfica de pista + raceline + tiempos
        plot_track_result(
            geo, ctrl, stats["best_u"], name,
            t_center, stats["best"],
            save_path=os.path.join(OUTPUT, f"{name}_raceline.png"),
        )

        # Perfil de velocidad
        plot_velocity_comparison(
            geo, ctrl, stats["best_u"], name,
            save_path=os.path.join(OUTPUT, f"{name}_velocity.png"),
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
