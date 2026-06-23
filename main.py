"""
main.py — Módulo 4: Pipeline completo de optimización de racelines en pistas de F1
====================================================================================
USO:
    python main.py

1. Carga y visualiza las 5 pistas (vista general + una imagen individual por pista)
2. Optimización comparativa FULL vs CTRL_ONLY en paralelo para cada pista
   (delega a compare.py — FULL y CTRL_ONLY corren en dos procesos simultáneos)
3. Tabla comparativa de resultados entre pistas (basada en variante FULL)

Estructura de outputs/:

    outputs/
    ├── tracks/
    │   ├── 00_circuits_overview.png      → vista general (grid)
    │   └── {Pista}_track.png             → pista individual sin raceline
    ├── {Pista}/
    │   ├── {Pista}_raceline.csv          → mejor trayectoria (FULL)
    │   ├── {Pista}_raceline.png          → pista con raceline (FULL)
    │   ├── {Pista}_velocity.png          → perfil de velocidad (FULL)
    │   ├── full/                         → variante FULL  (csv/png/velocity)
    │   ├── ctrl_only/                    → variante CTRL_ONLY (csv/png/velocity)
    │   ├── comparativa_{Pista}.png       → tabla FULL vs CTRL_ONLY
    │   └── comparativa_{Pista}.txt       → resumen en texto
    ├── 99_summary_table.png              → tabla comparativa entre pistas (FULL)
    └── 99_comparativa_global.png         → tabla global FULL vs CTRL_ONLY

Nota: cada pista se optimiza UNA sola vez por variante. Los archivos en
outputs/{Pista}/ (nivel principal) son una copia de la variante FULL,
generados por compare.py sin duplicar trabajo.
"""

import os

from config import TRACKS, N_CTRL, N_RUNS, OUTPUT
from track_model import load_track, compute_track_geometry, track_length
from visualization import (plot_all_tracks, plot_track_individual,
                           plot_results_table)
import compare


# -----------------------------------------------------------------------------
# CONFIGURACIÓN — toda en config.py (TRACKS, N_CTRL, N_RUNS, OUTPUT y AG).
# compare.py también importa desde config.py, así que no hay que sincronizar.
# -----------------------------------------------------------------------------
TRACKS_DIR = os.path.join(OUTPUT, "tracks")


# -----------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# -----------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT, exist_ok=True)
    os.makedirs(TRACKS_DIR, exist_ok=True)

    # -- PASO 1: Cargar y visualizar pistas ------------------------------------
    print("\n" + "=" * 65)
    print("  PASO 1: Cargando y visualizando pistas")
    print("=" * 65)

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

    print("\n    Generando vista general de circuitos...")
    plot_all_tracks(
        all_geo,
        title="Circuitos F1 Seleccionados — Temporada 2026",
        save_path=os.path.join(TRACKS_DIR, "00_circuits_overview.png"),
    )

    print("    Generando vista individual por circuito (sin raceline)...")
    for name, geo in all_geo.items():
        plot_track_individual(
            geo, name,
            save_path=os.path.join(TRACKS_DIR, f"{name}_track.png"),
        )

    # -- PASO 2: Optimización FULL vs CTRL_ONLY en paralelo --------------------
    # compare.main() por cada pista lanza dos procesos simultáneos (FULL y
    # CTRL_ONLY). La variante FULL además copia sus outputs al nivel
    # outputs/{Pista}/ para que sirvan como resultado "principal".
    # compare.main() retorna all_comparisons: {name: (stats_full, stats_ctrl)}
    print("\n" + "=" * 65)
    print("  PASO 2: Optimización con AG — FULL, luego CTRL_ONLY (secuencial)")
    print("=" * 65)

    all_comparisons = compare.main()

    # -- PASO 3: Tabla comparativa entre pistas (basada en FULL) ---------------
    if all_comparisons and len(all_comparisons) > 1:
        print("\n" + "=" * 65)
        print("  PASO 3: Tabla comparativa entre pistas (variante FULL)")
        print("=" * 65)

        # Construir el dict que espera plot_results_table desde los stats de FULL
        all_results = {
            name: {
                "t_center": sf["t_center"],
                "best":     sf["t_real"],    # tiempo real sin penalización
                "mean":     sf["mean"],
                "std":      sf["std"],
                "ctrl":     None,            # no necesario para la tabla
            }
            for name, (sf, _) in all_comparisons.items()
        }

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

    print(f"\n{'=' * 65}")
    print(f"   Pipeline completo. Todos los archivos en '{OUTPUT}/'")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()