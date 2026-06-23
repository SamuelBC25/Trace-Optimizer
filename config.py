"""
config.py — PUNTO ÚNICO DE CONFIGURACIÓN DEL PROYECTO
=====================================================
TODAS las variables del proyecto viven aquí. Si quieres cambiar algo
(velocidad máxima, fricción, parámetros del AG, número de pistas, etc.)
modifícalo SOLO en este archivo. El resto de módulos lo importa desde aquí.

    track_model.py        → física y factibilidad
    genetic_algorithm.py  → defaults de GAConfig
    main.py / compare.py  → pistas y pipeline
    visualization.py      → usa VMAX para los ejes
"""

# -----------------------------------------------------------------------------
# FÍSICA DEL VEHÍCULO Y LA PISTA
# -----------------------------------------------------------------------------
MU   = 0.8      # Coeficiente de fricción neumático–pista
G    = 9.81     # Aceleración gravitacional (m/s²)
VMAX = 83.0     # Velocidad máxima en recta (m/s).  83.0 ≈ 300 km/h
                # ↑ cámbiala aquí a 100.0 si quieres ~360 km/h. UNA sola vez.

# -----------------------------------------------------------------------------
# FACTIBILIDAD DE LA TRAYECTORIA
# -----------------------------------------------------------------------------
SAFETY_MARGIN  = 0.5    # Margen al borde (m): mantiene el carro dentro de pista
PENALTY_WEIGHT = 0.10   # Peso de la penalización por salirse (por metro·muestra)
OVERSAMPLE     = 5      # Densidad de muestreo de la trayectoria reconstruida

# -----------------------------------------------------------------------------
# ALGORITMO GENÉTICO  (defaults de GAConfig)
# -----------------------------------------------------------------------------
POP_SIZE      = 100      # Tamaño de población
N_GENERATIONS = 600      # Número de generaciones 
PC            = 0.90     # Probabilidad de cruzamiento SBX
PM            = None     # Probabilidad de mutación (None → 1/N)
ETA_C         = 5.0      # Índice de distribución SBX (Nc)
ETA_M         = 15.0     # Índice de distribución mutación (Nm)
TOURNAMENT_K  = 2        # Tamaño del torneo
SEED          = None     # Semilla aleatoria (None = aleatoria)
VERBOSE       = True     # Imprimir progreso
MAX_EVALS     = None     # Número máximo de evaluaciones
# --- PARÁMETROS PARA CONVERGENCIA (EARLY STOPPING) ---
PATIENCE      = 50        # Generaciones sin mejora antes de detenerse (None para desactivar)
TOLERANCE     = 1e-4      # Mejora mínima en segundos para considerar que no ha estancado (0.0001 = 0.1 ms)


# -----------------------------------------------------------------------------
# PIPELINE (pistas, control points, corridas, salidas)
# -----------------------------------------------------------------------------
TRACKS = {
    "Shanghai":    "tracks/Shanghai.csv",
    "Suzuka":      "tracks/Suzuka.csv",
    "Silverstone": "tracks/Silverstone.csv",
    "Zandvoort":   "tracks/Zandvoort.csv",
    "Spa":         "tracks/Spa.csv",
}

N_CTRL = 30        # Puntos de control que optimiza el AG por pista
N_RUNS = 10        # Corridas independientes del AG (estadística)
OUTPUT = "outputs" # Carpeta de resultados

# -----------------------------------------------------------------------------
# VISUALIZACIÓN
# -----------------------------------------------------------------------------
VEL_SMOOTH_WINDOW = 61   # Ventana (impar) del suavizado visual del perfil de
                         # velocidad. NO cambia los datos: sólo dibuja encima
                         # una línea-guía legible. Sube el valor = más suave.
