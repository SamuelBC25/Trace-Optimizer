"""
track_model.py — Módulo 1: Modelo de Pista y Vehículo 
"""

import numpy as np
from scipy.interpolate import Akima1DInterpolator

# -----------------------------------------------------------------------------
# PARÁMETROS FÍSICOS GLOBALES
# -----------------------------------------------------------------------------
MU    = 0.8     # Coeficiente de fricción neumático–pista
G     = 9.81    # Aceleración gravitacional (m/s²)
VMAX  = 83.0    # Velocidad máxima F1 en curva: ~300 km/h ≈ 83 m/s

# Parámetros de factibilidad de la trayectoria
SAFETY_MARGIN  = 0.5    # margen al borde (m): mantiene el carro dentro de pista
PENALTY_WEIGHT = 0.10   # peso de la penalización por salirse (por metro·muestra)
OVERSAMPLE     = 5      # densidad de muestreo de la trayectoria reconstruida

# -----------------------------------------------------------------------------
# 1. CARGA DE PISTA
# -----------------------------------------------------------------------------

def load_track(filepath: str) -> dict:
    """
    Carga un CSV.

    Columnas: x_m, y_m, w_tr_right_m, w_tr_left_m
    Retorna dict con arrays numpy: cx, cy, dE, dI
    """
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    return {
        "cx": data[:, 0],
        "cy": data[:, 1],
        "dE": data[:, 2],
        "dI": data[:, 3],
    }

# -----------------------------------------------------------------------------
# 2. GEOMETRÍA DE PISTA Y SELECCIÓN DE PUNTOS DE CONTROL
# -----------------------------------------------------------------------------

def compute_track_geometry(track: dict) -> dict:
    """
    Calcula vectores tangentes, normales unitarios y coordenadas de los bordes.

    Ec. (1)  vᵢ = (xᵢ₊₁ - xᵢ, yᵢ₊₁ - yᵢ)
    Ec. (2)  n̂ᵢ = (dyᵢ, - dxᵢ) / ‖vᵢ‖
    Ec. (3)  B_ext = cᵢ + dEᵢ·n̂ᵢ
             B_int = cᵢ - dIᵢ·n̂ᵢ
    """
    cx, cy = track["cx"], track["cy"]
    dE, dI = track["dE"], track["dI"]
    dx = np.roll(cx, -1) - cx
    dy = np.roll(cy, -1) - cy
    norms = np.sqrt(dx**2 + dy**2)
    norms[norms < 1e-9] = 1e-9
    nx =  dy / norms
    ny = -dx / norms
    return {
        **track,
        "nx": nx, "ny": ny,
        "ext_x": cx + dE * nx,
        "ext_y": cy + dE * ny,
        "int_x": cx - dI * nx,
        "int_y": cy - dI * ny,
    }


def _curvature_profile(geo: dict) -> np.ndarray:
    """Curvatura aproximada |κ| en cada punto de la línea central (1/r)."""
    cx, cy = geo["cx"], geo["cy"]
    xp, yp = np.roll(cx, 1),  np.roll(cy, 1)
    xn, yn = np.roll(cx, -1), np.roll(cy, -1)
    a = np.hypot(cx - xn, cy - yn)
    b = np.hypot(xp - xn, yp - yn)
    c = np.hypot(xp - cx, yp - cy)
    area = 0.5 * np.abs(xp * (cy - yn) + cx * (yn - yp) + xn * (yp - cy))
    r = np.where(area < 1e-9, 1e9, a * b * c / (4 * area + 1e-12))
    return 1.0 / r


def select_control_points(geo: dict, n_ctrl: int = 40,
                          adaptive: bool = True) -> dict:
    """
    Selecciona n_ctrl puntos de control a lo largo de la pista.

    adaptive=True  → más densos en zonas de alta curvatura.
    adaptive=False → espaciado uniforme.

    Retorna un dict de control que conserva:
      - cx, cy, nx, ny, dE, dI en los puntos de control (para límites y dibujo)
      - _geo: geometría DENSA completa (para reconstrucción fiel)
      - _idx: índices de los puntos de control dentro de la línea central
    """
    N = len(geo["cx"])
    if adaptive:
        kappa = _curvature_profile(geo)
        weight = 1.0 + 6.0 * (kappa / (kappa.max() + 1e-12))
        cum = np.cumsum(weight)
        cum = cum / cum[-1]
        targets = np.linspace(0, 1, n_ctrl, endpoint=False)
        idx = np.searchsorted(cum, targets) % N
        idx = np.unique(idx)
        if len(idx) < n_ctrl:
            extra = np.setdiff1d(np.arange(N), idx)
            need = n_ctrl - len(idx)
            idx = np.sort(np.concatenate([idx, extra[:need]]))
    else:
        idx = np.round(np.linspace(0, N, n_ctrl, endpoint=False)).astype(int) % N
        idx = np.unique(idx)
    ctrl = {k: (v[idx] if isinstance(v, np.ndarray) else v)
            for k, v in geo.items()}
    ctrl["_geo"] = geo
    ctrl["_idx"] = idx
    return ctrl

# -----------------------------------------------------------------------------
# 3. MODELO DEL VEHÍCULO
# -----------------------------------------------------------------------------

def curvature_radius(p1, p2, p3):
    """Radio de curvatura local por la fórmula del circunradio."""
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
    if area < 1e-9:
        return 1e9
    return (a * b * c) / (4.0 * area)


def max_safe_velocity(r, mu=MU, g=G, vmax_limit=VMAX):
    """Velocidad máxima segura sin derrapar."""
    return min(np.sqrt(mu * g * r), vmax_limit)

# -----------------------------------------------------------------------------
# 4. TRAYECTORIA   
# -----------------------------------------------------------------------------
def _akima_periodic(s_nodes, vals, N, s_query, pad=3):
    """Interpolación Akima con envoltura periódica (sin salto en la meta)."""
    order = np.argsort(s_nodes)
    s_nodes = s_nodes[order]
    vals = vals[order]
    s_ext = np.concatenate([s_nodes[-pad:] - N, s_nodes, s_nodes[:pad] + N])
    v_ext = np.concatenate([vals[-pad:], vals, vals[:pad]])
    return Akima1DInterpolator(s_ext, v_ext)(s_query)


def reconstruct_trajectory(ctrl, u, oversample=OVERSAMPLE):
    """
    Reconstruye la trayectoria continua.

    En lugar de interpolar las posiciones de los puntos de control (lo que cortaba
    las curvas), interpola el DESPLAZAMIENTO LATERAL u sobre la línea central
    densa y luego aplica  pᵢ = cᵢ + uᵢ·n̂ᵢ en alta resolución.

    Retorna (traj_x, traj_y).
    """
    tx, ty, _, _, _ = reconstruct_with_offset(ctrl, u, oversample)
    return tx, ty


def reconstruct_with_offset(ctrl, u, oversample=OVERSAMPLE):
    """
    Reconstruye la trayectoria continua con desplazamiento lateral.

    Interpola el desplazamiento lateral u sobre la línea central densa y luego
    aplica  pᵢ = cᵢ + uᵢ·n̂ᵢ en alta resolución.

    Retorna (traj_x, traj_y, u_f, dE, dI).
    """
    geo = ctrl["_geo"]
    idx = ctrl["_idx"]
    N = len(geo["cx"])
    base = np.arange(N)
    s = np.linspace(0, N, N * oversample, endpoint=False)
    cx = np.interp(s, base, geo["cx"], period=N)
    cy = np.interp(s, base, geo["cy"], period=N)
    nx = np.interp(s, base, geo["nx"], period=N)
    ny = np.interp(s, base, geo["ny"], period=N)
    nn = np.hypot(nx, ny); nn[nn < 1e-9] = 1e-9
    nx /= nn; ny /= nn
    dE = np.interp(s, base, geo["dE"], period=N)
    dI = np.interp(s, base, geo["dI"], period=N)
    u_f = _akima_periodic(idx.astype(float), np.asarray(u, float), N, s)
    return cx + u_f * nx, cy + u_f * ny, u_f, dE, dI

# -----------------------------------------------------------------------------
# 5. FUNCIÓN OBJETIVO Y PENALIZACIÓN
# -----------------------------------------------------------------------------

def _lap_time_vec(tx, ty):
    """Tiempo de vuelta vectorizado sobre toda la trayectoria."""
    xp, yp = np.roll(tx, 1), np.roll(ty, 1)
    xn, yn = np.roll(tx, -1), np.roll(ty, -1)
    a = np.hypot(tx - xn, ty - yn)
    b = np.hypot(xp - xn, yp - yn)
    c = np.hypot(xp - tx, yp - ty)
    area = 0.5 * np.abs(xp*(ty - yn) + tx*(yn - yp) + xn*(yp - ty))
    r = np.where(area < 1e-9, 1e9, a * b * c / (4 * area + 1e-12))
    v = np.minimum(np.sqrt(MU * G * r), VMAX)
    d = np.hypot(xn - tx, yn - ty)
    return float(np.sum(d / v))


def lap_time(traj_x, traj_y):
    """Tiempo de vuelta sobre una trayectoria dada."""
    return _lap_time_vec(traj_x, traj_y)


def trajectory_violation(ctrl, u, margin=0.0, oversample=OVERSAMPLE):
    """
    Mide cuánto se sale la trayectoria interpolada de los bordes reales.
    Retorna (violacion_total_m, violacion_maxima_m).
    """    
    _, _, u_f, dE, dI = reconstruct_with_offset(ctrl, u, oversample)
    over  = np.maximum(0.0, u_f - (dE - margin))
    under = np.maximum(0.0, -(dI - margin) - u_f)
    ov = over + under
    return float(ov.sum()), float(ov.max())


def objective_function(u, ctrl, penalty=PENALTY_WEIGHT,
                       margin=SAFETY_MARGIN, oversample=OVERSAMPLE):
    """
    Función objetivo a minimizar: tiempo de vuelta + penalización por salirse.
    El AG optimiza sobre el vector u de desplazamientos laterales en los puntos de control, 
    pero la función objetivo evalúa la trayectoria reconstruida en alta resolución.
    Retorna el tiempo de vuelta penalizado.
     - T = tiempo de vuelta sobre la trayectoria reconstruida.
     - Si penalty > 0, se añade una penalización proporcional a cuánto se sale la
         trayectoria de los bordes (medida por trajectory_violation) multiplicada por penalty.
        - margin: margen de seguridad para mantener el carro dentro de la pista (m).
        - oversample: densidad de muestreo de la trayectoria reconstruida (más = más precisa pero más lenta).

    """
    tx, ty, u_f, dE, dI = reconstruct_with_offset(ctrl, u, oversample)
    T = _lap_time_vec(tx, ty)
    if penalty > 0:
        over  = np.maximum(0.0, u_f - (dE - margin))
        under = np.maximum(0.0, -(dI - margin) - u_f)
        viol = float((over + under).sum())
        T *= (1.0 + penalty * viol)
    return T


def control_bounds(ctrl, margin=SAFETY_MARGIN):
    """
    Calcula los límites de control para cada punto de control, considerando un margen de seguridad.
    Retorna (lb, ub) donde cada uno es un array de longitud n_ctrl con los límites inferiores y superiores.
    """
    lb = -(ctrl["dI"] - margin)
    ub =  (ctrl["dE"] - margin)
    return lb, ub

# -----------------------------------------------------------------------------
# 6. UTILIDADES
# -----------------------------------------------------------------------------

def track_length(geo):
    """ 
    Longitud aproximada de la pista sumando distancias entre puntos centrales + cierre.
    Esto es útil para mostrar la longitud de cada pista en la salida, aunque no es exacto.
     - geo: geometría de la pista con campos 'cx' y 'cy'.
     - Retorna la longitud aproximada en kilómetros.
    """
    cx, cy = geo["cx"], geo["cy"]
    d = np.sqrt(np.diff(cx)**2 + np.diff(cy)**2).sum()
    closing = np.sqrt((cx[0]-cx[-1])**2 + (cy[0]-cy[-1])**2)
    return (d + closing) / 1000.0


def velocity_profile(traj_x, traj_y):
    """
    Perfil de velocidad y distancia acumulada a lo largo de la trayectoria.
    Retorna (distancias_m, velocidades_kmh). Vectorizado.
    """
    xp, yp = np.roll(traj_x, 1), np.roll(traj_y, 1)
    xn, yn = np.roll(traj_x, -1), np.roll(traj_y, -1)
    a = np.hypot(traj_x - xn, traj_y - yn)
    b = np.hypot(xp - xn, yp - yn)
    c = np.hypot(xp - traj_x, yp - traj_y)
    area = 0.5 * np.abs(xp*(traj_y - yn) + traj_x*(yn - yp) + xn*(yp - traj_y))
    r = np.where(area < 1e-9, 1e9, a * b * c / (4 * area + 1e-12))
    vels = np.minimum(np.sqrt(MU * G * r), VMAX) * 3.6
    seg = np.hypot(np.diff(traj_x), np.diff(traj_y))
    dists = np.concatenate([[0.0], np.cumsum(seg)])
    return dists, vels
