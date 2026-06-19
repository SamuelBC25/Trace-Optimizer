"""
genetic_algorithm.py — Módulo 2: Algoritmo Genético (GA)
=========================================================
Implementación completa del AG para optimización de trayectorias.

Operadores incluidos:
- Selección por torneo
- Cruce SBX (Simulated Binary Crossover)
- Mutación polinomial
Configuración flexible a través de GAConfig, con soporte para múltiples corridas
y análisis estadístico de resultados.

"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class GAConfig:
    """Parámetros del Algoritmo Genético.
    
    Punto único de configuración: modificar estos valores afecta
    directamente a main.py sin necesidad de tocar otro archivo.
    """
    pop_size:       int   = 50       # Tamaño de población
    n_generations:  int   = 200     # Número de generaciones
    pc:             float = 0.90     # Probabilidad de cruzamiento SBX
    pm:             float = None     # Probabilidad de mutación (None → 1/N)
    eta_c:          float = 5.0      # Índice de distribución SBX (Nc)
    eta_m:          float = 20.0     # Índice de distribución mutación (Nm)
    tournament_k:   int   = 2        # Tamaño del torneo
    seed:           int   = None     # Semilla aleatoria (None = aleatoria)
    verbose:        bool  = True     # Imprimir progreso
    max_evals:      int   = 10_000   # Número máximo de evaluaciones

# -----------------------------------------------------------------------------
# ESTRUCTURA DE RESULTADOS
# -----------------------------------------------------------------------------

@dataclass
class GAResult:
    """Resultado de una corrida del AG."""
    best_u:       np.ndarray            # Mejor solución encontrada
    best_fitness: float                 # Mejor tiempo de vuelta (s)
    history:      list = field(default_factory=list)  # Mejor fitness por generación
    n_evals:      int  = 0              # Evaluaciones usadas
    n_gen:        int  = 0              # Generaciones ejecutadas


# -----------------------------------------------------------------------------
# OPERADORES DEL ALGORITMO GENÉTICO
# -----------------------------------------------------------------------------

def _tournament_selection(pop, fitness, k, rng):
    """
    Selección por torneo determinístico.
    Elige k individuos al azar y retorna el de menor fitness (minimización).
    """
    candidates = rng.integers(0, len(pop), size=k)
    winner = candidates[np.argmin(fitness[candidates])]
    return pop[winner].copy()


def _sbx_crossover(p1, p2, lb, ub, eta_c, pc, rng):
    """
    Cruce SBX (Simulated Binary Crossover).
    Genera dos hijos a partir de dos padres p1 y p2.
    Cada gen se cruza con probabilidad pc, usando la distribución controlada por eta_c.
    """
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > pc:
        return c1, c2
    n = len(p1)
    for i in range(n):
        if rng.random() < 0.5:
            continue
        if abs(p1[i] - p2[i]) < 1e-10:
            continue
        u = rng.random()
        beta = (2*u)**(1/(eta_c+1)) if u <= 0.5 \
               else (1/(2*(1-u)))**(1/(eta_c+1))
        c1[i] = 0.5 * ((1 + beta)*p1[i] + (1 - beta)*p2[i])
        c2[i] = 0.5 * ((1 - beta)*p1[i] + (1 + beta)*p2[i])
    
    # Reparar dentro de los límites
    c1 = np.clip(c1, lb, ub)
    c2 = np.clip(c2, lb, ub)
    return c1, c2


def _polynomial_mutation(ind, lb, ub, eta_m, pm, rng):
    """
    Mutación polinomial.
    Cada gen de ind se muta con probabilidad pm, usando la distribución controlada por eta_m.
    """
    mutant = ind.copy()
    for i in range(len(ind)):
        if rng.random() >= pm:
            continue
        u = rng.random()
        delta = (2*u)**(1/(eta_m+1)) - 1 if u < 0.5 \
                else 1 - (2*(1-u))**(1/(eta_m+1))
        mutant[i] = np.clip(ind[i] + delta * (ub[i] - lb[i]), lb[i], ub[i])
    return mutant

# -----------------------------------------------------------------------------
# ALGORITMO GENÉTICO PRINCIPAL
# -----------------------------------------------------------------------------

def run_ga(objective, lb, ub, config=None, callback=None):
    """Ejecuta el Algoritmo Genético para minimizar la función objetivo dada.
    Args:
        objectivo: Función a minimizar (recibe un vector de control points).
        lb: Vector de límites inferiores para cada gen.
        ub: Vector de límites superiores para cada gen.
        config: Configuración del AG (GAConfig). Si es None, se usan valores por defecto.
        callback: Función opcional que se llama al final de cada generación con
                  (gen, best_fitness, best_u) para seguimiento en tiempo real.
    Returns:
        GAResult con la mejor solución encontrada, su fitness, historia de mejoras
        y estadísticas de ejecución.
    """
    if config is None:
        config = GAConfig()
    rng = np.random.default_rng(config.seed)
    n   = len(lb)
    pm  = config.pm if config.pm is not None else 1.0 / n

    max_evals = config.max_evals if config.max_evals is not None else config.n_generations * (config.pop_size - 1) + config.pop_size
    """
    Inicialización de la población.
    - Se genera una población aleatoria uniforme dentro de los límites.
    - Se asegura que el primer individuo sea el vector cero (trayectoria base).
    - Se generan algunos individuos cercanos al vector cero para acelerar la convergencia.
    - Se evalúa el fitness de toda la población inicial."""
    pop = rng.uniform(lb, ub, size=(config.pop_size, n))
    n_seed = max(1, config.pop_size // 4)
    pop[0] = np.clip(np.zeros(n), lb, ub)
    span = (ub - lb)
    for i in range(1, n_seed):
        pop[i] = np.clip(rng.normal(0.0, 0.15 * span), lb, ub)
    fitness = np.array([objective(ind) for ind in pop])
    n_evals = config.pop_size

    best_idx     = np.argmin(fitness)
    best_u       = pop[best_idx].copy()
    best_fitness = fitness[best_idx]
    history      = [best_fitness]
    gen          = 0

    if config.verbose:
        print(f"  Gen 000 | Best: {best_fitness:.4f}s | Evals: {n_evals}")

    # Bucle principal del AG: selección, cruce, mutación y reemplazo elitista.
    while n_evals < config.max_evals:
        new_pop     = []
        new_fitness = []

        # Elitismo: el mejor pasa directamente
        new_pop.append(best_u.copy())
        new_fitness.append(best_fitness)

        # Llenar nueva población con cruzamiento + mutación
        while len(new_pop) < config.pop_size:
            # Selección por torneo
            p1 = _tournament_selection(pop, fitness, config.tournament_k, rng)
            p2 = _tournament_selection(pop, fitness, config.tournament_k, rng)
            
            # Cruzamiento SBX
            c1, c2 = _sbx_crossover(p1, p2, lb, ub, config.eta_c, config.pc, rng)
            
            # Mutación polinómica
            c1 = _polynomial_mutation(c1, lb, ub, config.eta_m, pm, rng)
            c2 = _polynomial_mutation(c2, lb, ub, config.eta_m, pm, rng)

            for child in (c1, c2):
                if len(new_pop) >= config.pop_size:
                    break
                f = objective(child)
                new_pop.append(child)
                new_fitness.append(f)
                n_evals += 1
                if n_evals >= config.max_evals:
                    break
            if n_evals >= config.max_evals:
                break

        pop     = np.array(new_pop)
        fitness = np.array(new_fitness)
        gen    += 1

        # Actualizar el mejor individuo global  
        idx = np.argmin(fitness)
        if fitness[idx] < best_fitness:
            best_fitness = fitness[idx]
            best_u       = pop[idx].copy()
        history.append(best_fitness)

        if callback:
            callback(gen, best_fitness, best_u)

        if config.verbose and gen % 20 == 0:
            print(f"  Gen {gen:03d} | Best: {best_fitness:.4f}s"
                  f" | Mean: {fitness.mean():.4f}s | Evals: {n_evals}")

    if config.verbose:
        print(f"  Done — Best: {best_fitness:.4f}s | Gens: {gen} | Evals: {n_evals}")

    return GAResult(best_u=best_u, best_fitness=best_fitness,
                    history=history, n_evals=n_evals, n_gen=gen)


# -----------------------------------------------------------------------------
# EXPERIMENTO MULTI-CORRIDA
# -----------------------------------------------------------------------------

def multi_run(objective, lb, ub, config, n_runs=10, verbose=True):
    """
    Ejecuta múltiples corridas del AG para obtener estadísticas.
    Args:
        objective: Función a minimizar.
        lb: Límites inferiores.
        ub: Límites superiores.
        config: Configuración base del AG (GAConfig).
        n_runs: Número de corridas independientes a ejecutar.
        verbose: Imprimir progreso de cada corrida.
    Returns:
        Diccionario con resultados agregados:
        - "results": lista de GAResult de cada corrida
        - "best": mejor fitness encontrado entre todas las corridas
        - "mean": La media del fitness de las mejores soluciones de cada corrida
        - "worst": peor fitness encontrado entre todas las corridas
        - "std": desviación estándar del fitness de las mejores soluciones
        - "best_u": vector de control points de la mejor solución encontrada
        - "histories": lista de historias de mejora por generación para cada corrida
    """
    results = []
    fitnesses = []
    for run in range(n_runs):
        cfg = GAConfig(**{**config.__dict__, "seed": config.seed + run
                         if config.seed is not None else None, "verbose": False})
        if verbose:
            print(f"  Run {run+1}/{n_runs}...", end=" ", flush=True)
        result = run_ga(objective, lb, ub, cfg)
        results.append(result)
        fitnesses.append(result.best_fitness)
        if verbose:
            print(f"{result.best_fitness:.4f}s")

    fitnesses = np.array(fitnesses)
    best_idx  = np.argmin(fitnesses)
    return {
        "results":   results,
        "best":      fitnesses.min(),
        "mean":      fitnesses.mean(),
        "worst":     fitnesses.max(),
        "std":       fitnesses.std(),
        "best_u":    results[best_idx].best_u,
        "histories": [r.history for r in results],
    }